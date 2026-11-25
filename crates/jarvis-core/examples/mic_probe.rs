//! Боевой тракт wake word: микрофон → ресемплинг → CTC-детектор.
//!
//! Это ровно то, что делает движок, но без LLM и TTS — чтобы отделить
//! «детектор не слышит» от «движок ломает состояние». Звук с микрофона
//! пишется в WAV (16 кГц mono) — его можно прослушать и сравнить с тем,
//! что слышит детектор.
//!
//! Запуск:
//!   cargo run -p jarvis-core --example mic_probe -- [секунды] [файл.wav]
//!
//! Во время работы скажи активационное слово. Пример печатает декод CTC по
//! каждому окну и вердикт в конце.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use jarvis_core::stt::Stt;
use jarvis_core::wake_ctc::{CtcConfig, CtcWakeEngine};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

fn write_wav(path: &std::path::Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let args: Vec<String> = std::env::args().collect();
    let secs: f64 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(20.0);
    let dump = args.get(2).cloned();

    let settings = jarvis_core::settings::Settings::load(&jarvis_core::paths::settings_file())
        .unwrap_or_default();
    let word = settings.wake_word.clone();

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("нет устройства ввода по умолчанию")?;
    let name = device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "?".into());
    let sup = device.default_input_config()?;
    let ch = sup.channels() as usize;
    let declared = sup.sample_rate() as usize;
    println!("🎙️ Микрофон: «{name}», {ch} каналов, {declared} Гц");

    let captured: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let cap_cb = Arc::clone(&captured);
    let counter = Arc::new(AtomicU64::new(0));
    let cb_counter = Arc::clone(&counter);
    let ch_cb = ch;
    let mut cfg: cpal::StreamConfig = sup.into();
    cfg.buffer_size = cpal::BufferSize::Default;

    let downmix = move |d: &[f32]| -> Vec<f32> {
        d.chunks(ch_cb.max(1))
            .map(|c| c.iter().sum::<f32>() / c.len() as f32)
            .collect()
    };
    let built = match sup.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            cfg,
            move |d: &[f32], _| {
                cb_counter.fetch_add((d.len() / ch_cb.max(1)) as u64, Ordering::Relaxed);
                let mono = downmix(d);
                cap_cb.lock().unwrap_or_else(|e| e.into_inner()).extend_from_slice(&mono);
            },
            |e| eprintln!("ошибка потока: {e}"),
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            cfg,
            move |d: &[i16], _| {
                cb_counter.fetch_add((d.len() / ch_cb.max(1)) as u64, Ordering::Relaxed);
                let mono: Vec<f32> = d
                    .chunks(ch_cb.max(1))
                    .map(|c| {
                        c.iter()
                            .map(|&x| x as f32 / 32768.0)
                            .sum::<f32>()
                            / c.len() as f32
                    })
                    .collect();
                cap_cb.lock().unwrap_or_else(|e| e.into_inner()).extend_from_slice(&mono);
            },
            |e| eprintln!("ошибка потока: {e}"),
            None,
        ),
        f => anyhow::bail!("неподдерживаемый формат микрофона: {f:?}"),
    };
    let stream = built?;
    stream.play()?;
    println!(
        "⏳ Слушаю {secs:.0} с — скажите «{word}». Слово: «{}»",
        jarvis_core::wake_ctc::normalize_word(&word)
    );

    // Детектор — на CPU, как в бою.
    let stt = Arc::new(Mutex::new(Stt::load_on(
        &jarvis_core::paths::models_dir(),
        candle_core::Device::Cpu,
    )?));
    let mut config = CtcConfig::default();
    config.window_ms = settings.wake_window_ms.clamp(1200, 2500);
    config.threshold = settings.wake_ctc_threshold;
    let mut det = CtcWakeEngine::new(Arc::clone(&stt), &word, config.clone());

    let t0 = std::time::Instant::now();
    let mut last_frames = 0u64;
    let mut fired = false;
    let mut n = 0usize;
    let mut processed = 0usize;
    while t0.elapsed().as_secs_f64() < secs {
        std::thread::sleep(std::time::Duration::from_millis(60));
        let frames = counter.load(Ordering::Relaxed);
        let new_frames = frames - last_frames;
        last_frames = frames;
        if new_frames == 0 {
            continue;
        }
        let native: Vec<f32> = captured.lock().unwrap_or_else(|e| e.into_inner()).clone();
        // Тот же ресемплинг, что в движке (engine.rs) — но по НАКОПЛЕННОМУ
        // буферу, иначе на швах чанков теряется дробная часть сэмпла.
        let chunk16 = jarvis_core::audio::resample_audio(&native, declared, 16_000);
        // В детектор — только новые сэмплы.
        let fresh = &chunk16[processed..];
        processed = chunk16.len();
        if fresh.is_empty() {
            continue;
        }
        n += 1;
        let chunk = fresh.to_vec();
        let hit = det.process(&chunk);
        if let Some(diag) = det.last_diag()
            && diag.encoder_ran
        {
            println!(
                "  rms {:.4}  пол {:.4}  голос {:<5}  декод «{}»  счёт {}",
                diag.rms,
                diag.noise_floor,
                diag.voiced,
                diag.decoded,
                diag.score
                    .map(|s| format!("{s:+.3}"))
                    .unwrap_or_else(|| "  —  ".into()),
            );
        }
        if hit.is_some() {
            fired = true;
            println!("  >>> СРАБОТАЛО");
        }
    }

    let native = captured.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let mono16 = jarvis_core::audio::resample_audio(&native, declared, 16_000);
    if let Some(path) = dump {
        let p = std::path::PathBuf::from(&path);
        write_wav(&p, &mono16)?;
        println!("💾 Записано: {}", p.display());
    }
    let rms = (mono16.iter().map(|x| x * x).sum::<f32>() / mono16.len().max(1) as f32).sqrt();
    println!(
        "{} (RMS записи {rms:.4}, {} с)",
        if fired { "✅ СРАБОТАЛО" } else { "❌ НЕ СРАБОТАЛО" },
        mono16.len() as f32 / 16_000.0
    );
    Ok(())
}
