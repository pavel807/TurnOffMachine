//! Покадровая диагностика wake-детектора: что именно модель слышит в
//! каждом окне и почему фраза не срабатывает.
//!
//! Запуск:
//!   cargo run -p jarvis-core --example wake_diag -- <файл.wav> [слово]
//!
//! Печатает построчно: номер окна, RMS, декод CTC, счёт совпадения
//! с фразой и вердикт (сработало / нет). VAD в детекторе нет — энкодер
//! считается на каждом hop (репорт: VAD глотал начало «Джарвиса»).
//!
//! Отличие от `wake_live_sim`: детектор считается на CPU — ровно как в
//! бою (см. `wake::create_engine`), и печатается КАЖДОЕ окно, а не лучшее.

use anyhow::{Context, Result};
use jarvis_core::stt::Stt;
use jarvis_core::wake_ctc::{CtcConfig, CtcWakeEngine};
use jarvis_core::{audio::resample_audio, settings::Settings};
use std::sync::{Arc, Mutex};

/// Прочитать wav любого формата (i16/i32/f32) и привести к моно 16 кГц.
fn read_wav_16k(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("открыть {path}"))?;
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.unwrap_or(0) as f32 / (1i32 << (spec.bits_per_sample - 1)) as f32)
            .collect(),
    };
    let mono: Vec<f32> = if spec.channels == 1 {
        raw
    } else {
        raw.chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / c.len() as f32)
            .collect()
    };
    Ok(resample_audio(&mono, spec.sample_rate as usize, 16_000))
}

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let args: Vec<String> = std::env::args().collect();
    let Some(wav) = args.get(1) else {
        anyhow::bail!("использование: wake_diag <файл.wav> [слово]");
    };
    let settings = Settings::load(&jarvis_core::paths::settings_file()).unwrap_or_default();
    let word = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| settings.wake_word.clone());
    let normalized = jarvis_core::wake_ctc::normalize_word(&word);

    let samples = read_wav_16k(wav)?;
    let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len().max(1) as f32).sqrt();
    println!(
        "🎤 {wav}: слово «{word}» → «{normalized}», {:.2} с, RMS {rms:.4}",
        samples.len() as f32 / 16_000.0
    );

    // CPU — как в бою: Metal-энкодер не тянет окно в реальном времени.
    let stt = Arc::new(Mutex::new(
        Stt::load_on(&jarvis_core::paths::models_dir(), candle_core::Device::Cpu)?,
    ));

    let mut config = CtcConfig::default();
    config.window_ms = settings.wake_window_ms.clamp(1200, 2500);
    config.threshold = settings.wake_ctc_threshold;
    println!(
        "⚙️ окно {} мс, хоп {} мс, порог {}, min_hits {}, сглаживание {}",
        config.window_ms, config.hop_ms, config.threshold, config.min_hits, config.smoothing
    );

    // Тишина по краям, чтобы окно наполнилось до фразы (как в бою).
    // Перед фразой — 3 с: столько нужно, чтобы истёк cooldown детектора
    // (он считается от последнего reset, а не от последнего срабатывания).
    let pad = vec![0.0f32; 16_000 * 3];
    let mut padded = pad.clone();
    padded.extend_from_slice(&samples);
    padded.extend_from_slice(&pad);

    let mut det = CtcWakeEngine::new(Arc::clone(&stt), &word, config.clone());
    let mut n = 0usize;
    let mut fired = false;
    // Хоп-батч: движок отдаёт звук чанками; здесь — по 85 мс, как в бою.
    for chunk in padded.chunks(1365) {
        n += 1;
        let hit = det.process(chunk);
        let diag = det.last_diag();
        if let Some(d) = diag {
            if d.encoder_ran {
                println!(
                    "  окно #{n:>3}  rms {:.4}  декод «{}»  счёт {}  {:?}",
                    d.rms,
                    d.decoded,
                    d.score
                        .map(|s| format!("{s:+.3}"))
                        .unwrap_or_else(|| "  —  ".into()),
                    if hit.is_some() { "СРАБОТАЛО" } else { "" }
                );
            }
        }
        if hit.is_some() {
            fired = true;
        }
    }
    println!(
        "{}",
        if fired {
            "✅ СРАБОТАЛО"
        } else {
            "❌ НЕ СРАБОТАЛО"
        }
    );
    Ok(())
}
