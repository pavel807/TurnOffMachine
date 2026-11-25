//! Симуляция живого пути wake word: файл 16 кГц → апсемпл до 48 кГц →
//! broadband-шум (как у реального микрофона) → чанки по 4096 сэмплов →
//! ресемплинг 48k→16k тем же `resample_audio`, что в движке → боевой
//! `CtcWakeEngine` с настройками из settings.toml.
//!
//! Зачем: wake_phrase_check подаёт в детектор чистый 16k-файл и мимо
//! проходит весь путь, где детектор мог оглохнуть — ресемплинг без
//! антиалиасинга (8–24 кГц микрофона складываются в речевую полосу),
//! дробление на чанки, окно из настроек. Этот пример закрывает разрыв.
//!
//! Запуск:
//!   wake_live_sim <файл.wav> [уровень-шума]
//! Уровень шума — RMS добавляемого белого шума (0 — чистый путь).
//! По умолчанию перебираются реалистичные уровни: 0 / 0.002 / 0.005 / 0.01.

use anyhow::{Context, Result};
use jarvis_core::stt::Stt;
use jarvis_core::wake_ctc::{CtcConfig, CtcWakeEngine};
use jarvis_core::{audio::resample_audio, settings::Settings};
use std::sync::{Arc, Mutex};

fn read_wav_16k(path: &str) -> Result<(Vec<f32>, usize)> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("открыть {path}"))?;
    let spec = reader.spec();
    // Записи бывают и Float32 (train/*.wav — 44.1 кГц float), и целочисленные.
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
        }
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.unwrap_or(0) as f32 / (1i32 << (spec.bits_per_sample - 1)) as f32)
            .collect(),
    };
    let mono: Vec<f32> = if spec.channels == 1 {
        raw
    } else {
        raw.chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    };
    let mono = resample_audio(&mono, spec.sample_rate as usize, 16_000);
    Ok((mono, spec.sample_rate as usize))
}

/// Апсемпл до 48 кГц, добавить шум по всей полосе (до 24 кГц — то, что
/// реальный микрофон снимает, а линейный ресемплинг без фильтра сложит
/// в речевую полосу), нарезать на чанки по 4096 сэмплов (как cpal с
/// BufferSize::Fixed(4096) на 48 кГц) и прогнать через боевой ресемплинг.
fn simulate_live_chunked(samples_16k: &[f32], noise_rms: f32, chunk_frames: usize) -> Vec<f32> {
    let up48 = resample_audio(samples_16k, 16_000, 48_000);
    let mut noisy = up48;
    if noise_rms > 0.0 {
        let mut state = 0x2545F4914F6CDD1Du64;
        for s in noisy.iter_mut() {
            // xorshift — быстрый «белый» шум без внешних крейтов: берём 24
            // старших бита состояния → равномерно в [−1, 1).
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = ((state >> 40) & 0xFF_FFFF) as f32 / 8_388_608.0 - 1.0;
            *s += u * noise_rms;
        }
    }
    let mut out = Vec::with_capacity(samples_16k.len());
    for chunk in noisy.chunks(chunk_frames) {
        out.extend(resample_audio(chunk, 48_000, 16_000));
    }
    out
}

fn run_detector(
    stt: &Arc<Mutex<Stt>>,
    word: &str,
    audio: &[f32],
    window_ms: usize,
    threshold: f32,
    min_hits: usize,
) -> Result<(bool, Option<f32>, String, String)> {
    let mut config = CtcConfig::default();
    config.window_ms = window_ms.clamp(1200, 2500).max(config.window_ms);
    config.threshold = threshold;
    config.min_hits = min_hits;
    let mut det = CtcWakeEngine::new(Arc::clone(stt), word, config);
    // Кормим теми же кусками, что движок: по чанку ~85 мс 16k.
    let hop = 1365;
    let mut best: Option<f32> = None;
    let mut best_decoded = String::new();
    let mut fired = false;
    let mut windows_ran = 0usize;
    let mut encoder_ran = 0usize;
    let mut last_diag_text = String::new();
    for chunk in audio.chunks(hop) {
        if let Some(d) = det.process(chunk) {
            fired = true;
            if best.is_none_or(|b| d.score > b) {
                best = Some(d.score);
                best_decoded = d.decoded;
            }
        }
        if let Some(diag) = det.last_diag() {
            if diag.encoder_ran {
                encoder_ran += 1;
                if let Some(s) = diag.score
                    && best.is_none_or(|b| s > b)
                {
                    best = Some(s);
                    best_decoded = diag.decoded.clone();
                }
            }
            windows_ran += 1;
            last_diag_text = format!(
                "rms {:.4}, окон {}, энкодер {}",
                diag.rms, windows_ran, encoder_ran
            );
        }
    }
    Ok((fired, best, best_decoded, last_diag_text))
}

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let args: Vec<String> = std::env::args().collect();
    let Some(wav) = args.get(1) else {
        anyhow::bail!("использование: wake_live_sim <файл.wav> [уровень-шума]");
    };
    let (samples, _src_rate) = read_wav_16k(wav)?;
    let settings = Settings::load(&jarvis_core::paths::settings_file()).unwrap_or_default();
    let word = crate_word(&settings.wake_word);
    // Окно (3-й арг) и min_hits (4-й арг) переопределяются для сравнения
    // режимов: 1200 vs 2500, min_hits 2 vs 1.
    let window_ms: usize = args
        .get(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(settings.wake_window_ms);
    let min_hits: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(2);
    // Размер чанка 48k-потока для ресемплинга (6-й арг): 4096 — cpal Fixed,
    // 512 — CoreAudio Default. Швы ресемплинга тем чаще, чем меньше чанк.
    let chunk_frames: usize = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let threshold = settings.wake_ctc_threshold;
    println!(
        "⚙️ слово «{word}», окно {window_ms} мс, порог {threshold}, min_hits {min_hits}, чанк {chunk_frames}"
    );

    // Живая фраза окружена тишиной комнаты; паддинга должно хватить, чтобы
    // окно детектора наполнилось ДО фразы: окно + 1 с с каждой стороны.
    let pad_samples = 16_000 * (window_ms + 1000) / 1000;
    let pad = vec![0.0f32; pad_samples];
    let mut padded = pad.clone();
    padded.extend_from_slice(&samples);
    padded.extend_from_slice(&pad);
    let samples = padded;
    println!(
        "📄 {wav}: {} сэмплов 16k ({:.2} с, с тишиной по краям)",
        samples.len(),
        samples.len() as f32 / 16_000.0
    );

    let stt = Arc::new(Mutex::new(Stt::load(&jarvis_core::paths::models_dir())?));

    let levels: Vec<f32> = match args.get(2).and_then(|v| v.parse().ok()) {
        Some(v) => vec![v],
        None => vec![0.0, 0.002, 0.005, 0.01],
    };
    println!(
        "{:>10}  {:>6}  {:>7}  {}",
        "шум-RMS", "сработал", "счёт", "декод"
    );
    for level in levels {
        let live = simulate_live_chunked(&samples, level, chunk_frames);
        let (fired, best, decoded, diag) =
            run_detector(&stt, &word, &live, window_ms, threshold, min_hits)?;
        println!(
            "{:>10.3}  {:>6}  {:>7}  {}",
            level,
            if fired { "ДА" } else { "нет" },
            best.map(|b| format!("{b:.3}"))
                .unwrap_or_else(|| "—".into()),
            decoded
        );
        if best.is_none() {
            println!("            диаг: {diag}");
        }
    }
    Ok(())
}

fn crate_word(word: &str) -> String {
    jarvis_core::wake_ctc::normalize_word(word)
}
