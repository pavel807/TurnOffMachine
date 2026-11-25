//! Стенд регрессии wake word: воспроизводит БОЕВОЙ тракт целиком.
//!
//! Тракт: микрофон 48 кГц → чанки кадрами cpal → `resample_audio` на КАЖДЫЙ
//! чанк отдельно (как в engine.rs) → CTC-детектор на CPU (как в бою).
//!
//! Для каждого образца печатает: сработало ли, лучший счёт, декод и причину
//! провала. Это регрессионный тест: после правок стенд должен зелёным.
//!
//! Запуск:
//!   cargo run -p jarvis-core --example wake_bench -- [шум-RMS ...]

use anyhow::{Context, Result};
use jarvis_core::stt::Stt;
use jarvis_core::wake_ctc::{CtcConfig, CtcWakeEngine};
use std::sync::{Arc, Mutex};

/// Прочитать wav любого формата (i16/i32/f32) в моно.
fn read_wav(path: &std::path::Path) -> Result<(Vec<f32>, usize)> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("открыть {}", path.display()))?;
    let spec = r.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        hound::SampleFormat::Int => r
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
    Ok((mono, spec.sample_rate as usize))
}

/// Шум с фиксированным зерном (xorshift) — воспроизводимо между прогонами.
fn add_noise(s: &mut [f32], rms: f32, seed: &mut u64) {
    if rms <= 0.0 {
        return;
    }
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    for s in s.iter_mut() {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let u = ((*seed >> 40) & 0xFF_FFFF) as f32 / 8_388_608.0 - 1.0;
        *s += u * rms;
    }
}

/// Прогнать образец через боевой тракт.
///
/// `chunk_frames` — размер чанка микрофона в кадрах 48 кГц (cpal Default
/// на CoreAudio ≈ 480, WASAPI ≈ 4096).
fn run(
    stt: &Arc<Mutex<Stt>>,
    word: &str,
    samples: &[f32],
    src_rate: usize,
    noise_rms: f32,
    chunk_frames: usize,
    config: &CtcConfig,
) -> (bool, Option<f32>, String) {
    // 48 кГц — как типичный микрофон.
    let up = jarvis_core::audio::resample_audio(samples, src_rate, 48_000);
    let mut noisy = up;
    let mut seed = 0x1234_5678_9ABC_DEF1u64;
    add_noise(&mut noisy, noise_rms, &mut seed);

    let mut det = CtcWakeEngine::new(Arc::clone(stt), word, config.clone());
    let mut best: Option<f32> = None;
    let mut best_decoded = String::new();
    let mut fired = false;
    // Тишина перед фразой: 3 с — иначе не истёк cooldown детектора.
    // Тишина ДО и ПОСЛЕ фразы: микрофон в бою никогда не обрывается, а
    // детектору нужно, чтобы слово целиком лежало в окне (окно 1.2 с).
    // Без хвоста после слова окно обрезалось и CTC декодировал только
    // начало слова — артефакт стенда, а не движка.
    let mut pending: Vec<f32> = vec![0.0; 48_000 * 3];
    let tail_pad: Vec<f32> = Vec::new();
    let _ = tail_pad;
    // Сколько сэмплов 16 кГц уже отдано детектору: без этого буфер
    // перепроцетывался целиком на каждой итерации (квадратичный рост
    // времени — стенд на 100+ образцов не заканчивался).
    let mut processed = 0usize;

    // Ресемплер с состоянием — ровно как в engine.rs.
    let mut resampler = jarvis_core::resample::Resampler::new(48_000, 16_000)
        .expect("ресемплер 48k→16k");
    // Хвост тишины после слова — как в бою (микрофон продолжает писать).
    let mut padded = noisy.clone();
    padded.extend(std::iter::repeat_n(0.0f32, 48_000 * 2));

    for chunk in padded.chunks(chunk_frames) {
        let mut c16 = Vec::new();
        resampler.process(chunk, &mut c16);
        pending.extend_from_slice(&c16);
        if pending.len() - processed < 16_000 / 10 {
            continue;
        }
        // Отдаём детектору порциями по ~85 мс — только новые сэмплы.
        let hop = 1365;
        let mut i = processed;
        while i < pending.len() {
            let end = (i + hop).min(pending.len());
            let piece = &pending[i..end];
            i = end;
            if let Some(d) = det.process(piece) {
                fired = true;
                if best.is_none_or(|b| d.score > b) {
                    best = Some(d.score);
                    best_decoded = d.decoded.clone();
                }
            }
            if let Some(diag) = det.last_diag()
                && diag.encoder_ran
                && let Some(s) = diag.score
                && best.is_none_or(|b| s > b)
            {
                best = Some(s);
                best_decoded = diag.decoded.clone();
            }
        }
        processed = pending.len();
        // Держим буфер небольшим, но не режем слова на полуслове.
        let keep = 16_000 * 2;
        if pending.len() > keep {
            pending.drain(..pending.len() - keep);
            processed = pending.len();
        }
    }
    (fired, best, best_decoded)
}

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let settings = jarvis_core::settings::Settings::load(&jarvis_core::paths::settings_file())
        .unwrap_or_default();
    let word = settings.wake_word.clone();
    let args: Vec<String> = std::env::args().collect();
    let levels: Vec<f32> = if args.len() > 1 {
        args[1..].iter().filter_map(|v| v.parse().ok()).collect()
    } else {
        vec![0.0, 0.002, 0.005, 0.01]
    };

    let mut config = CtcConfig::default();
    config.window_ms = settings.wake_window_ms.clamp(1200, 2500);
    config.threshold = settings.wake_ctc_threshold;

    let stt = Arc::new(Mutex::new(Stt::load_on(
        &jarvis_core::paths::models_dir(),
        candle_core::Device::Cpu,
    )?));

    // Позитивы: записи голоса владельца + синтез фразы ZONOS.
    let mut pos: Vec<std::path::PathBuf> = Vec::new();
    for i in 1..=8 {
        let p = std::path::PathBuf::from(format!("train/{i}.wav"));
        if p.exists() {
            pos.push(p);
        }
    }
    if let Ok(rd) = std::fs::read_dir("train/phrase") {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wav")) {
                pos.push(p);
            }
        }
    }
    pos.sort();

    // Негативы: обычная речь, которая НЕ должна будить Джарвиса.
    let mut neg: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir("train/neg") {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wav")) {
                neg.push(p);
            }
        }
    }
    neg.sort();

    println!(
        "🎯 Слово «{}», окно {} мс, порог {}, min_hits {}, уверенный {}; \n   позитивов: {}, негативов: {}",
        jarvis_core::wake_ctc::normalize_word(&word),
        config.window_ms,
        config.threshold,
        config.min_hits,
        config.strong_threshold,
        pos.len(),
        neg.len()
    );

    // --- Позитивы: должны срабатывать ---
    let mut pos_total = 0usize;
    let mut pos_passed = 0usize;
    println!("\n── ПОЗИТИВЫ (должны сработать) ──");
    for path in &pos {
        let (samples, rate) = read_wav(path)?;
        let mut row = Vec::new();
        for &lvl in &levels {
            let (fired, best, _decoded) = run(&stt, &word, &samples, rate, lvl, 480, &config);
            pos_total += 1;
            if fired {
                pos_passed += 1;
            }
            row.push(format!(
                "{}{}",
                if fired { "✔" } else { "✘" },
                best.map(|b| format!("({b:+.2})")).unwrap_or_default()
            ));
        }
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        println!("  {name:<28} {}", row.join("  "));
    }

    // --- Негативы: не должны срабатывать ---
    let mut neg_total = 0usize;
    let mut neg_false = 0usize;
    println!("\n── НЕГАТИВЫ (не должны сработать) ──");
    for path in &neg {
        let (samples, rate) = read_wav(path)?;
        let mut row = Vec::new();
        for &lvl in &levels {
            let (fired, best, _decoded) = run(&stt, &word, &samples, rate, lvl, 480, &config);
            neg_total += 1;
            if fired {
                neg_false += 1;
            }
            row.push(format!(
                "{}{}",
                if fired { "✘" } else { "·" },
                best.map(|b| format!("({b:+.2})")).unwrap_or_default()
            ));
        }
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        let flag = if row.iter().any(|r| r.starts_with('✘')) { "  ← ЛОЖНОЕ" } else { "" };
        println!("  {name:<28} {}{flag}", row.join("  "));
    }

    println!("\n✅ Позитивы: {pos_passed}/{pos_total} сработали");
    println!("❌ Ложные срабатывания: {neg_false}/{neg_total}");
    Ok(())
}
