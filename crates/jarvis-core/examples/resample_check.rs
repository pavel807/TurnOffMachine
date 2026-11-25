//! Сравнение двух путей приведения звука к 16 кГц: прямой (файл → 16 кГц)
//! и «через микрофон» (файл → 48 кГц → потоковый ресемплер → 16 кГц).
//!
//! Зачем: стенд wake_bench показал, что через путь «через микрофон» фраза
//! находится, но срабатывает реже. Нужно понять, что именно теряется:
//! уровень, спектр или фаза.

use anyhow::Result;
use jarvis_core::stt::Stt;
use jarvis_core::wake_ctc::{CtcConfig, CtcWakeEngine};
use std::sync::{Arc, Mutex};

fn read_wav(path: &str) -> Result<(Vec<f32>, usize)> {
    let mut r = hound::WavReader::open(path)?;
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

fn rms(s: &[f32]) -> f32 {
    (s.iter().map(|x| x * x).sum::<f32>() / s.len().max(1) as f32).sqrt()
}

/// Путь «через микрофон»: 48 кГц, чанки по 480 кадров, потоковый ресемплер.
fn via_mic(samples: &[f32], src_rate: usize) -> Vec<f32> {
    let up = jarvis_core::audio::resample_audio(samples, src_rate, 48_000);
    let mut resampler = jarvis_core::resample::Resampler::new(48_000, 16_000).expect("ресемплер");
    let mut out = Vec::new();
    for chunk in up.chunks(480) {
        let mut c = Vec::new();
        resampler.process(chunk, &mut c);
        out.extend_from_slice(&c);
    }
    out
}

fn run(stt: &Arc<Mutex<Stt>>, word: &str, audio: &[f32]) -> (bool, Vec<String>, Vec<Option<f32>>) {
    let mut config = CtcConfig::default();
    config.window_ms = 1200;
    let mut det = CtcWakeEngine::new(Arc::clone(stt), word, config);
    let mut fired = false;
    let mut decodes = Vec::new();
    let mut scores = Vec::new();
    let mut padded = vec![0.0f32; 16_000 * 3];
    padded.extend_from_slice(audio);
    for chunk in padded.chunks(1365) {
        if det.process(chunk).is_some() {
            fired = true;
        }
        if let Some(d) = det.last_diag()
            && d.encoder_ran
        {
            decodes.push(d.decoded.clone());
            scores.push(d.score);
        }
    }
    (fired, decodes, scores)
}

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "train/2.wav".into());
    let settings = jarvis_core::settings::Settings::load(&jarvis_core::paths::settings_file())
        .unwrap_or_default();
    let word = settings.wake_word.clone();

    let (samples, rate) = read_wav(&path)?;
    let direct = jarvis_core::audio::resample_audio(&samples, rate, 16_000);
    let mic = via_mic(&samples, rate);

    println!("📄 {path}: {rate} Гц → 16 кГц");
    println!("   прямой:  len {} rms {:.4}", direct.len(), rms(&direct));
    println!("   через 48k: len {} rms {:.4}", mic.len(), rms(&mic));

    // Корреляция по перекрывающейся части.
    let n = direct.len().min(mic.len()).saturating_sub(4_000);
    if n > 1_000 {
        let mut num = 0.0;
        let mut da = 0.0;
        let mut db = 0.0;
        for i in 0..n {
            let a = direct[2_000 + i];
            let b = mic[2_000 + i];
            num += a * b;
            da += a * a;
            db += b * b;
        }
        println!("   корреляция: {:.4}", num / (da * db).sqrt());
    }

    let stt = Arc::new(Mutex::new(Stt::load_on(
        &jarvis_core::paths::models_dir(),
        candle_core::Device::Cpu,
    )?));

    for (name, audio) in [("прямой", &direct), ("через 48k", &mic)] {
        let (fired, decodes, scores) = run(&stt, &word, audio);
        println!(
            "\n🧪 {name}: сработало = {fired}, окон = {}",
            decodes.len()
        );
        for (i, (d, s)) in decodes.iter().zip(scores.iter()).enumerate() {
            if s.is_some() || !d.is_empty() {
                println!(
                    "   #{i:>3} «{d}» {}",
                    s.map(|v| format!("{v:+.3}")).unwrap_or_default()
                );
            }
        }
    }
    Ok(())
}
