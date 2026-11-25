//! Стенд синтеза: скорость и точность ZONOS3 при разных параметрах.
//!
//! Печатает по каждой попытке: сходство сказанного с текстом (её же
//! считает боевая самопроверка) и время. Нужен, чтобы выбрать температуру
//! и число попыток: сейчас первая попытка часто искажает слова, и ответ
//! «Синтез: 3.2 сек звука за 12.0s» — это четыре пересборки подряд.
//!
//! Запуск: cargo run -p jarvis-core --example tts_sweep

use anyhow::Result;
use jarvis_core::tts_zonos::{Sampling, SpeechEngine};

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let text = std::env::var("JARVIS_SMOKE_TEXT").unwrap_or_else(|_| {
        "Привет! Проверка синтеза речи на этой машине.".into()
    });

    let started = std::time::Instant::now();
    let mut voice = jarvis_core::tts::open_zonos_voice("jarvis")?;
    println!("✅ ZONOS3 загрузился за {:.1?}", started.elapsed());

    let temperatures: Vec<f32> = std::env::var("JARVIS_SWEEP_TEMP")
        .ok()
        .and_then(|v| {
            let parsed: Vec<f32> = v
                .split(',')
                .filter_map(|s| s.trim().parse::<f32>().ok())
                .collect();
            if parsed.is_empty() { None } else { Some(parsed) }
        })
        .unwrap_or_else(|| vec![0.75, 0.6, 0.45, 0.3]);
    let cfg_scale: f32 = std::env::var("JARVIS_SWEEP_CFG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.5);

    for temp in temperatures {
        voice.set_sampling(Sampling {
            temperature: temp,
            ..Sampling::default()
        });
        voice.set_cfg_scale(cfg_scale);
        let started = std::time::Instant::now();
        let audio = voice.speak(&text, 1.0)?;
        let secs = audio.len() as f32 / voice.sample_rate() as f32;
        println!(
            "🌡 temp {temp:.2} cfg {cfg_scale:.1}: {secs:.1} с звука за {:.1?} (×{:.1} к реальному времени)",
            started.elapsed(),
            started.elapsed().as_secs_f32() / secs.max(0.1)
        );
    }
    Ok(())
}
