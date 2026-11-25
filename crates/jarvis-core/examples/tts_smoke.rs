//! Диагностика TTS на этой машине: загружается ли ZONOS3-Mini и синтезирует
//! ли звук (RMS > 0 = не тишина). Модель ищется в системной папке моделей,
//! путь можно переопределить: JARVIS_MODELS_DIR=/path/to/models
//! Запись кладётся в /tmp/jarvis_tts_smoke.wav — можно послушать.
//! Запуск: cargo run -p jarvis-core --example tts_smoke

use anyhow::Result;
use jarvis_core::tts_zonos::SpeechEngine;

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let started = std::time::Instant::now();
    let voice = std::env::var("JARVIS_SMOKE_VOICE").unwrap_or_else(|_| "jarvis".into());
    let engine = jarvis_core::tts::open_zonos_voice(&voice)?;
    println!("✅ ZONOS3 загрузился за {:.1?}", started.elapsed());

    // Ударения и омографы прогреваются синхронно: в приложении это делает
    // загрузчик TTS параллельно с движком, здесь — руками, чтобы замерить
    // и синтез с метками ударений.
    let started = std::time::Instant::now();
    jarvis_core::stress::warm_up();
    jarvis_core::omograph::warm_up();
    println!("✅ Ударитель прогрет за {:.1?}", started.elapsed());

    let text = std::env::var("JARVIS_SMOKE_TEXT")
        .unwrap_or_else(|_| "Привет! Проверка синтеза речи на этой машине.".into());
    let started = std::time::Instant::now();
    let audio = engine.speak(&text, 1.0)?;
    let secs = audio.len() as f32 / engine.sample_rate() as f32;
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    println!(
        "✅ Синтез: {:.1} сек звука за {:.1?}, RMS {:.4}{}",
        secs,
        started.elapsed(),
        rms,
        if rms > 0.001 { "" } else { " ⚠️ тишина!" }
    );

    // Потоковый синтез (JARVIS_TTS_STREAM=1): замер времени до первого
    // блока и до конца, запись для сравнения с классическим путём.
    if std::env::var("JARVIS_TTS_STREAM")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let started = std::time::Instant::now();
        let mut blocks = 0usize;
        let mut total: Vec<f32> = Vec::new();
        let mut first_at = None;
        engine.speak_streaming(&text, 1.0, &cancel, &mut |block, last| {
            if first_at.is_none() {
                first_at = Some(started.elapsed());
            }
            blocks += 1;
            total.extend_from_slice(&block);
            if last {
                let secs = total.len() as f32 / engine.sample_rate() as f32;
                println!(
                    "✅ Стрим: первый блок через {:?}, блоков {}, {:.1} сек звука за {:?}",
                    first_at.unwrap(),
                    blocks,
                    secs,
                    started.elapsed()
                );
            }
        })?;
        let path = "/tmp/jarvis_tts_stream.wav";
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: engine.sample_rate(),
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for s in &total {
            writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
        println!("📂 Стрим-запись: {path}");
    }

    // Конверт речи: огибающая по 100-мс окнам — шум плоский, речь неровная.
    let win = engine.sample_rate() as usize / 10;
    let env: Vec<f32> = audio
        .chunks(win)
        .map(|c| (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt())
        .collect();
    println!(
        "🔇 Огибающая ({} окон по 100 мс): {:?}",
        env.len(),
        env.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>()
    );

    let path = "/tmp/jarvis_tts_smoke.wav";
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: engine.sample_rate(),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for s in &audio {
        writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    println!("📂 Запись: {path}");
    Ok(())
}
