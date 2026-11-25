//! Реальный конвейер озвучки без колонок: боевые normalize → split_speech →
//! speak_streaming с ASR-проверкой (как в плеере Tauri). Пишет WAV-куски в
//! /tmp/jarvis_real_*.wav — можно прослушать ровно то, что услышит
//! пользователь, и меряет TTFA (до первого звука), время синтеза и косинус
//! голоса к референсу по каждому куску.
//!
//! USAGE: cargo run --example tts_real -p jarvis-core -- [текст]
//! Текст по умолчанию — реалистичный ответ Джарвиса с цифрами и именем.

use anyhow::{Context, Result};
use jarvis_core::tts::open_zonos_voice;
use jarvis_core::tts_zonos::SpeechEngine;

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let voice_name = std::env::var("JARVIS_SMOKE_VOICE").unwrap_or_else(|_| "jarvis".into());
    let engine = open_zonos_voice(&voice_name)?;

    let default_text = "Сэр, все системы работают в штатном режиме. Курс установлен, скорость \
максимальная. Обнаружено входящее сообщение от полковника Флэнегана, содержащее семнадцать \
документов и три приложения. Рекомендую ознакомиться в ближайшее время. Также напомню: в \
половине восьмого у вас встреча с представителями Stark Industries.";
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| default_text.into());

    // Боевой конвейер плеера: нормализация → куски ≤180 симв.
    let text = jarvis_core::tts_normalizer::normalize(&text);
    let chunks = jarvis_core::tts::split_speech(&text, 180);
    println!(
        "текст {} симв → {} куска/кусков",
        text.chars().count(),
        chunks.len()
    );

    // Эмбеддинг референса для косинуса.
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let target: Vec<f32> = {
        let path = dir.join(format!("voices/{voice_name}.embedding.npy"));
        let data = std::fs::read(&path).context("нет эмбеддинга голоса")?;
        let hlen = u16::from_le_bytes([data[8], data[9]]) as usize;
        let off = 10 + hlen;
        let header = String::from_utf8_lossy(&data[10..10 + hlen]).to_string();
        let n: usize = header
            .split("'shape':")
            .nth(1)
            .and_then(|s| s.split('(').nth(1))
            .unwrap_or("")
            .split(|c| c == ',' || c == ')')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .product();
        data[off..off + n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let spk = jarvis_core::tts_zonos::SpeakerEncoder::open(&dir.join("spk-encoder.gguf"))?;

    let cancel = std::sync::atomic::AtomicBool::new(false);
    let mut first_audio: Option<std::time::Duration> = None;
    let started = std::time::Instant::now();
    let mut total_audio = 0f32;
    let mut chunk_index = 0usize;

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_started = std::time::Instant::now();
        let mut audio: Vec<f32> = Vec::new();
        let mut last_flag = false;
        engine.speak_streaming(chunk, 1.0, &cancel, &mut |block, last| {
            if first_audio.is_none() {
                first_audio = Some(chunk_started.elapsed());
                println!(
                    "🔊 TTFA (кусок 1): {:.2} с",
                    first_audio.unwrap().as_secs_f32()
                );
            }
            last_flag = last;
            audio.extend_from_slice(&block);
        })?;
        let chunk_secs = chunk_started.elapsed().as_secs_f32();
        let audio_secs = audio.len() as f32 / engine.sample_rate() as f32;
        total_audio += audio_secs;

        // Косинус голоса куска к референсу.
        let back = spk.embedding(&audio, engine.sample_rate())?;
        let dot: f32 = back.iter().zip(&target).map(|(x, y)| x * y).sum();
        let na = back.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = target.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb).max(1e-9);

        let path = format!("/tmp/jarvis_real_{i}.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: engine.sample_rate(),
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec)?;
        for s in &audio {
            writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
        println!(
            "кусок {i} ({}): {} симв → {:.1} с аудио за {:.1} с (косинус {cos:.3}), {}",
            if last_flag {
                "последний"
            } else {
                "..."
            },
            chunk.chars().count(),
            audio_secs,
            chunk_secs,
            path
        );
        chunk_index += 1;
    }

    println!(
        "ИТОГО: кусков {chunk_index}, аудио {total_audio:.1} с, полный конвейер {:.1} с, TTFA {:.2} с",
        started.elapsed().as_secs_f32(),
        first_audio.map(|t| t.as_secs_f32()).unwrap_or(0.0)
    );
    Ok(())
}
