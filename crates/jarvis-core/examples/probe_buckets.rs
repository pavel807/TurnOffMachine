//! Перебор бакетов условий (громкость, f_max) с метрикой похожести голоса:
//! синтез → спикер-энкодер → косинус к целевому эмбеддингу. Бакеты в бою
//! захардкожены «unknown» — зонд ищет, есть ли значение ближе к референсу.
//!
//! USAGE: probe_buckets <голос.npy> [слова-фраза]
//! Перебор: громкость 0..count0, f_max 0..count1 (темп — unknown).

use anyhow::{Context, Result};

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let mut args = std::env::args().skip(1);
    let voice_npy = args.next().context("Укажи npy голоса")?;
    let target: Vec<f32> = read_npy(&voice_npy)?;

    let dir = jarvis_core::model_fetch::zonos3_dir();
    let gguf = jarvis_core::tts_zonos::GgufFile::open(&dir.join("backbone-ptq.gguf"))?;
    let mut engine = jarvis_core::tts_zonos::ZonosEngine::load(&gguf, candle_core::Device::Cpu)?;
    let dec = jarvis_core::tts_zonos::ZacDecoder::open(&dir.join("zac-keep2.gguf"), &candle_core::Device::Cpu)?;
    let spk = jarvis_core::tts_zonos::SpeakerEncoder::open(&dir.join("spk-encoder.gguf"))?;

    let counts: Vec<usize> = engine.cfg.condition_bucket_counts.clone();
    println!("бакеты: громкость {}, f_max {}, темп {}", counts[0], counts[1], counts[2]);

    let unknown = [
        engine.cfg.unknown_bucket(0),
        engine.cfg.unknown_bucket(1),
        engine.cfg.unknown_bucket(2),
    ];
    let text = "Все системы работают в штатном режиме.";
    let mut sampling = jarvis_core::tts_zonos::Sampling::default();
    sampling.temperature = 0.6;
    sampling.seed = 1;

    let cosine = |audio: &[f32], target: &[f32]| -> f32 {
        let back = spk.embedding(audio, 44100).expect("эмбеддинг");
        let dot: f32 = back.iter().zip(target).map(|(x, y)| x * y).sum();
        let na = back.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = target.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb).max(1e-9)
    };

    // Базовая точка: все unknown.
    let out = engine.synthesize(text, Some(&target), unknown, &sampling, 1.0, Some(&dec))?;
    println!("unknown/unknown: {:.4}", cosine(&out.audio, &target));

    // Кандидатские комбинации: громкость 8 с f_max 0/2 + вторая фраза.
    let text2 = "Обнаружено входящее сообщение, сэр.";
    for buckets in [[8usize, 0, unknown[2]], [8, 2, unknown[2]], [8, 9, unknown[2]]] {
        let out = engine.synthesize(text, Some(&target), buckets, &sampling, 1.0, Some(&dec))?;
        let c1 = cosine(&out.audio, &target);
        let out2 = engine.synthesize(text2, Some(&target), buckets, &sampling, 1.0, Some(&dec))?;
        let c2 = cosine(&out2.audio, &target);
        println!("{:?}: фраза1 {c1:.4} фраза2 {c2:.4}", buckets);
    }

    // Перебор громкости при остальных unknown.
    for b in 0..counts[0] {
        let buckets = [b, unknown[1], unknown[2]];
        let out = engine.synthesize(text, Some(&target), buckets, &sampling, 1.0, Some(&dec))?;
        println!("громкость {b}: {:.4}", cosine(&out.audio, &target));
    }

    // Перебор f_max при остальных unknown.
    for b in 0..counts[1] {
        let buckets = [unknown[0], b, unknown[2]];
        let out = engine.synthesize(text, Some(&target), buckets, &sampling, 1.0, Some(&dec))?;
        println!("f_max {b}: {:.4}", cosine(&out.audio, &target));
    }
    Ok(())
}

fn read_npy(path: &str) -> Result<Vec<f32>> {
    let data = std::fs::read(path)?;
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
    Ok(data[off..off + n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
