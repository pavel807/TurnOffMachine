//! Генерация НЕГАТИВНОГО корпуса для wake word: обычная речь, которая НЕ
//! должна будить Джарвиса.
//!
//! Зачем: без «чужих» записей нельзя измерить ложные срабатывания. Мягкий
//! матчер легко начинает ловить «сервис» вместо «Джарвис» — и это не видно
//! по позитивам.
//!
//! Запуск:
//!   cargo run -p jarvis-core --example wake_neg_gen -- [папка]
//!
//! Пишет wav 16 кГц mono в `train/neg` (по умолчанию).

use anyhow::{Context, Result};
use jarvis_core::tts_zonos::{Sampling, SpeechEngine, ZonosVoice};

/// Обычная речь. Первые строки — слова, звучащие похоже на «Джарвис»
/// («сервис», «заря», «червь»), остальные — типичные команды и фразы.
const NEGATIVES: &[&str] = &[
    "сервис",
    "заря",
    "тяжёлый",
    "червь",
    "паритет",
    "машина",
    "джамбо",
    "джунгли",
    "джаффи",
    "чародей",
    "привет",
    "погода",
    "что делаешь",
    "открой музыку",
    "поставь таймер",
    "сколько времени",
    "выключи свет",
    "напомни купить хлеб",
    "работа невозможна",
    "соберись",
    "принеси документы",
    "зарплата",
    "сберди",
    "вернись домой",
    "переведи на русский",
    "который час",
    "старт",
    "стоп",
    "вперёд",
];

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
    let dir = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("train/neg"));
    std::fs::create_dir_all(&dir).with_context(|| format!("создать {}", dir.display()))?;

    let mut engine: ZonosVoice =
        jarvis_core::tts::open_zonos_voice("jarvis").context("ZONOS3 недоступен")?;

    let mut written = 0usize;
    for (i, phrase) in NEGATIVES.iter().enumerate() {
        // По три «диктора» на фразу: разные тембры и темпы.
        for (k, (speed, seed)) in [
            (0.95f32, 11u32),
            (1.0f32, 37u32),
            (1.06f32, 71u32),
        ]
        .into_iter()
        .enumerate()
        {
            engine.set_sampling(Sampling {
                seed,
                ..Sampling::default()
            });
            let Ok(audio) = engine.speak(phrase, speed) else {
                continue;
            };
            let mono = jarvis_core::audio::resample_audio(&audio, 44_100, 16_000);
            let name = format!("neg_{i:02}_{k}_{}.wav", sanitize(phrase));
            write_wav(&dir.join(name), &mono)?;
            written += 1;
        }
    }
    println!("✅ Записано {written} негативных образцов в {}", dir.display());
    Ok(())
}

/// Имя файла из фразы: буквы и цифры, без пробелов и кириллицы в пути.
fn sanitize(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        }
    }
    out
}
