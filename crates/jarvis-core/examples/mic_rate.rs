//! Замер ФАКТИЧЕСКОЙ частоты микрофона: устройство может заявлять 48 кГц,
//! а отдавать 24 кГц (известная особенность встроенных микрофонов на
//! Apple Silicon). Тогда ресемплинг растягивает речь вдвое, CTC декодирует
//! мусор — и wake word не срабатывает никогда.
//!
//! Запуск:
//!   cargo run -p jarvis-core --example mic_rate -- [секунды]
//!
//! Печатает: заявленную частоту, измеренную частоту кадров, отношение и
//! вердикт (совпадает / устройство врёт).

use anyhow::Context as _;
use cpal::traits::DeviceTrait;
use cpal::traits::HostTrait;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    jarvis_core::init_tuning();
    let secs: f64 = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.0);

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("нет устройства ввода по умолчанию"))?;
    let name = device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "?".into());
    let sup = device.default_input_config()?;
    let ch = sup.channels() as usize;
    let declared = sup.sample_rate() as usize;

    let counter = Arc::new(AtomicU64::new(0));
    let cb_counter = Arc::clone(&counter);
    let ch_cb = ch;
    let mut cfg: cpal::StreamConfig = sup.into();
    cfg.buffer_size = cpal::BufferSize::Default;
    let built = match sup.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            cfg,
            move |d: &[f32], _| {
                cb_counter.fetch_add((d.len() / ch_cb.max(1)) as u64, Ordering::Relaxed);
            },
            |e| eprintln!("ошибка потока: {e}"),
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            cfg,
            move |d: &[i16], _| {
                cb_counter.fetch_add((d.len() / ch_cb.max(1)) as u64, Ordering::Relaxed);
            },
            |e| eprintln!("ошибка потока: {e}"),
            None,
        ),
        f => anyhow::bail!("неподдерживаемый формат: {f:?}"),
    };
    let stream = built?;
    use cpal::traits::StreamTrait as _;
    stream.play()?;

    // Прогрев: первые кадры — щелчок запуска, их не считаем.
    std::thread::sleep(Duration::from_millis(300));
    let start_frames = counter.load(Ordering::Relaxed);
    let t0 = Instant::now();
    std::thread::sleep(Duration::from_secs_f64(secs));
    let frames = counter.load(Ordering::Relaxed) - start_frames;
    let elapsed = t0.elapsed().as_secs_f64();

    let measured = (frames as f64 / elapsed).round() as usize;
    let ratio = measured as f64 / declared as f64;
    println!("🎙️ Устройство: «{name}»");
    println!("   заявлено: {declared} Гц ({ch} каналов)");
    println!("   измерено: {measured} Гц ({frames} кадров за {elapsed:.2} с)");
    println!("   отношение: {ratio:.4}");
    if (ratio - 1.0).abs() > 0.05 {
        println!(
            "⚠️ Устройство отдаёт данные НЕ на заявленной частоте — ресемплинг должен \
             идти по {measured} Гц, иначе речь растягивается и CTC декодирует мусор."
        );
    } else {
        println!("✅ Частота совпадает с заявленной.");
    }
    Ok(())
}
