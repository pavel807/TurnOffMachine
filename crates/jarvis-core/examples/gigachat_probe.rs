//! Временная живая проверка GigaChat-пути (не входит в постоянный код).
//! Ключ не печатается — только ответ модели и служебные события.

use jarvis_core::events::JarvisEvent;
use jarvis_core::llm::api_backend::ApiEngine;
use jarvis_core::settings::Settings;
use std::sync::atomic::AtomicBool;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "settings.toml".to_string());
    let settings = Settings::load(std::path::Path::new(&path))?;
    settings.apply();
    let (tx, rx) = std::sync::mpsc::channel::<JarvisEvent>();
    let engine = ApiEngine::load(&tx)?;
    println!(
        "[probe] пресет={} хост={} модель={}",
        settings.api_preset, settings.api_base_url, settings.api_model
    );
    let cancel = AtomicBool::new(false);
    let answer = engine.generate_streaming_with_params(
        "Отвечай по-русски, коротко.",
        &[],
        "Скажи одним предложением: работает ли облако?",
        None,
        &cancel,
        None,
        |s| println!("[озвучка] {s}"),
        |p| println!("[прогресс] {p}"),
    )?;
    println!("[ответ] {answer}");
    for ev in rx {
        if let JarvisEvent::Log { message } = ev {
            println!("[событие] {message}");
        }
    }
    Ok(())
}
