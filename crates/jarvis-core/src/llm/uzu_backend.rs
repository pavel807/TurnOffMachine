//! Uzu-бэкенд LLM (движок Mirai, github.com/trymirai/uzu, MIT):
//! только Apple Silicon, Metal из коробки, модели из каталога Mirai
//! (свой формат, не GGUF). Подключается как cargo-зависимость — принцип
//! «чистый Rust» не нарушается.
//!
//! Отличия от candle-бэкенда:
//! - модель скачивается сама (`~/.cache/mirai/`), с прогрессом и докачкой;
//! - сессия долгоживущая: префиксный кэш внутри движка — каждый вызов
//!   префиллит только дельту относительно прошлых сообщений;
//! - prefill не поддерживается: форсирование tool-вызова делается
//!   JSON-грамматикой, а нативные tool-call'ы модели мостятся назад
//!   в наш формат `<tool>{...}</tool>`.

use super::config::{self, GenerationParams};
use super::streaming::{push_delta, speakable};
use super::template::ChatTurn;
use anyhow::{Context, Result, anyhow, bail};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use uzu::engine::{Engine, EngineConfig};
use uzu::session::chat::{ChatSession, ChatSessionStreamChunk};
use uzu::storage::DownloadPhase;
use uzu::types::basic::{ReasoningEffort, SamplingMethod, SamplingPolicy};
use uzu::types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig};

/// Таймаут на инициализацию движка (подъём каталога Mirai по сети).
const ENGINE_INIT_TIMEOUT: Duration = Duration::from_secs(120);

/// Имя модели из `settings.toml` → идентификатор в каталоге Mirai.
///
/// Формат: `vendor:model:size:quant-family:quant:bits`; `mirai-m` — 4-бит,
/// `mirai-l` — 8-бит. Старое имя `qwen3.5-0.8b` (каталог candle) отображается
/// на 4-битную uzu-версию той же модели.
pub fn registry_id(model_name: &str) -> Option<&'static str> {
    match model_name {
        "qwen3.5-0.8b" | "qwen3.5-0.8b-mirai-m" => Some("alibaba:qwen3.5:0.8b:mirai:mirai-m:4"),
        "qwen3.5-0.8b-mirai-l" => Some("alibaba:qwen3.5:0.8b:mirai:mirai-l:8"),
        "qwen3.5-2b-mirai-m" => Some("alibaba:qwen3.5:2b:mirai:mirai-m:4"),
        "qwen3.5-2b-mirai-l" => Some("alibaba:qwen3.5:2b:mirai:mirai-l:8"),
        "qwen3.5-4b-mirai-m" => Some("alibaba:qwen3.5:4b:mirai:mirai-m:4"),
        "qwen3.5-4b-mirai-l" => Some("alibaba:qwen3.5:4b:mirai:mirai-l:8"),
        "qwen3.5-9b-mirai-m" => Some("alibaba:qwen3.5:9b:mirai:mirai-m:4"),
        "qwen3.5-9b-mirai-l" => Some("alibaba:qwen3.5:9b:mirai:mirai-l:8"),
        _ => None,
    }
}

pub struct UzuEngine {
    rt: tokio::runtime::Runtime,
    engine: Arc<Engine>,
    /// Долгоживущая сессия: держит веса и KV-кэш. Каждый `reply` передаёт
    /// полную историю — движок сам находит общий префикс и префиллит дельту.
    session: ChatSession,
    #[allow(dead_code)]
    model_id: String,
}

/// Каталог скачанной модели в кэше Mirai: ~/.cache/mirai/models/mirai/
/// <имя-с-дефисами>/<hash>/config.json. Имя каталога — идентификатор
/// модели с «:» → «-». Резолв по этому пути не ходит в сеть.
fn mirai_cache_model_dir(uz_id: &str) -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dash_id = uz_id.replace(':', "-");
    let root = std::path::PathBuf::from(home)
        .join(".cache")
        .join("mirai")
        .join("models")
        .join("mirai")
        .join(dash_id);
    if !root.is_dir() {
        return None;
    }
    // Весы лежат в hash-подкаталоге (config.json + safetensors). Если
    // несколько версий — берём свежую по времени изменения.
    let mut best: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
    for entry in std::fs::read_dir(&root).ok()?.flatten() {
        let candidate = entry.path().join("config.json");
        if candidate.is_file()
            && let Ok(modified) = entry.metadata().and_then(|m| m.modified())
            && best.as_ref().is_none_or(|(t, _)| modified > *t)
        {
            best = Some((modified, entry.path()));
        }
    }
    let dir = best.map(|(_, p)| p).unwrap_or(root);
    // Заглушка без весов (обрыв скачивания: config.json + README, но
    // safetensors нет) не считается моделью: она затеняла каталог Mirai,
    // и движок не докачивал веса — LLM падал «пуст или неполон»
    // (репорт 2026-11-21: в кэше 128 КБ вместо ~2 ГБ).
    if !mirai_weights_present(&dir) {
        return None;
    }
    Some(dir)
}

/// Есть ли в каталоге модели сами веса: любой файл ≥ 50 МБ (safetensors/
/// gguf/бин.), на любой глубине ≤ 3. Конфиг и README в расчёт не идут —
/// их наличие не означает скачанную модель.
fn mirai_weights_present(dir: &std::path::Path) -> bool {
    const MIN_WEIGHT_BYTES: u64 = 50_000_000;
    let mut stack = vec![(dir.to_path_buf(), 0u8)];
    while let Some((cur, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&cur) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.is_file() && meta.len() >= MIN_WEIGHT_BYTES {
                return true;
            }
            if meta.is_dir() && depth < 3 {
                stack.push((path, depth + 1));
            }
        }
    }
    false
}

/// Каталог кэша Mirai для модели по имени из настроек
/// (name → идентификатор каталога → каталог с дефисами). None — модель не
/// из каталога Mirai или весов нет (заглушка обрывом скачивания).
pub fn mirai_model_dir_by_name(name: &str) -> Option<std::path::PathBuf> {
    mirai_cache_model_dir(registry_id(name)?)
}

/// Удалить модель Mirai целиком: все версии в кэше `~/.cache/mirai` и
/// локальную копию `models/uzu/<имя>`. Возвращает true, если что-то было
/// удалено. Локальная копия uzu может лежать под идентификатором каталога
/// (с дефисами) — удаляем и её, если найдена.
pub fn mirai_model_remove(name: &str) -> anyhow::Result<bool> {
    let Some(uz_id) = registry_id(name) else {
        return Ok(false);
    };
    let dash = uz_id.replace(':', "-");
    let mut removed = false;
    if let Some(home) = std::env::var_os("HOME") {
        let root = std::path::PathBuf::from(home)
            .join(".cache")
            .join("mirai")
            .join("models")
            .join("mirai")
            .join(&dash);
        if root.is_dir() {
            std::fs::remove_dir_all(&root)
                .with_context(|| format!("Не удалось удалить {}", root.display()))?;
            removed = true;
        }
    }
    for alias in [name, dash.as_str()] {
        let local = crate::paths::models_dir().join("uzu").join(alias);
        if local.is_dir() {
            std::fs::remove_dir_all(&local)
                .with_context(|| format!("Не удалось удалить {}", local.display()))?;
            removed = true;
        }
    }
    Ok(removed)
}

impl UzuEngine {
    /// Инициализация движка, скачивание модели (если нужно) и открытие сессии.
    pub fn load(models_dir: &std::path::Path, events: &crate::events::EventTx) -> Result<Self> {
        let model_name = config::model_name();
        let registry_id_opt = registry_id(&model_name);

        // Локальный реестр: models/uzu/<имя> — модели, встроенные в .app
        // (build_macos.sh кладёт models/ в payload, provision_data
        // распаковывает). Имя каталога = идентификатор модели в LocalRegistry.
        let local_root = models_dir.join("uzu");

        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .context("Не удалось поднять tokio runtime для uzu")?;

        let started = Instant::now();
        println!("🧩 Uzu: инициализация движка…");
        // allow_ollama/lmstudio выключены: не сканируем localhost на предмет
        // чужих серверов. application_identifier не задаём — без него uzu не
        // лезет в keychain/settings-хранилище. local_path подключает
        // models/uzu как реестр локальных моделей (если каталог есть).
        // timeout создаётся ВНУТРИ async-блока: снаружи block_on ещё нет
        // контекста рантайма, и tokio паникует на создании таймера.
        let engine = rt
            .block_on(async {
                tokio::time::timeout(
                    ENGINE_INIT_TIMEOUT,
                    Engine::new(EngineConfig {
                        local_path: local_root
                            .is_dir()
                            .then(|| local_root.to_string_lossy().into_owned()),
                        allow_ollama_usage: false,
                        allow_lmstudio_usage: false,
                        ..Default::default()
                    }),
                )
                .await
            })
            .map_err(|_| anyhow!("Uzu: движок не инициализировался за {ENGINE_INIT_TIMEOUT:?}"))?
            .context("Uzu: не удалось инициализировать движок (каталог Mirai недоступен?)")?;
        let engine = Arc::new(engine);
        println!("🧩 Uzu: движок готов за {:.1?}", started.elapsed());

        // Разрешение модели: сначала локальная копия (models/uzu/<имя>),
        // затем кэш каталога Mirai на диске (~/.cache/mirai — резолв по
        // пути, без сети), и только потом сам каталог с скачиванием.
        // Сетевой запрос каталога стоил 15 секунд на каждый запуск.
        let started = Instant::now();
        let mut uz_id_for_download: Option<String> = None;
        let mut model = None;

        let mut source = String::new();
        if let Some(uz_id) = registry_id_opt {
            if let Some(dir) = mirai_cache_model_dir(uz_id) {
                match rt
                    .block_on(async {
                        tokio::time::timeout(
                            Duration::from_secs(15),
                            engine.model(dir.to_string_lossy().into_owned()),
                        )
                        .await
                    })
                    .ok()
                    .and_then(|r| r.ok())
                    .flatten()
                {
                    Some(m) => {
                        model = Some(m);
                        source = format!("кэш Mirai ({})", dir.display());
                    }
                    None => {}
                }
            }
        }

        let model = match model {
            Some(m) => Some(m),
            None => {
                let resolved = rt
                    .block_on(async {
                        tokio::time::timeout(
                            Duration::from_secs(60),
                            engine.model(model_name.clone()),
                        )
                        .await
                    })
                    .map_err(|_| anyhow!("Uzu: каталог не ответил за 60 c"))?
                    .context("Uzu: ошибка запроса каталога")?;
                if resolved.is_some() {
                    source = format!("локальная копия {}", local_root.join(&model_name).display());
                }
                resolved
            }
        };

        let mut model = match model {
            Some(m) => m,
            None => {
                if source.is_empty() {
                    source = format!("локальная копия {}", local_root.join(&model_name).display());
                }
                let uz_id = registry_id_opt
                    .with_context(|| {
                        format!(
                            "Модель «{model_name}» не найдена ни в {}, ни в кэше Mirai, ни в каталоге — \
                             выбери qwen3.5-*-mirai-* или поставь llm_backend = \"candle\"",
                            local_root.display()
                        )
                    })?
                    .to_string();
                let from_catalog = rt
                    .block_on(async {
                        tokio::time::timeout(Duration::from_secs(60), engine.model(uz_id.clone()))
                            .await
                    })
                    .map_err(|_| anyhow!("Uzu: каталог не ответил за 60 c"))?
                    .context("Uzu: ошибка запроса каталога")?;
                let found = from_catalog
                    .with_context(|| format!("Uzu: модель {uz_id} не найдена в каталоге Mirai"))?;
                uz_id_for_download = Some(uz_id.clone());
                source = format!("каталог Mirai ({uz_id})");
                found
            }
        };
        let uz_id = uz_id_for_download.unwrap_or_else(|| model_name.clone());
        println!(
            "🧩 Uzu: источник модели: {source} (за {:.1?})",
            started.elapsed()
        );

        // is_on_device() у uzu значит «локальная, а не облачная», а не «файлы
        // на диске»: каталогные модели помечены OnDevice{Registry} ещё ДО
        // скачивания. Наличие файлов проверяем через model_path().
        let mut model_path = rt.block_on(engine.model_path(&model));
        if model_path.is_none() {
            let total_gb = model_size_gb(&model);
            println!("⬇️  Uzu: скачивание модели {uz_id} (~{total_gb:.1} ГБ)…");
            let _ = events.send(crate::events::JarvisEvent::Log {
                message: format!("Скачивание модели Uzu ({total_gb:.1} ГБ)"),
            });
            let dl_started = Instant::now();
            let mut last_pct = 0u32;
            let stream = rt
                .block_on(engine.download(&model))
                .context("Uzu: не удалось начать скачивание модели")?;
            while let Some(state) = rt.block_on(stream.next()) {
                match &state.phase {
                    DownloadPhase::Error { message } => {
                        bail!("Uzu: ошибка скачивания модели: {message}")
                    }
                    DownloadPhase::Downloaded {} => break,
                    phase if phase.is_in_progress() => {
                        let pct = (state.progress() * 100.0) as u32;
                        if pct >= last_pct + 5 {
                            last_pct = pct;
                            println!(
                                "⬇️  Uzu: {:.0}% ({:.2} / {total_gb:.2} ГБ)",
                                pct,
                                state.downloaded_bytes as f64 / 1e9
                            );
                            let _ = events.send(crate::events::JarvisEvent::Log {
                                message: format!("Скачивание модели Uzu: {pct}%"),
                            });
                        }
                    }
                    _ => {}
                }
            }
            println!("✅ Uzu: модель скачана за {:.1?}", dl_started.elapsed());
            // accessibility мог закэшироваться как Remote — перечитываем модель.
            model = rt
                .block_on(engine.model(uz_id.clone()))
                .context("Uzu: ошибка перечитывания каталога")?
                .with_context(|| format!("Uzu: модель {uz_id} пропала из каталога"))?;
            model_path = rt.block_on(engine.model_path(&model));
            if model_path.is_none() {
                bail!("Uzu: модель скачана, но путь к файлам не появился");
            }
        }

        if let Some(p) = &model_path {
            println!("   Uzu: файлы модели: {p}");
        }

        let started = Instant::now();
        println!("🧩 Uzu: загрузка весов {uz_id}…");
        let session = rt
            .block_on(engine.chat(model, ChatConfig::default()))
            .context("Uzu: не удалось загрузить модель в память")?;
        println!("✅ Uzu: модель готова за {:.1?}", started.elapsed());

        Ok(Self {
            rt,
            engine,
            session,
            model_id: uz_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_params(
        &mut self,
        system: &str,
        history: &[ChatTurn],
        user: &str,
        params_override: Option<GenerationParams>,
        cancel: &AtomicBool,
        // prefill в uzu недоступен (см. комментарий ниже); принимаем для
        // паритета с интерфейсом candle-бэкенда.
        _prefill: Option<&str>,
        mut on_sentence: impl FnMut(&str),
        mut on_progress: impl FnMut(&str),
    ) -> Result<String> {
        let params = params_override.clone().unwrap_or_else(|| {
            config::generation_params()
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        });

        let mut messages = vec![
            ChatMessage::system()
                .with_text(system.to_string())
                // Qwen3.5 по умолчанию думает (<think>), сжигая лимит токенов и
                // время: голосовому ассистенту рассуждения не слышны. Контрольный
                // блок на system-сообщении рендерится в enable_thinking=false.
                .with_reasoning_effort(ReasoningEffort::Disabled),
        ];
        for (question, answer) in history {
            messages.push(ChatMessage::user().with_text(question.clone()));
            messages.push(ChatMessage::assistant().with_text(answer.clone()));
        }
        messages.push(ChatMessage::user().with_text(user.to_string()));

        let session = self.session.clone();
        // Сессия uzu накапливает диалог сама (reply дописывает в состояние и
        // вопрос, и ответ), поэтому передаём только НОВЫЕ сообщения — так
        // работает дельта-префилл. Если наша история разошлась с тем, что
        // сессия уже отрендерила (окно истории уехало, мост tool-call'ов,
        // повтор после ошибки) — сбрасываем сессию и рендерим заново.
        let messages: Vec<ChatMessage> = self.rt.block_on(async {
            let stored = session.messages().await;
            let common = stored
                .iter()
                .zip(messages.iter())
                .take_while(|(a, b)| a.role == b.role && a.text() == b.text())
                .count();
            if common == stored.len() && common < messages.len() {
                Ok::<Vec<ChatMessage>, anyhow::Error>(messages[common..].to_vec())
            } else {
                session
                    .reset()
                    .await
                    .map_err(|e| anyhow!("Uzu: не удалось сбросить сессию: {e:?}"))?;
                Ok(messages.clone())
            }
        })?;

        // prefill в uzu не существует. Форсирование tool-вызова пока только
        // через инструкцию в промпте и мост нативных tool-call'ов (см. ниже);
        // когда в uzu вернут рабочую capability-grammar — добавить
        // Grammar::JsonSchema со схемой tool-JSON.
        //
        // Сэмплинг: без override (разговорная ветка) — рекомендованные
        // каталогом дефолты модели: навязывание candle-значений давало бред.
        // С override (веб-ответы по выдаче и tool-действия) — сознательно
        // детерминированный сэмплинг: низкая температура и min_p давят
        // выдумывание фактов и битый tool-JSON, при этом не задевают
        // разговорную ветку.
        let sampling_policy = params_override
            .map(|p| SamplingPolicy::Custom {
                method: SamplingMethod::Stochastic {
                    temperature: Some(p.temperature),
                    top_k: None,
                    top_p: Some(p.top_p),
                    min_p: Some(0.05),
                    repetition_penalty: None,
                    suffix_repetition_length: None,
                },
            })
            .unwrap_or(SamplingPolicy::Default {});
        let reply_config = ChatReplyConfig {
            token_limit: token_limit_from(params.max_tokens),
            sampling_policy,
            grammar: None,
            tool_turn_limit: Some(0),
        };

        let t_start = Instant::now();
        let mut first_token: Option<Duration> = None;

        let result: Result<String> = self.rt.block_on(async {
            let stream = session.reply_with_stream(messages, reply_config).await;
            let cancel_token = stream.cancel_token();
            let mut text = String::new();
            let mut buf = String::new();
            let mut forced_tool_text: Option<String> = None;

            loop {
                if cancel.load(Ordering::SeqCst) {
                    cancel_token.cancel();
                }
                let Some(chunk) = stream.next().await else {
                    break;
                };
                match chunk {
                    ChatSessionStreamChunk::Replies { replies } => {
                        if first_token.is_none() {
                            first_token = Some(t_start.elapsed());
                        }
                        let Some(reply) = replies.last() else {
                            continue;
                        };

                        // Нативные tool-call'ы Qwen мостим в наш формат,
                        // чтобы pipeline (tools::parse_lenient) работал без
                        // изменений.
                        let calls = reply.message.tool_calls();
                        if !calls.is_empty() {
                            let mut t = String::new();
                            for call in calls {
                                t.push_str(&format!(
                                    "<tool>{{\"name\":\"{}\",\"args\":{}}}</tool>",
                                    call.name, call.arguments.json
                                ));
                            }
                            forced_tool_text = Some(t);
                            break;
                        }

                        let full = reply.message.text().unwrap_or_default();
                        if full.len() > text.len() && full.starts_with(&text) {
                            let delta = full[text.len()..].to_string();
                            text = full;
                            push_delta(&mut buf, &delta, &mut on_sentence);
                            on_progress(&text);
                        } else if full != text {
                            text = full;
                            buf.clear();
                            on_progress(&text);
                        }
                    }
                    ChatSessionStreamChunk::Error { error } => {
                        bail!("Uzu: ошибка генерации: {error:?}");
                    }
                    ChatSessionStreamChunk::ToolResults { .. } => {}
                }
            }

            if forced_tool_text.is_none()
                && !cancel.load(Ordering::SeqCst)
                && !buf.trim().is_empty()
                && !buf.contains("<tool>")
                && !buf.contains("</tool>")
                && speakable(&buf)
            {
                on_sentence(&buf);
            }
            Ok(forced_tool_text.unwrap_or(text))
        });

        let elapsed = t_start.elapsed();
        if let Ok(text) = &result {
            println!(
                "   Uzu: первый токен за {:.0?}, ответ {} симв. за {:.0?}",
                first_token.unwrap_or_default(),
                text.chars().count(),
                elapsed
            );
        }
        result
    }

    /// Синхронный сброс сессии (на будущее; сейчас не нужен — reprefill
    /// внутри uzu сам разбирается с расходящейся историей).
    #[allow(dead_code)]
    pub fn reset_session(&self) -> Result<()> {
        let session = self.session.clone();
        self.rt
            .block_on(async { session.reset().await })
            .map_err(|e| anyhow!("Uzu: не удалось сбросить сессию: {e:?}"))
    }
}

/// Лимит генерации для uzu: `0` — без лимита (None у uzu означает
/// «до конца»); ненулевое — страховочный кап в токенах.
fn token_limit_from(max_tokens: usize) -> Option<u32> {
    u32::try_from(max_tokens).ok().filter(|limit| *limit > 0)
}

/// Приблизительный размер модели в ГБ из метаданных каталога.
fn model_size_gb(model: &uzu::types::model::Model) -> f64 {
    if let uzu::types::model::ModelAccessibility::OnDevice {
        source: uzu::types::model::ModelSource::Registry { files, .. },
    } = &model.accessibility
    {
        let bytes: i64 = files.iter().map(|f| f.size.max(0)).sum();
        if bytes > 0 {
            return bytes as f64 / 1e9;
        }
    }
    1.1 // типичный размер 2B 4-бит; для лога точность не критична
}

/// Доступ к движку для тестов/диагностики (например, список моделей каталога).
impl UzuEngine {
    #[allow(dead_code)]
    pub fn engine(&self) -> &Arc<Engine> {
        &self.engine
    }
}

#[cfg(test)]
mod tests {
    use super::token_limit_from;

    #[test]
    fn zero_max_tokens_means_unlimited_for_uzu() {
        // None у uzu — «без лимита»: длинные ответы не обрываются.
        assert_eq!(token_limit_from(0), None);
        // Ненулевое — страховочный кап, проходит как есть.
        assert_eq!(token_limit_from(256), Some(256));
        assert_eq!(token_limit_from(1), Some(1));
    }
}
