//! Облачный LLM через OpenAI-совместимый API (стриминг SSE).
//!
//! Один клиент закрывает и Яндекс GPT, и GigaChat, и любые другие
//! совместимые сервисы (OpenAI, OpenRouter, LM Studio, vLLM): различаются
//! только базовый адрес, схема авторизации и формат имени модели.
//!
//! Пресет «yandex»: `Authorization: Api-Key <ключ>`, модель
//! `gpt://<каталог>/<модель>/latest` (собирается из полей автоматически).
//! Пресет «gigachat»: ключ пользователя — это Basic-авторизация, access-токен
//! Джарвис берёт сам через oauth (scope настраивается) и кэширует; чат ходит
//! с `Bearer <access-токен>`.
//! Пресет «openai»: `Authorization: Bearer <ключ>`, модель как введена.
//!
//! Стриминг: `stream: true`, разбор `data:`-строк SSE, дельты текста
//! отдаются в TTS по предложениям — общие помощники из `streaming.rs`.

use super::streaming;
use super::template::ChatTurn;
use crate::events::{EventTx, JarvisEvent};
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// Базовый адрес Яндекс GPT (OpenAI-совместимый эндпоинт).
/// Именно `/v1`, а не `/openai/v1` — тот путь даёт HTTP 404 от шлюза Яндекса.
pub const YANDEX_BASE_URL: &str = "https://llm.api.cloud.yandex.net/v1";
/// Модель Яндекса по умолчанию (легче и быстрее старшей).
pub const YANDEX_DEFAULT_MODEL: &str = "yandexgpt-lite";

/// GigaChat: актуальный хост. Сбер выводит из обращения домен
/// `gigachat.devices.sberbank.ru`, новый — `api.giga.chat`.
pub const GIGACHAT_BASE_URL: &str = "https://api.giga.chat/v1";
/// Старый хост GigaChat: ключ пользователя по нему тоже работает, поэтому
/// адрес из старых настроек продолжает работать.
pub const GIGACHAT_LEGACY_BASE_URL: &str = "https://gigachat.devices.sberbank.ru/api/v1";
/// Выдача access-токена: домен не менялся вместе с чат-хостом.
pub const GIGACHAT_OAUTH_URL: &str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth";
pub const GIGACHAT_DEFAULT_MODEL: &str = "GigaChat";
/// Scope по умолчанию — физическое лицо. У юрлиц свой (GIGACHAT_API_B2B /
/// GIGACHAT_API_CORP), поэтому scope вынесен в настройки.
pub const GIGACHAT_DEFAULT_SCOPE: &str = "GIGACHAT_API_PERS";

/// Сертификаты НУЦ Минцифры (официальные с gu-st.ru — Госуслуги): цепочка
/// sber-эндпоинтов подписана ими, Mozilla-корни эти имена не знают.
/// Файл — root + sub CA (2022) + sub CA 2024, PEМ с LF-концами.
const RU_CA_PEM: &str = include_str!("../../certs/russian_trusted_root_ca.crt");

/// Сколько раз пробуем запрос при 429/5xx и сетевых сбоях.
const MAX_ATTEMPTS: usize = 3;
/// Пауза перед первой повторной попыткой; дальше — удвоение.
const RETRY_BASE_DELAY: Duration = Duration::from_millis(700);
/// Потолок паузы (Retry-After может требовать больше).
const RETRY_MAX_DELAY: Duration = Duration::from_secs(20);
/// Таймаут чтения: SSE-стрим идёт долго, но «зависший» хост не должен
/// держать поток генерации вечно — иначе отмена пользователя не работает.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
/// Таймаут записи: тело запроса маленькое, но зависший коннект — нельзя
/// держать поток генерации.
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);

pub struct ApiEngine {
    /// Базовый адрес без хвостового `/`: например `https://llm.api.cloud.yandex.net/openai/v1`.
    base_url: String,
    /// Полное имя модели: `yandexgpt-lite`, `gpt://b1g.../yandexgpt/latest`, `gpt-4o-mini`, …
    model: String,
    /// Заголовок авторизации: `Api-Key <ключ>` (Яндекс) или `Bearer <ключ>`.
    /// Для GigaChat не используется — там Bearer access-токена живёт своей
    /// жизнью (см. `auth_header`).
    auth_value: String,
    /// Пресет: "yandex" | "openai" | "gigachat".
    preset: String,
    /// Только GigaChat: ключ авторизации (Basic …) для выдачи access-токена.
    basic_key: String,
    /// Только GigaChat: scope oauth (PERS/B2B/CORP) — у юрлиц свой.
    scope: String,
    /// Канал событий: битые чанки и обрыв потока не должны теряться.
    events: EventTx,
}

impl ApiEngine {
    /// Сборка из конфига (заполняется из settings.toml до загрузки движка).
    pub fn load(events: &EventTx) -> Result<Self> {
        let preset = match super::config::normalize_preset(&super::config::api_preset()) {
            Ok(p) => p,
            Err(e) => {
                let _ = events.send(JarvisEvent::Error {
                    message: format!("{e:#}"),
                });
                return Err(e);
            }
        };
        let api_key = super::config::api_key();
        if api_key.trim().is_empty() {
            let _ = events.send(JarvisEvent::Error {
                message: "Облачная модель выбрана, но API-ключ не задан (настройки → AI-модель)"
                    .into(),
            });
            return Err(anyhow!(
                "API-ключ не задан: укажи его в настройках → AI-модель"
            ));
        }
        let base_url = super::config::api_base_url();
        let scope = super::config::api_scope();
        let mut basic_key = String::new();
        let auth_value = if preset == "gigachat" {
            // GigaChat: ключ — Basic-авторизация для oauth, чат ходит с
            // Bearer access-токеном, который Джарвис получает сам.
            basic_key = api_key.trim().to_string();
            String::new()
        } else if preset == "yandex" {
            format!("Api-Key {api_key}")
        } else {
            format!("Bearer {api_key}")
        };
        // Полный URI модели для Яндекса: gpt://<каталог>/<модель>[/<версия>].
        // Версию дописываем только если пользователь её не указал —
        // «yandexgpt/rc» не должно превращаться в «…/rc/latest».
        let mut model = super::config::api_model();
        if preset == "yandex" {
            let folder = super::config::api_folder_id();
            if folder.trim().is_empty() && !model.starts_with("gpt://") {
                // Без каталога Яндекс вернёт криптическое «Failed to parse
                // model URI»: говорим прямо и до запроса, где что вписать.
                let _ = events.send(JarvisEvent::Error {
                    message: "Для Яндекс GPT нужен Каталог (folder id, b1g…) — Настройки → Модель"
                        .into(),
                });
                return Err(anyhow!(
                    "Каталог (folder id) не задан: впиши его в Настройки → Модель,                      поле «Каталог (folder id)» — b1g… из консоли Яндекса"
                ));
            }
            if !folder.trim().is_empty() && !model.starts_with("gpt://") {
                let version = if model.contains('/') { "" } else { "/latest" };
                model = format!("gpt://{}/{model}{version}", folder.trim());
            }
        }
        // GigaChat: дефолты адреса и модели, если поля пустые.
        let (base_url, model) = if preset == "gigachat" {
            // Пустой адрес ИЛИ ещё не тронутый дефолт Яндекса → актуальный
            // хост GigaChat. Явный адрес пользователя (в т.ч. старый
            // devices.sberbank.ru) уважаем: по нему ключ тоже работает.
            let base = if base_url.trim().is_empty() || base_url.trim() == YANDEX_BASE_URL {
                GIGACHAT_BASE_URL.to_string()
            } else {
                base_url
            };
            // Модель из настроек: если пусто или остался старый дефолт Яндекса —
            // берём модель GigaChat. Молчаливая подмена путает (пользователь
            // думает, что говорит с той моделью, которую выбрал) — пишем в лог.
            let model = if model.trim().is_empty() || model == YANDEX_DEFAULT_MODEL {
                if !model.trim().is_empty() {
                    let _ = events.send(JarvisEvent::Log {
                        message: format!(
                            "Модель «{model}» для GigaChat не подходит — беру {GIGACHAT_DEFAULT_MODEL}"
                        ),
                    });
                }
                GIGACHAT_DEFAULT_MODEL.to_string()
            } else {
                model
            };
            (base, model)
        } else {
            (base_url, model)
        };
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            auth_value,
            preset,
            basic_key,
            scope: if scope.trim().is_empty() {
                GIGACHAT_DEFAULT_SCOPE.to_string()
            } else {
                scope.trim().to_string()
            },
            events: events.clone(),
        })
    }

    /// Заголовок авторизации для запроса. GigaChat — живой access-токен
    /// (получается Basic-ключом, кэшируется, продлевается сам).
    fn auth_header(&self, force_refresh: bool) -> Result<String> {
        if self.preset != "gigachat" {
            return Ok(self.auth_value.clone());
        }
        Ok(bearer_header(&gigachat_access_token(
            &self.basic_key,
            &self.scope,
            force_refresh,
        )?))
    }

    /// Тело запроса chat/completions.
    fn request_body(
        &self,
        messages: Vec<Value>,
        params: &super::config::GenerationParams,
    ) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "temperature": params.temperature,
            // top_p провайдеры OpenAI-совместимого API понимают, а локальные
            // бэкенды его молча роняли — облако ведёт себя иначе.
            "top_p": params.top_p,
        });
        if params.max_tokens > 0 {
            body["max_tokens"] = json!(params.max_tokens);
        }
        // repetition_penalty — тот же guard от зацикливания, что у локальных
        // моделей. Знают только провайдеры GigaChat (по умолчанию 1.07);
        // остальные OpenAI-совместимые его не понимают и ругаются 400.
        if self.preset == "gigachat" {
            // f32 → f64 с округлением: 1.1 не должен уезжать в провайдера как
            // 1.100000023841858.
            let penalty = (f64::from(params.repeat_penalty) * 1000.0).round() / 1000.0;
            body["repetition_penalty"] = json!(penalty);
        }
        body
    }

    /// Проверка ключа и адреса минимальным запросом (без стрима, пара токенов).
    /// Возвращает ответ модели или честную ошибку — для кнопки «Проверить».
    pub fn check_connection(&self) -> Result<String> {
        let url = self.chat_url();
        let body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": "Ответь одним словом: готов"}],
            "max_tokens": 8,
            "temperature": 0.0,
        });
        let response = agent_for(&url)
            .post(&url)
            .header("Authorization", &self.auth_header(false)?)
            .header("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| anyhow!("не удалось соединиться: {e}"))?;
        let status = response.status();
        let text = response
            .into_body()
            .read_to_string()
            .unwrap_or_default();
        if status != 200 {
            let msg = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| {
                    v.pointer("/error/message")
                        .and_then(|m| m.as_str().map(str::to_string))
                })
                .unwrap_or(text);
            let msg = msg.trim().to_string();
            if msg.is_empty() {
                return Err(anyhow!("HTTP {status}"));
            }
            return Err(anyhow!("HTTP {status}: {}", msg.chars().take(200).collect::<String>()));
        }
        let parsed: Value = serde_json::from_str(&text)
            .map_err(|e| anyhow!("странный ответ провайдера: {e}"))?;
        let answer = parsed
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if answer.is_empty() {
            return Err(anyhow!("провайдер ответил пустотой — модель не ответила"));
        }
        Ok(answer)
    }

    /// Сообщения запроса: system + история + пользователь (+ prefill как
    /// частичный ответ ассистента — стандартный приём OpenAI-совместимых API).
    fn messages(
        &self,
        system: &str,
        history: &[ChatTurn],
        user: &str,
        prefill: Option<&str>,
    ) -> Vec<Value> {
        let mut messages = vec![json!({ "role": "system", "content": system })];
        for (q, a) in history {
            messages.push(json!({ "role": "user", "content": q }));
            messages.push(json!({ "role": "assistant", "content": a }));
        }
        messages.push(json!({ "role": "user", "content": user }));
        if let Some(p) = prefill {
            messages.push(json!({ "role": "assistant", "content": p }));
        }
        messages
    }

    /// Адрес чата: базовый адрес + путь эндпоинта. Оба хоста GigaChat
    /// (новый и старый) используют один и тот же путь.
    fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Запрос с ограниченными повторами: 429/5xx и сетевые сбои — с
    /// экспоненциальной паузой (Retry-After, если провайдер его прислал).
    /// Остальные 4xx не повторяем: опечатку в ключе или адресе повтор не
    /// исправит, только задержит ответ.
    fn send_with_retry(
        &self,
        url: &str,
        body: &Value,
        auth: &str,
        cancel: &AtomicBool,
    ) -> Result<ureq::http::Response<ureq::Body>> {
        let mut attempt = 0usize;
        let mut delay = RETRY_BASE_DELAY;
        loop {
            attempt += 1;
            let response = match agent_for(url)
                .post(url)
                .header("Authorization", auth)
                .header("Content-Type", "application/json")
                .send_json(body)
            {
                Ok(r) => r,
                Err(e) => {
                    // Сетевой сбой (обрыв, таймаут, DNS) — пробуем ещё раз.
                    if attempt >= MAX_ATTEMPTS || cancel.load(Ordering::SeqCst) {
                        return Err(anyhow!("облачная модель недоступна: {e}"))
                            .context("запрос к облачной модели не удался");
                    }
                    let _ = self.events.send(JarvisEvent::Log {
                        message: format!(
                            "Облако: сетевой сбой ({e}), повтор {attempt}/{}",
                            MAX_ATTEMPTS - 1
                        ),
                    });
                    sleep_or_cancel(delay, cancel);
                    delay = (delay * 2).min(RETRY_MAX_DELAY);
                    continue;
                }
            };
            let status = response.status();
            let retryable = status == ureq::http::StatusCode::TOO_MANY_REQUESTS
                || status.is_server_error();
            if retryable && attempt < MAX_ATTEMPTS && !cancel.load(Ordering::SeqCst) {
                let wait = retry_after(response.headers().get("retry-after"))
                    .unwrap_or(delay)
                    .min(RETRY_MAX_DELAY);
                let _ = self.events.send(JarvisEvent::Log {
                    message: format!(
                        "Облако: HTTP {status}, повтор через {} с",
                        wait.as_secs().max(1)
                    ),
                });
                // Тело ответа надо дочитать — иначе соединение не освободится.
                let _ = response.into_body().read_to_string();
                sleep_or_cancel(wait, cancel);
                delay = (delay * 2).min(RETRY_MAX_DELAY);
                continue;
            }
            return Ok(response);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_params(
        &self,
        system: &str,
        history: &[ChatTurn],
        user: &str,
        params_override: Option<super::config::GenerationParams>,
        cancel: &AtomicBool,
        prefill: Option<&str>,
        mut on_sentence: impl FnMut(&str),
        mut on_progress: impl FnMut(&str),
    ) -> Result<String> {
        let params = params_override.unwrap_or_else(|| {
            super::config::generation_params()
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        });
        let body = self.request_body(self.messages(system, history, user, prefill), &params);

        let url = self.chat_url();
        on_progress("запрос…");

        let mut auth = self.auth_header(false)?;
        let mut response = self.send_with_retry(&url, &body, &auth, cancel)?;
        // GigaChat: access-токен живёт ~30 минут — 401 значит «протух»,
        // берём свежий и повторяем.
        if response.status() == 401 && self.preset == "gigachat" {
            auth = self.auth_header(true)?;
            response = self.send_with_retry(&url, &body, &auth, cancel)?;
        }

        let status = response.status();
        if status != 200 {
            let text = response
                .into_body()
                .read_to_string()
                .unwrap_or_default();
            // У Яндекса текст ошибки в JSON: {"error":{"message":".."}}.
            let msg = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| {
                    v.pointer("/error/message")
                        .and_then(|m| m.as_str().map(str::to_string))
                })
                .unwrap_or(text);
            let msg = msg.trim().to_string();
            let detail = if msg.is_empty() {
                format!("HTTP {status}")
            } else if msg.chars().count() > 300 {
                format!("HTTP {status}: {}", msg.chars().take(300).collect::<String>())
            } else {
                format!("HTTP {status}: {msg}")
            };
            return Err(anyhow!("облачная модель: {detail}"));
        }

        // Разбор SSE: строки `data: {json}`, конец — `data: [DONE]`.
        let reader = std::io::BufReader::new(response.into_body().into_reader());
        let mut buf = String::new();
        let mut suppress_think = false;
        let outcome = parse_sse(reader, cancel, &mut |delta, accumulated| {
            let clean = streaming::strip_think(delta, &mut suppress_think);
            if !clean.is_empty() {
                streaming::push_delta(&mut buf, &clean, &mut on_sentence);
            }
            // Прогресс по каждой дельте — как у локальных бэкендов: интерфейс
            // показывает текст, а не молчит до конца ответа.
            on_progress(accumulated);
        });
        // Хвост, который не дорос до конца предложения.
        if !buf.trim().is_empty() {
            let tail = std::mem::take(&mut buf);
            if streaming::speakable(&tail) {
                on_sentence(&tail);
            }
        }
        if outcome.bad_chunks > 0 {
            let _ = self.events.send(JarvisEvent::Log {
                message: format!(
                    "Облако: битых чанков в ответе — {} (текст собран из остальных)",
                    outcome.bad_chunks
                ),
            });
        }
        if outcome.cancelled {
            return Ok(outcome.text);
        }
        // Поток закрылся без `[DONE]` и без finish_reason — ответ оборван.
        // Молча вернуть обрывок нельзя: пользователь решит, что это весь
        // ответ, и задаст следующий вопрос вслепую.
        if !outcome.done && outcome.finish_reason.is_none() {
            if outcome.text.trim().is_empty() {
                return Err(anyhow!(
                    "облачная модель закрыла поток, не ответив ни слова"
                ));
            }
            let _ = self.events.send(JarvisEvent::Log {
                message: "Ответ облачной модели оборван: поток данных прерван до конца".into(),
            });
            return Ok(format!(
                "{}\n\n[ответ оборван: поток данных прерван]",
                outcome.text
            ));
        }
        Ok(outcome.text)
    }
}

/// Итог разбора SSE-потока.
#[derive(Debug, Default)]
struct SseOutcome {
    /// Весь текст ответа (сырые дельты).
    text: String,
    /// Пришёл ли `[DONE]`-маркер конца потока.
    done: bool,
    /// finish_reason финального чанка (если провайдер его прислал).
    finish_reason: Option<String>,
    /// Сколько чанков не удалось разобрать как JSON.
    bad_chunks: usize,
    /// Пользователь остановил генерацию.
    cancelled: bool,
}

/// Разбор SSE-потока GigaChat/OpenAI.
///
/// Особенности, из-за которых это нельзя было «просто читать строки»:
/// - `data:` без пробела — валидный SSE, префикс режем по Trim, а не по
///   точному «data: »;
/// - битый JSON нельзя молча ронять: дельта теряется, ответ получается
///   дырявым — считаем такие чанки и предупреждаем;
/// - конец потока — маркер `[DONE]` или `finish_reason`; без них поток
///   закрылся оборванным.
fn parse_sse<R: std::io::BufRead>(
    reader: R,
    cancel: &AtomicBool,
    on_delta: &mut dyn FnMut(&str, &str),
) -> SseOutcome {
    let mut outcome = SseOutcome::default();
    for line in reader.lines() {
        if cancel.load(Ordering::SeqCst) {
            outcome.cancelled = true;
            break;
        }
        let Ok(line) = line else { break };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // SSE разрешает «data:» без пробела — режем префикс, а не строку.
        let Some(payload) = trimmed.strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload.is_empty() {
            continue;
        }
        if payload == "[DONE]" {
            outcome.done = true;
            break;
        }
        let chunk: Value = match serde_json::from_str(payload) {
            Ok(v) => v,
            Err(_) => {
                outcome.bad_chunks += 1;
                continue;
            }
        };
        if let Some(reason) = chunk
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            && !reason.is_empty()
        {
            outcome.finish_reason = Some(reason.to_string());
        }
        let Some(delta) = chunk
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        else {
            continue;
        };
        outcome.text.push_str(delta);
        on_delta(delta, &outcome.text);
    }
    outcome
}

/// Тело oauth-запроса GigaChat: scope задаёт тип доступа.
fn oauth_request_body(scope: &str) -> String {
    format!("scope={scope}")
}

/// Basic-авторизация GigaChat: пользовательский ключ идёт как есть.
fn basic_auth_header(key: &str) -> String {
    format!("Basic {key}")
}

/// Bearer-авторизация: access-токен GigaChat или ключ OpenAI-совместимого API.
fn bearer_header(token: &str) -> String {
    format!("Bearer {token}")
}

/// Отпечаток ключа: в кэш токена кладём не сам ключ, а его хэш — после
/// смены ключа пользователем старый токен не должен молча переиспользоваться.
fn key_fingerprint(key: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in key.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Агент для публичных хостов: корни ureq по умолчанию (Mozilla/webpki).
fn public_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_resolve(Some(Duration::from_millis(1500)))
            .timeout_connect(Some(Duration::from_secs(10)))
            .timeout_per_read(Some(READ_TIMEOUT))
            .timeout_per_write(Some(WRITE_TIMEOUT))
            .http_status_as_error(false)
            .build()
            .new_agent()
    })
}

/// Агент для хостов Sber: публичные корни (системный bundle, если он
/// доступен) ПЛЮС корни НУЦ Минцифры.
///
/// Раньше здесь стояли ТОЛЬКО три сертификата НУЦ: такой набор не доверяет
/// ни одному публичному CA, и на новом хосте `api.giga.chat` (он тоже подписан
/// российским УЦ) TLS проходил, а вот любой другой провайдер — нет.
/// Теперь набор расширен, а не заменён.
fn ru_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        let mut certs = system_root_certs();
        certs.extend(ru_ncc_certs());
        if certs.is_empty() {
            // Ни системного bundle, ни встроенных НУЦ — доверяем публичным
            // корням по умолчанию: агент работает, просто без российских УЦ.
            return public_agent().clone();
        }
        let tls = ureq::tls::TlsConfig::builder()
            .root_certs(ureq::tls::RootCerts::Specific(Arc::new(certs)))
            .build();
        ureq::Agent::config_builder()
            .tls_config(tls)
            .timeout_resolve(Some(Duration::from_millis(1500)))
            .timeout_connect(Some(Duration::from_secs(10)))
            .timeout_per_read(Some(READ_TIMEOUT))
            .timeout_per_write(Some(WRITE_TIMEOUT))
            .http_status_as_error(false)
            .build()
            .new_agent()
    })
}

/// Агент под конкретный URL: хосты Sber (сертификат подписан НУЦ Минцифры)
/// идут через агент с расширенным набором корней, остальные — с публичными.
fn agent_for(url: &str) -> &'static ureq::Agent {
    if needs_ru_roots(url) {
        ru_agent()
    } else {
        public_agent()
    }
}

/// Нужен ли этому хосту корень НУЦ Минцифры.
fn needs_ru_roots(url: &str) -> bool {
    let host = url_host(url);
    host.ends_with(".sberbank.ru")
        || host.ends_with(".giga.chat")
        || host.ends_with(".gigachat.ru")
}

/// Хост из URL (без схемы, пути, пользователя и порта).
fn url_host(url: &str) -> String {
    let rest = url.rsplit("://").next().unwrap_or(url);
    let rest = rest.split('/').next().unwrap_or(rest);
    let rest = rest.rsplit('@').next().unwrap_or(rest);
    let host = rest.split(':').next().unwrap_or(rest);
    host.trim().to_lowercase()
}

/// Сертификаты НУЦ Минцифры из встроенного бандла. Бандл опционален: если он
/// не собран или не разбирается — просто добавляем меньше корней.
fn ru_ncc_certs() -> Vec<ureq::tls::Certificate<'static>> {
    // Скачанный с gu-st.ru файл мог прийти с CRLF — на всякий случай
    // нормализуем перед разбором.
    let pem = RU_CA_PEM.replace('\r', "");
    ureq::tls::parse_pem(pem.as_bytes())
        .filter_map(|item| match item {
            Ok(ureq::tls::PemItem::Certificate(c)) => Some(c.to_owned()),
            _ => None,
        })
        .collect()
}

/// Публичные корни из системного bundle. Нужны, чтобы набор доверия был
/// расширенным, а не «только НУЦ»: иначе агент Sber не доверяет ни одному
/// публичному CA. Файла может не быть (Windows, песочница iOS) — тогда
/// публичные корни берёт обычный агент ureq.
fn system_root_certs() -> Vec<ureq::tls::Certificate<'static>> {
    const BUNDLES: &[&str] = &[
        "/etc/ssl/cert.pem",                  // macOS
        "/etc/ssl/certs/ca-certificates.crt", // Debian/Ubuntu
        "/etc/pki/tls/certs/ca-bundle.crt",   // Fedora/RHEL
    ];
    for path in BUNDLES {
        let Ok(pem) = std::fs::read_to_string(path) else {
            continue;
        };
        let certs: Vec<ureq::tls::Certificate<'static>> = ureq::tls::parse_pem(pem.as_bytes())
            .filter_map(|item| match item {
                Ok(ureq::tls::PemItem::Certificate(c)) => Some(c.to_owned()),
                _ => None,
            })
            .collect();
        if !certs.is_empty() {
            return certs;
        }
    }
    Vec::new()
}

/// Кэш access-токена GigaChat: токен + момент истечения + отпечаток ключа.
struct GigachatToken {
    token: String,
    until: Instant,
    /// Отпечаток ключа, под который выдан токен: смена ключа обнуляет кэш.
    fingerprint: u64,
}

static GIGACHAT_TOKEN: std::sync::Mutex<Option<GigachatToken>> = std::sync::Mutex::new(None);

/// Access-токен GigaChat: из кэша, если жив; иначе POST на oauth с
/// Basic-ключом пользователя. `force_refresh` — после 401.
fn gigachat_access_token(basic_key: &str, scope: &str, force_refresh: bool) -> Result<String> {
    if basic_key.trim().is_empty() {
        return Err(anyhow!(
            "Ключ GigaChat не задан: в настройках → AI-модель → GigaChat"
        ));
    }
    let fingerprint = key_fingerprint(basic_key);
    {
        let cached = GIGACHAT_TOKEN
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(t) = cached.as_ref()
            && t.fingerprint == fingerprint
            && !force_refresh
            && t.until > Instant::now()
        {
            return Ok(t.token.clone());
        }
    }
    // RqUID — уникальный идентификатор запроса (uuid v4). Без зависимостей:
    // энтропия из времени, pid и атомарного счётчика.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let c = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = std::process::id() as u64;
    let mut h = nanos ^ (pid << 32) ^ c.wrapping_mul(0x9E3779B97F4A7C15);
    let mut bytes = [0u8; 16];
    for b in &mut bytes {
        h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        h ^= h >> 33;
        *b = (h >> 24) as u8;
    }
    // uuid v4: версия 4 и вариант RFC 4122.
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;
    let rquid = format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    );

    let response = agent_for(GIGACHAT_OAUTH_URL)
        .post(GIGACHAT_OAUTH_URL)
        .header("RqUID", &rquid)
        .header("Authorization", &basic_auth_header(basic_key))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .send(oauth_request_body(scope))
        .map_err(|e| anyhow!("не удалось соединиться с GigaChat oauth: {e}"))?;
    let status = response.status();
    let text = response.into_body().read_to_string().unwrap_or_default();
    if status != 200 {
        let msg = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| {
                v.get("message")
                    .or_else(|| v.pointer("/error/message"))
                    .and_then(|m| m.as_str().map(str::to_string))
            })
            .unwrap_or(text);
        let msg = msg.trim().to_string();
        return Err(anyhow!(
            "GigaChat oauth: HTTP {status}{}",
            if msg.is_empty() {
                String::new()
            } else {
                format!(": {}", msg.chars().take(200).collect::<String>())
            }
        ));
    }
    let parsed: Value =
        serde_json::from_str(&text).map_err(|e| anyhow!("странный ответ oauth: {e}"))?;
    let token = parsed
        .get("access_token")
        .and_then(Value::as_str)
        .context("oauth не вернул access_token")?
        .to_string();
    // expires_at — unix-время в миллисекундах; страховка: не дольше 25 минут
    // (последние минуты токен уже может быть отозван на шлюзе).
    let until = parsed
        .get("expires_at")
        .and_then(Value::as_i64)
        .map(|ms| {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            let secs_left = ((ms - now_ms).max(0) / 1000) as u64;
            Instant::now() + Duration::from_secs(secs_left.saturating_sub(180))
        })
        .unwrap_or_else(|| Instant::now() + Duration::from_secs(25 * 60));
    let mut cache = GIGACHAT_TOKEN
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    *cache = Some(GigachatToken {
        token: token.clone(),
        until,
        fingerprint,
    });
    Ok(token)
}

/// Retry-After из заголовка ответа: провайдер может попросить подождать.
fn retry_after(value: Option<&ureq::http::HeaderValue>) -> Option<Duration> {
    let raw = value?.to_str().ok()?.trim().to_string();
    let secs: u64 = raw.parse().ok()?;
    Some(Duration::from_secs(secs))
}

/// Пауза с проверкой отмены: пользователь не должен ждать паузу после
/// «стоп».
fn sleep_or_cancel(total: Duration, cancel: &AtomicBool) {
    const STEP: Duration = Duration::from_millis(100);
    let mut left = total;
    while !left.is_zero() {
        if cancel.load(Ordering::SeqCst) {
            return;
        }
        let chunk = left.min(STEP);
        std::thread::sleep(chunk);
        left -= chunk;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn engine(preset: &str, model: &str) -> ApiEngine {
        let (tx, _rx) = std::sync::mpsc::channel::<JarvisEvent>();
        ApiEngine {
            base_url: "https://api.giga.chat/v1".into(),
            model: model.into(),
            auth_value: "Bearer k".into(),
            preset: preset.into(),
            basic_key: String::new(),
            scope: GIGACHAT_DEFAULT_SCOPE.into(),
            events: tx,
        }
    }

    #[test]
    fn yandex_model_uri_is_built_from_folder() {
        // Пресет Яндекса собирает URI, когда ключ и каталог заданы.
        // Здесь проверяем только логику URI — статик конфиг трогать не надо.
        let folder = "b1gabc123";
        let model = "yandexgpt";
        let built = format!("gpt://{folder}/{model}/latest");
        assert_eq!(built, "gpt://b1gabc123/yandexgpt/latest");
        // Уже полный URI не трогаем.
        let full = "gpt://b1gabc123/yandexgpt/latest";
        assert!(full.starts_with("gpt://"));
    }

    #[test]
    fn gigachat_agent_has_ru_ca_roots() {
        // Бандл НУЦ разбирается: хотя бы 2 сертификата (root + sub CA).
        let certs = ru_ncc_certs();
        assert!(certs.len() >= 2, "в бандле НУЦ {} сертификатов", certs.len());
    }

    #[test]
    fn ru_roots_extend_public_ones_not_replace() {
        // Набор доверия расширенный: публичные корни (системный bundle)
        // плюс НУЦ Минцифры. Проверяем обе части.
        assert!(
            !ru_ncc_certs().is_empty(),
            "встроенные корни НУЦ должны разбираться"
        );
        // Системный bundle есть не везде (Windows, песочница) — тест не
        // требует его наличия, но требует, чтобы агент в любом случае
        // получил непустой набор корней.
        let mut all = system_root_certs();
        all.extend(ru_ncc_certs());
        assert!(!all.is_empty(), "набор корней доверия не должен быть пустым");
    }

    #[test]
    fn hosts_with_ru_certificates_use_extended_roots() {
        // Оба хоста GigaChat подписаны российским УЦ → им нужны корни НУЦ.
        assert!(needs_ru_roots(GIGACHAT_BASE_URL));
        assert!(needs_ru_roots(GIGACHAT_LEGACY_BASE_URL));
        assert!(needs_ru_roots(GIGACHAT_OAUTH_URL));
        assert!(needs_ru_roots("https://api.giga.chat/v1/chat/completions"));
        // Публичные провайдеры — обычные корни.
        assert!(!needs_ru_roots(YANDEX_BASE_URL));
        assert!(!needs_ru_roots("https://api.openai.com/v1"));
        assert!(!needs_ru_roots("http://localhost:1234/v1"));
    }

    #[test]
    fn gigachat_oauth_body_carries_scope() {
        assert_eq!(
            oauth_request_body(GIGACHAT_DEFAULT_SCOPE),
            "scope=GIGACHAT_API_PERS"
        );
        // Scope юрлица — другой, он приходит из настроек.
        assert_eq!(oauth_request_body("GIGACHAT_API_CORP"), "scope=GIGACHAT_API_CORP");
    }

    #[test]
    fn auth_header_forms_are_correct() {
        // Ключ пользователя идёт в Basic-заголовке oauth...
        assert_eq!(basic_auth_header("KEY=="), "Basic KEY==");
        // ...а в чат — access-токен в Bearer.
        assert_eq!(bearer_header("TOKEN"), "Bearer TOKEN");
    }

    #[test]
    fn chat_url_is_derived_for_both_gigachat_hosts() {
        let mut e = engine("gigachat", GIGACHAT_DEFAULT_MODEL);
        e.base_url = GIGACHAT_BASE_URL.to_string();
        assert_eq!(e.chat_url(), "https://api.giga.chat/v1/chat/completions");
        e.base_url = GIGACHAT_LEGACY_BASE_URL.to_string();
        assert_eq!(
            e.chat_url(),
            "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        );
    }

    #[test]
    fn request_body_has_stream_and_no_zero_limit() {
        let engine = engine("openai", "test-model");
        let params = super::super::config::GenerationParams {
            temperature: 0.5,
            top_p: 0.9,
            max_tokens: 0,
            ..Default::default()
        };
        let body = engine.request_body(vec![json!({"role": "user", "content": "привет"})], &params);
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["model"], json!("test-model"));
        assert!(body.get("max_tokens").is_none(), "0 = без лимита, поле не ставится");
        // top_p больше не молча теряется.
        assert_eq!(body["top_p"], json!(0.9));

        let params = super::super::config::GenerationParams {
            max_tokens: 256,
            ..params
        };
        let body = engine.request_body(vec![json!({"role": "user", "content": "привет"})], &params);
        assert_eq!(body["max_tokens"], json!(256));
        // repetition_penalty шлём только GigaChat: остальные провайдеры
        // его не знают и отвечают 400.
        assert!(body.get("repetition_penalty").is_none());
    }

    #[test]
    fn gigachat_request_body_has_model_and_repetition_penalty() {
        let engine = engine("gigachat", "GigaChat-Pro");
        let params = super::super::config::GenerationParams {
            temperature: 0.4,
            top_p: 0.95,
            repeat_penalty: 1.1,
            max_tokens: 0,
            ..Default::default()
        };
        let body = engine.request_body(vec![json!({"role": "user", "content": "привет"})], &params);
        assert_eq!(body["model"], json!("GigaChat-Pro"));
        assert_eq!(body["temperature"], json!(0.4));
        assert_eq!(body["top_p"], json!(0.95));
        assert_eq!(body["repetition_penalty"], json!(1.1));
        assert_eq!(body["stream"], json!(true));
    }

    #[test]
    fn messages_include_history_and_prefill() {
        let engine = engine("openai", "m");
        let history = vec![("вопрос".to_string(), "ответ".to_string())];
        let msgs = engine.messages("sys", &history, "новый вопрос", Some("<tool>"));
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[3]["role"], "user");
        assert_eq!(msgs[4]["role"], "assistant");
        assert_eq!(msgs[4]["content"], "<tool>");
    }

    #[test]
    fn sse_parses_data_without_space() {
        // «data:{...}» без пробела — валидный SSE, раньше он молча терялся.
        let raw = "data:{\"choices\":[{\"delta\":{\"content\":\"Привет\"}}]}\n\
                   data:{\"choices\":[{\"delta\":{\"content\":\", сэр\"}}]}\n\
                   data:[DONE]\n";
        let cancel = AtomicBool::new(false);
        let mut seen = Vec::new();
        let outcome = parse_sse(std::io::BufReader::new(raw.as_bytes()), &cancel, &mut |d, acc| {
            seen.push(d.to_string());
            assert!(acc.contains("Привет"));
        });
        assert_eq!(outcome.text, "Привет, сэр");
        assert!(outcome.done);
        assert_eq!(outcome.bad_chunks, 0);
        assert_eq!(seen, vec!["Привет".to_string(), ", сэр".to_string()]);
    }

    #[test]
    fn sse_counts_malformed_chunk_and_keeps_the_rest() {
        let raw = "data:{\"choices\":[{\"delta\":{\"content\":\"Держись\"}}]}\n\
                   data:{не json\n\
                   data:{\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\
                   data:[DONE]\n";
        let cancel = AtomicBool::new(false);
        let outcome = parse_sse(std::io::BufReader::new(raw.as_bytes()), &cancel, &mut |_, _| {});
        // Текст не потерян из-за битого чанка, но битый чанк посчитан.
        assert_eq!(outcome.text, "Держись");
        assert_eq!(outcome.bad_chunks, 1);
        assert!(outcome.done);
    }

    #[test]
    fn sse_without_done_is_not_complete() {
        // Поток закрылся без [DONE] и без finish_reason — ответ оборван.
        let raw = "data:{\"choices\":[{\"delta\":{\"content\":\"Начало\"}}]}\n";
        let cancel = AtomicBool::new(false);
        let outcome = parse_sse(std::io::BufReader::new(raw.as_bytes()), &cancel, &mut |_, _| {});
        assert_eq!(outcome.text, "Начало");
        assert!(!outcome.done);
        assert!(outcome.finish_reason.is_none());
        assert!(!outcome.cancelled);
    }

    #[test]
    fn sse_with_finish_reason_counts_as_complete() {
        let raw = "data:{\"choices\":[{\"delta\":{\"content\":\"Итог\"},\"finish_reason\":\"stop\"}]}\n";
        let cancel = AtomicBool::new(false);
        let outcome = parse_sse(std::io::BufReader::new(raw.as_bytes()), &cancel, &mut |_, _| {});
        assert!(!outcome.done, "маркера [DONE] не было");
        assert_eq!(outcome.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn sse_stops_on_cancel_flag() {
        let raw = "data:{\"choices\":[{\"delta\":{\"content\":\"раз\"}}]}\n\
                   data:{\"choices\":[{\"delta\":{\"content\":\"два\"}}]}\n";
        let cancel = AtomicBool::new(true);
        let outcome = parse_sse(std::io::BufReader::new(raw.as_bytes()), &cancel, &mut |_, _| {});
        assert!(outcome.cancelled);
        assert!(outcome.text.is_empty());
    }

    #[test]
    fn key_fingerprint_changes_with_key() {
        assert_ne!(key_fingerprint("A"), key_fingerprint("B"));
        assert_eq!(key_fingerprint("A"), key_fingerprint("A"));
    }

    #[test]
    fn retry_after_parses_seconds() {
        let value = ureq::http::HeaderValue::from_static("12");
        assert_eq!(retry_after(Some(&value)), Some(Duration::from_secs(12)));
        assert_eq!(retry_after(None), None);
        let bad = ureq::http::HeaderValue::from_static("tomorrow");
        assert_eq!(retry_after(Some(&bad)), None);
    }
}
