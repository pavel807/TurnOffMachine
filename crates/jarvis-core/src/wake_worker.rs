//! Wake-детектор в отдельном потоке.
//!
//! Почему так: CTC-детектор считает окно (1,2 с звука) на Metal — это
//! сотни миллисекунд, и Metal-GPU в это время занят. Пока расчёт шёл
//! прямо в цикле микрофона, движок на нём стоял: очередь звука (128
//! чанков по ~20 мс — это секунды) разрасталась, и активационная фраза
//! обрабатывалась с задержкой. Пользователь слышал отклик через
//! три-четыре секунды после фразы, а спектр в интерфейсе отставал:
//! уровень звука тоже шлётся из этого цикла.
//!
//! Теперь детектор живёт в своём потоке. Цикл микрофона только кладёт
//! ему звук и забирает готовый результат: он никогда не ждёт GPU.
//! Задержка на распознавание ограничена самим инференсом.

use std::collections::VecDeque;
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::wake::{WakeEngine, WakeHit};

/// Порог самопочинки по ДЛИТЕЛЬНОСТИ очереди (3 с звука): если детектор
/// отстал больше, окно сбрасывается и копится заново. Дропать чанки точечно
/// НЕЛЬЗЯ — окно CTC живёт внутри детектора, дыра от пропущенного куска
/// делает его слепым ровно во время речи.
///
/// Раньше порог был в чанках (24), но размер чанка зависит от устройства и
/// драйвера: WASAPI даёт ~85 мс (24 чанка ≈ 2 с), CoreAudio — миллисекунды,
/// Bluetooth-гарнитура — ~170 мс (24 чанка ≈ 4 с). На чанках одна и та же
/// константа означала «полсекунды» или «4 секунды» на разных машинах.
const AUDIO_HEAL_SAMPLES: usize = 3 * crate::wake::SAMPLE_RATE;

/** Очередь звука для детектора: без потерь. */
struct AudioQueue {
    inner: Mutex<VecDeque<Vec<f32>>>,
    cv: Condvar,
}

impl AudioQueue {
    fn new() -> Self {
        Self {
            inner: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
        }
    }

    /// Кладёт чанк без потерь.
    fn push(&self, chunk: Vec<f32>) {
        let mut q = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        q.push_back(chunk);
        self.cv.notify_one();
    }

    /// Забирает ВСЁ накопленное одним забором (батч): воркер догоняет за
    /// один вызов детектора — он сам дробит батч по хопам. Пусто — ждёт
    /// не дольше `timeout`.
    fn pop_batch(&self, timeout: Duration) -> Option<Vec<f32>> {
        let mut q = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if q.is_empty() {
            let (guard, _) = self
                .cv
                .wait_timeout(q, timeout)
                .unwrap_or_else(|e| e.into_inner());
            q = guard;
        }
        if q.is_empty() {
            return None;
        }
        let mut joined = Vec::new();
        for chunk in q.drain(..) {
            joined.extend_from_slice(&chunk);
        }
        Some(joined)
    }

    /// Сколько сэмплов ждёт обработки (для самопочинки по длительности).
    fn len_samples(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|c| c.len())
            .sum()
    }

    fn clear(&self) {
        let mut q = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        q.clear();
    }
}

/// Команды детектору: сброс состояния или подмена модели.
enum WakeCtl {
    Reset,
    Replace(Box<WakeEngine>),
}

/// Готовый результат детектора.
pub struct WakeReply {
    /// Срабатывание активационной фразы (если было).
    pub hit: Option<WakeHit>,
    /// Счёт последнего окна — для диагностики порога.
    pub score: Option<f32>,
    /// Человекочитаемая диагностика последнего окна.
    pub diag: String,
    /// Сколько занял последний расчёт — видно, кто виноват в задержке.
    pub elapsed: Duration,
}

/// Детектор активационной фразы в своём потоке.
pub struct WakeWorker {
    audio: Arc<AudioQueue>,
    reply: Arc<Mutex<Option<WakeReply>>>,
    ctl_tx: SyncSender<WakeCtl>,
}

impl WakeWorker {
    /// Запускает поток детектора с готовым движком.
    pub fn spawn(engine: WakeEngine) -> Self {
        let audio = Arc::new(AudioQueue::new());
        let reply = Arc::new(Mutex::new(None));
        let (ctl_tx, ctl_rx) = sync_channel::<WakeCtl>(4);
        let a = Arc::clone(&audio);
        let r = Arc::clone(&reply);

        std::thread::Builder::new()
            .name("wake-detector".into())
            .spawn(move || {
                let mut det = engine;
                // Дамп входа детектора: включается JARVIS_WAKE_DUMP=1.
                // Файл — сырой PCM s16le 16 кГц mono: конвертируй в wav
                // через `ffmpeg -f s16le -ar 16000 -ac 1 -i <файл> out.wav`.
                let mut dump_writer: Option<std::fs::File> =
                    if std::env::var_os("JARVIS_WAKE_DUMP").is_some() {
                        let path = std::env::temp_dir().join("jarvis_wake_dump.pcm");
                        match std::fs::File::create(&path) {
                            Ok(f) => {
                                println!("🎧 Дамп входа wake-детектора: {}", path.display());
                                Some(f)
                            }
                            Err(e) => {
                                eprintln!("⚠️ Не открыть дамп wake: {e}");
                                None
                            }
                        }
                    } else {
                        None
                    };
                loop {
                    // Команды (сброс, смена модели) — приоритетнее звука.
                    while let Ok(cmd) = ctl_rx.try_recv() {
                        match cmd {
                            WakeCtl::Reset => {
                                det.reset();
                                a.clear();
                            }
                            WakeCtl::Replace(next) => {
                                det = *next;
                                a.clear();
                            }
                        }
                    }

                    // Самопочинка: если детектор отстал больше чем на
                    // AUDIO_HEAL_SAMPLES (3 с звука), окно всё равно
                    // бесполезно — сбрасываем и начинаем заново, но НЕ
                    // выбрасываем куски точечно: дыры в окне делали детектор
                    // слепым ровно во время речи.
                    if a.len_samples() > AUDIO_HEAL_SAMPLES {
                        det.reset();
                        a.clear();
                    }

                    let Some(chunk) = a.pop_batch(Duration::from_millis(50)) else {
                        continue;
                    };
                    // Дамп входа детектора (JARVIS_WAKE_DUMP=1): ровно то,
                    // что детектор слышит после ресемплинга — им можно
                    // прогнать wake_live_sim и отличить «плохой звук» от
                    // бага пайплайна. Сырой PCM без заголовка: поток не
                    // завершается штатно, wav-заголовок остался бы битым.
                    if let Some(writer) = dump_writer.as_mut() {
                        use std::io::Write as _;
                        let mut bytes = Vec::with_capacity(chunk.len() * 2);
                        for &s in &chunk {
                            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }
                        let _ = writer.write_all(&bytes);
                        let _ = writer.flush();
                    }
                    let started = Instant::now();
                    let hit = det.process_16k(&chunk);
                    let score = det.last_score();
                    let diag = det.describe();
                    let mut slot = r.lock().unwrap_or_else(|e| e.into_inner());
                    *slot = Some(WakeReply {
                        hit,
                        score,
                        diag,
                        elapsed: started.elapsed(),
                    });
                }
            })
            .expect("не создать поток wake-детектора");

        Self {
            audio,
            reply,
            ctl_tx,
        }
    }

    /// Отдать звук детектору. Неблокирующий: цикл микрофона не ждёт GPU.
    pub fn push_audio(&self, chunk: Vec<f32>) {
        self.audio.push(chunk);
    }

    /// Забрать свежий результат (если он есть).
    pub fn take_reply(&self) -> Option<WakeReply> {
        self.reply.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    /// Сбросить состояние детектора (после срабатывания или паузы).
    pub fn reset(&self) {
        let _ = self.ctl_tx.try_send(WakeCtl::Reset);
        self.audio.clear();
    }

    /// Подменить модель детектора (новое слово-активация).
    pub fn replace(&self, engine: WakeEngine) {
        let _ = self.ctl_tx.try_send(WakeCtl::Replace(Box::new(engine)));
    }
}
