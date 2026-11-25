//! Аудио: захват с микрофона, даунмикс, ресемплинг, воспроизведение WAV.

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::WavReader;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// macOS-разрешение на микрофон (TCC).
///
/// При запрете macOS не возвращает ошибку cpal: HAL молча отдаёт тишину —
/// wake word не срабатывает, индикатор микрофона не горит, интерфейс выглядит
/// живым. Проверяем статус через AVFoundation, при первом запуске сами
/// показываем системный запрос, при запрете падаем с внятным сообщением.
#[cfg(target_os = "macos")]
mod mic_permission {
    // Грузим фреймворк в процесс, иначе класса AVCaptureDevice в рантайме нет.
    #[link(name = "AVFoundation", kind = "framework")]
    unsafe extern "C" {}

    use block2::RcBlock;
    use objc2::runtime::{AnyClass, Bool};
    use objc2_foundation::NSString;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};

    // GCD: системный TCC-диалог надёжно показывается только из главного
    // потока GUI-приложения — запрос из фонового потока может молча не
    // появиться (наблюдалось на macOS 26). dispatch живёт в libSystem.
    unsafe extern "C" {
        static _dispatch_main_q: u8;
        fn dispatch_async(queue: *mut c_void, work: *mut c_void);
    }

    /// AVAuthorizationStatus.
    const NOT_DETERMINED: isize = 0;
    const RESTRICTED: isize = 1;
    const DENIED: isize = 2;
    const AUTHORIZED: isize = 3;

    /// Запрос уже отправляли: системный диалог показывается один раз,
    /// дальше только опрашиваем статус.
    static REQUEST_FIRED: AtomicBool = AtomicBool::new(false);

    fn status() -> Option<isize> {
        let cls = AnyClass::get(c"AVCaptureDevice")?;
        // AVMediaTypeAudio — FourCC "soun" в виде NSString.
        let media = NSString::from_str("soun");
        Some(unsafe { objc2::msg_send![cls, authorizationStatusForMediaType: &*media] })
    }

    /// Показывает системный диалог «разрешить доступ к микрофону?».
    /// Ответ читаем опросом статуса — блок нужен только формально.
    /// Уходит в главный поток через GCD.
    fn fire_request() {
        let Some(cls) = AnyClass::get(c"AVCaptureDevice") else {
            return;
        };
        let media = NSString::from_str("soun");
        let block = RcBlock::new(move || {
            let handler = RcBlock::new(|_granted: Bool| {});
            unsafe {
                let _: () = objc2::msg_send![
                    cls,
                    requestAccessForMediaType: &*media,
                    completionHandler: &*handler
                ];
            }
        });
        unsafe {
            dispatch_async(
                std::ptr::addr_of!(_dispatch_main_q) as *mut c_void,
                &*block as *const _ as *mut c_void,
            );
        }
    }

    /// Шаг конечного автомата доступа: true — доступ есть, false — ждём
    /// ответа пользователя или запрещено. Вызывается из цикла запуска
    /// движка; при первом notDetermined показывает системный запрос.
    pub fn access_step() -> bool {
        let Some(st) = status() else {
            return true; // не смогли определить — не блокируем запуск
        };
        match st {
            AUTHORIZED => true,
            NOT_DETERMINED => {
                if !REQUEST_FIRED.swap(true, Ordering::SeqCst) {
                    fire_request();
                }
                false
            }
            DENIED | RESTRICTED => false,
            _ => false, // новые значения статуса трактует консервативно
        }
    }

    /// Для диагностики: true — authorized, false — denied или ждём ответа,
    /// None — статус не определился.
    pub fn authorized() -> Option<bool> {
        status().map(|st| st == AUTHORIZED)
    }

    /// Статус строкой для UI: authorized | denied | pending | unknown.
    pub fn status_str() -> &'static str {
        match status() {
            Some(AUTHORIZED) => "authorized",
            Some(NOT_DETERMINED) => "pending",
            Some(DENIED | RESTRICTED) => "denied",
            _ => "unknown",
        }
    }

    /// Повторный запрос из Настроек: не завязан на REQUEST_FIRED, поэтому
    /// показывает системный диалог снова, если статус notDetermined.
    /// После отказа macOS диалог не показывает — вернётся "denied",
    /// и вызывающий откроет Системные настройки.
    pub fn request_access() -> &'static str {
        if status() == Some(NOT_DETERMINED) {
            fire_request();
            "pending"
        } else {
            status_str()
        }
    }
}

/// Гейт перед открытием входного потока: при явном запрете — ошибка с
/// инструкцией, а не тишина в канале.
#[cfg(target_os = "macos")]
fn ensure_mic_permission() -> Result<()> {
    match mic_permission::authorized() {
        Some(true) => Ok(()),
        Some(false) => bail!(
            "Нет доступа к микрофону — macOS запрещает запись. Открой Системные настройки → \
             Конфиденциальность и безопасность → Микрофон, включи Jarvis (или свой терминал), \
             затем перезапусти приложение."
        ),
        None => Ok(()),
    }
}

/// Шаг автомата доступа к микрофону для цикла запуска движка: true —
/// доступ есть, false — ждём ответа пользователя (системный запрос
/// показан) или запрещено.
#[cfg(target_os = "macos")]
pub fn mic_access_step() -> bool {
    mic_permission::access_step()
}

/// Текущий статус доступа к микрофону: Some(true) — разрешён,
/// Some(false) — запрещён или ждём ответа, None — не удалось определить.
/// Публичный для диагностики (пример mic_check).
#[cfg(target_os = "macos")]
pub fn mic_authorized() -> Option<bool> {
    mic_permission::authorized()
}

/// Статус доступа для UI настроек: "authorized" | "denied" | "pending" |
/// "unknown" (pending — системный запрос показан и не отвечен).
#[cfg(target_os = "macos")]
pub fn mic_status() -> &'static str {
    mic_permission::status_str()
}

/// Повторный запрос доступа из Настроек. При notDetermined показывает
/// системный диалог снова; после отказа macOS диалог уже не показывает —
/// вызывающий открывает Системные настройки.
#[cfg(target_os = "macos")]
pub fn mic_request_access() -> &'static str {
    mic_permission::request_access()
}

/// iOS: аудио-сессия (AVAudioSession) и разрешение на микрофон.
///
/// На iPhone cpal-стрим без активной сессии открывается, но маршрут система
/// не выдаёт: запись молчит, вывод не слышен. Перед входным и выходным
/// потоком выставляем категорию PlayAndRecord (динамик вместо телефонного
/// наушника, Bluetooth-гарнитуры разрешены) и активируем сессию.
#[cfg(target_os = "ios")]
mod ios_session {
    #![allow(deprecated)] // recordPermission устарел в пользу AVAudioApplication (iOS 17+);
    // старый путь работает на всех поддерживаемых версиях iOS, а не только свежих.

    use anyhow::{Context, Result, bail};
    use objc2::rc::Retained;
    use objc2::runtime::Bool;
    use objc2_avf_audio::{
        AVAudioSession, AVAudioSessionCategoryOptions, AVAudioSessionRecordPermission,
    };
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Запрос уже отправляли: системный диалог показывается один раз,
    /// дальше только опрашиваем статус.
    static REQUEST_FIRED: AtomicBool = AtomicBool::new(false);

    fn session() -> Retained<AVAudioSession> {
        unsafe { AVAudioSession::sharedInstance() }
    }

    /// Активирует сессию: запись + воспроизведение, вывод на динамик.
    /// Повторные вызовы безопасны — setCategory/setActive идемпотентны.
    pub fn activate() -> Result<()> {
        let session = session();
        unsafe {
            let Some(category) = objc2_avf_audio::AVAudioSessionCategoryPlayAndRecord else {
                bail!("категория PlayAndRecord недоступна в этой версии iOS");
            };
            session
                .setCategory_withOptions_error(
                    &category,
                    AVAudioSessionCategoryOptions::DefaultToSpeaker
                        | AVAudioSessionCategoryOptions::AllowBluetoothHFP,
                )
                .context("не задать категорию аудио-сессии")?;
            session
                .setActive_error(true)
                .context("не активировать аудио-сессию")?;
        }
        Ok(())
    }

    /// Статус разрешения на микрофон: "authorized" | "denied" | "pending" |
    /// "unknown" (pending — системный запрос ещё не отвечен).
    pub fn status_str() -> &'static str {
        let permission = unsafe { session().recordPermission() };
        match permission {
            AVAudioSessionRecordPermission::Granted => "authorized",
            AVAudioSessionRecordPermission::Undetermined => "pending",
            AVAudioSessionRecordPermission::Denied => "denied",
            _ => "unknown",
        }
    }

    /// Показывает системный диалог «Разрешить Jarvis доступ к микрофону?»,
    /// если статус ещё не определён. Ответ приходит асинхронно — вызывающий
    /// опрашивает [`status_str`], как на macOS.
    pub fn request() -> &'static str {
        if unsafe { session().recordPermission() } == AVAudioSessionRecordPermission::Undetermined
            && !REQUEST_FIRED.swap(true, Ordering::SeqCst)
        {
            let session = session();
            let block = block2::RcBlock::new(|_granted: Bool| {
                // Ответ читаем опросом статуса — здесь делать нечего.
            });
            unsafe { session.requestRecordPermission(&block) };
        }
        status_str()
    }
}

/// Статус доступа к микрофону для UI: "authorized" | "denied" | "pending".
#[cfg(target_os = "ios")]
pub fn mic_status() -> &'static str {
    ios_session::status_str()
}

/// Запрос доступа к микрофону (системный диалог iOS). После отказа iOS
/// диалог не показывает — вернётся "denied", вызывающий откроет Настройки.
#[cfg(target_os = "ios")]
pub fn mic_request_access() -> &'static str {
    ios_session::request()
}

/// На Linux системного диалога доступа нет: «разрешён» = есть устройство
/// ввода. Запрос доступа — та же проверка: спрашивать нечего, кроме
/// наличия самого микрофона.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn mic_status() -> &'static str {
    mic_probe()
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn mic_request_access() -> &'static str {
    mic_probe()
}

/// Есть ли микрофон: pactl (PipeWire/PulseAudio), исключая источники
/// `.monitor` — это петля вывода, а не вход. Отсутствие pactl — «не
/// удалось определить», не ложное «разрешён».
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn mic_probe() -> &'static str {
    let Ok(out) = std::process::Command::new("sh")
        .arg("-c")
        .arg("pactl list sources short 2>/dev/null")
        .output()
    else {
        return "unknown";
    };
    let has_mic = String::from_utf8_lossy(&out.stdout)
        .lines()
        .any(|l| !l.trim().is_empty() && !l.contains(".monitor"));
    if has_mic { "authorized" } else { "unknown" }
}

pub fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

pub fn downmix_f32(data: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return data.to_vec();
    }
    data.chunks(channels)
        .map(|c| c.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

/// Обработчик ошибок входного потока cpal без спама. Пропавшие буферы
/// (underrun/overrun у Bluetooth-гарнитур и при перегрузке) не фатальны —
/// поток продолжает работать, но сыпятся десятками в секунду. Логируем
/// первую ошибку и далее раз в 30 секунд.
pub fn stream_error_handler() -> impl FnMut(cpal::Error) + Send + 'static {
    stream_error_counting(Arc::new(AtomicU64::new(0)))
}

/// То же, что [`stream_error_handler`], плюс счётчик ошибок: замер фактической
/// частоты считает раунд с underrun'ами недостоверным — голодный колбэк
/// недосчитывает кадры и выдаёт фиктивно низкую «частоту устройства».
pub fn stream_error_counting(counter: Arc<AtomicU64>) -> impl FnMut(cpal::Error) + Send + 'static {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    let last = Mutex::new(None::<Instant>);
    move |e| {
        counter.fetch_add(1, Ordering::Relaxed);
        let Ok(mut last) = last.lock() else { return };
        if last.is_none_or(|t| t.elapsed() > Duration::from_secs(30)) {
            *last = Some(Instant::now());
            crate::logging::warn(format!("🎙️ Входной поток: {e} (запись продолжается)"));
        }
    }
}

pub fn downmix_i16(data: &[i16], channels: usize) -> Vec<f32> {
    data.chunks(channels)
        .map(|c| c.iter().map(|&x| x as f32 / 32768.0).sum::<f32>() / channels as f32)
        .collect()
}

pub fn downmix_u16(data: &[u16], channels: usize) -> Vec<f32> {
    data.chunks(channels)
        .map(|c| {
            c.iter()
                .map(|&x| (x as f32 - 32768.0) / 32768.0)
                .sum::<f32>()
                / channels as f32
        })
        .collect()
}

/// Линейная интерполяция без антиалиасинга — для речи достаточно.
///
/// ИСТОРИЯ: здесь была линейная интерполяция без фильтра. Она складывала
/// 8–24 кГц микрофона обратно в речевую полосу (алиасинг) и рвала фазу на
/// стыках чанков. Теперь это — качественный sinc-ресемплер с антиалиасингом
/// и сохранением состояния (см. [`crate::resample`]).
///
/// ⚠️ Для ПОТОКА (микрофон → модель, TTS → устройство) эта функция не годится:
/// она обрабатывает буфер целиком и не хранит состояние между вызовами.
/// Там нужен [`crate::resample::Resampler`].
pub fn resample_audio(samples: &[f32], src_rate: usize, target_rate: usize) -> Vec<f32> {
    crate::resample::resample_buffer(samples, src_rate, target_rate)
}

/// Открывает микрофон по умолчанию и шлёт моно-f32 чанки в канал.
///
/// Возвращает **уже запущенный** поток и фактическую частоту кадров.
/// Запуск делается здесь, а не у вызывающего кода: реальную частоту можно
/// измерить только по фактическим кадрам, а cpal до `play()` колбэк не
/// вызывает — замер «до запуска» всегда давал заявленную частоту, и
/// проверка была мёртвым кодом. Устройство может отдавать данные не на
/// заявленной частоте (встроенный микрофон на Apple Silicon физически
/// работает на ~24 кГц, а поток называется 48k) — тогда речь растягивается
/// вдвое и CTC декодирует мусор.
/// Вердикт замера фактической частоты потока.
#[derive(Debug, PartialEq)]
enum RateVerdict {
    /// Совпадает с заявленной в пределах допуска.
    Declared,
    /// Устойчиво отличается — реальная частота устройства.
    Trusted(usize),
    /// Недостоверно (голод колбэка, разброс, ошибки потока) — нужен ещё раунд.
    Inconclusive,
}

/// Допуск «совпадает с заявленной», доля от заявленной.
const RATE_TOLERANCE: f64 = 0.05;
/// Реальная частота устройства ниже этой — не частота, а артефакт замера.
const RATE_MIN_PLAUSIBLE: usize = 8000;
/// Насколько под-окна должны сойтись, чтобы поверить отклонению (max/min).
const RATE_SPREAD_MAX: f64 = 1.3;

/// Вердикт по под-оконным скоростям (кадры/сек, нулевые окна уже отброшены).
///
/// Логика: медиана в допуске от заявленной — верим заявленной (голод не
/// меняет частоту устройства). Отклонение признаём реальным (как у
/// встроенных микрофонов Apple Silicon: заявлено 48k, физически 24k) только
/// если под-окна сходятся между собой и в окне не было ошибок потока —
/// голодный колбэк недосчитывает кадры и рисует фиктивно низкую частоту.
fn rate_verdict(sub_rates: &mut Vec<f64>, declared: usize, had_errors: bool) -> RateVerdict {
    if sub_rates.len() < 3 {
        return RateVerdict::Inconclusive;
    }
    sub_rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sub_rates[sub_rates.len() / 2];
    let declared_f = declared as f64;
    if (median - declared_f).abs() <= declared_f * RATE_TOLERANCE {
        return RateVerdict::Declared;
    }
    if had_errors || (median as usize) < RATE_MIN_PLAUSIBLE || median > declared_f * 1.1 {
        return RateVerdict::Inconclusive;
    }
    let (min, max) = (sub_rates[0], sub_rates[sub_rates.len() - 1]);
    if min > 0.0 && max / min <= RATE_SPREAD_MAX {
        RateVerdict::Trusted(median.round() as usize)
    } else {
        RateVerdict::Inconclusive
    }
}

/// Измерить фактическую частоту кадров запущенного входного потока.
///
/// Замер дробится на под-окна по 250 мс: голодный колбэк (старт приложения,
/// прогрев моделей, underrun у гарнитур) даёт провальные окна, которые
/// режутся медианой. Отклонение от заявленной принимается только при
/// сходимости под-окон и без ошибок потока; иначе — повторный раунд, а
/// после трёх неудач — заявленная частота (голод — не частота устройства).
fn measure_stream_rate(counter: &AtomicU64, errors: &AtomicU64, declared: usize) -> usize {
    const WARMUP: Duration = Duration::from_millis(300);
    const WINDOW: Duration = Duration::from_millis(250);
    const WINDOWS: usize = 6;
    const ROUNDS: usize = 3;
    const SETTLE: Duration = Duration::from_millis(300);

    // Прогрев: первые кадры — переходные (заполнение буферов устройства).
    std::thread::sleep(WARMUP);
    for round in 0..ROUNDS {
        if round > 0 {
            std::thread::sleep(SETTLE);
        }
        let err_before = errors.load(Ordering::Relaxed);
        let mut rates: Vec<f64> = Vec::with_capacity(WINDOWS);
        for _ in 0..WINDOWS {
            let f0 = counter.load(Ordering::Relaxed);
            let t0 = std::time::Instant::now();
            std::thread::sleep(WINDOW);
            let dt = t0.elapsed().as_secs_f64();
            let frames = counter.load(Ordering::Relaxed).saturating_sub(f0);
            if dt > 0.0 && frames > 0 {
                rates.push(frames as f64 / dt);
            }
        }
        let had_errors = errors.load(Ordering::Relaxed) != err_before;
        match rate_verdict(&mut rates, declared, had_errors) {
            RateVerdict::Declared => return declared,
            RateVerdict::Trusted(r) => return r,
            RateVerdict::Inconclusive => {
                eprintln!(
                    "🎤 Замер частоты микрофона недостоверный (раунд {}/{}): под-окон {}, ошибок потока {} — досматриваю",
                    round + 1,
                    ROUNDS,
                    rates.len(),
                    if had_errors { "были" } else { "нет" }
                );
            }
        }
    }
    eprintln!(
        "⚠️ Фактическую частоту микрофона измерить не удалось — беру заявленную {declared} Гц"
    );
    declared
}

pub fn spawn_stream(tx: SyncSender<Vec<f32>>) -> Result<(cpal::Stream, usize)> {
    println!("🔍 Поиск аудиоустройств...");

    #[cfg(target_os = "macos")]
    ensure_mic_permission()?;
    // iOS: при явном запрете — честная ошибка (система отдала бы тишину),
    // затем активация сессии без неё cpal-стрим не получает маршрут.
    #[cfg(target_os = "ios")]
    {
        if ios_session::status_str() == "denied" {
            bail!(
                "Нет доступа к микрофону — iOS запрещает запись. Открой Настройки → \
                 Jarvis → Микрофон и включи доступ, затем вернись в приложение."
            );
        }
        ios_session::activate().context("аудио-сессия iOS не активировалась")?;
    }

    let host = cpal::default_host();

    // Перебираем ВСЕ устройства ввода (default первым): «default» может
    // указывать на профиль, который сейчас недоступен (наушники переключили
    // профиль, USB-гарнитура занята), при этом остальные устройства живы.
    let mut names: Vec<String> = Vec::new();
    let mut devices: Vec<cpal::Device> = Vec::new();
    if let Ok(list) = host.input_devices() {
        for d in list {
            let name = d
                .description()
                .map(|desc| desc.name().to_string())
                .unwrap_or_else(|_| "?".into());
            names.push(name.clone());
            devices.push(d);
        }
    }
    if devices.is_empty() {
        bail!("❌ В системе не найдено ни одного устройства ввода.");
    }
    // Default — первым кандидатом.
    if let Some(def) = host.default_input_device() {
        let def_name = def
            .description()
            .map(|desc| desc.name().to_string())
            .unwrap_or_default();
        let pos = devices.iter().position(|d| {
            d.description()
                .map(|desc| desc.name() == def_name)
                .unwrap_or(false)
        });
        if let Some(pos) = pos {
            let d = devices.remove(pos);
            devices.insert(0, d);
        } else {
            devices.insert(0, def);
        }
    }

    let mut last_error: Option<String> = None;
    for dev in devices {
        let name = dev
            .description()
            .map(|desc| desc.name().to_string())
            .unwrap_or_else(|_| "?".into());
        let sup = match dev.default_input_config() {
            Ok(sup) => sup,
            Err(e) => {
                last_error = Some(format!("{name}: {e}"));
                continue;
            }
        };
        let ch = sup.channels() as usize;
        let declared_sr = sup.sample_rate() as usize;
        let mut cfg: cpal::StreamConfig = sup.into();
        // Дефолтный буфер первым: Fixed(4096) на CoreAudio искажал захват
        // (дамп входа транскрибировался в кашу «градус три шага», репорт
        // 2026-11-21) — wake/STT на таком звуке глухи. Каденция чанков для
        // wake-воркера не критична: самопочинка теперь по длительности.
        for buffer in [cpal::BufferSize::Default, cpal::BufferSize::Fixed(4096)] {
            cfg.buffer_size = buffer.clone();
            // Счётчик кадров и ошибок для измерения фактической частоты
            // потока: раунд замера с underrun'ами недостоверен.
            let frame_counter = Arc::new(AtomicU64::new(0));
            let error_counter = Arc::new(AtomicU64::new(0));
            let counter_cb = Arc::clone(&frame_counter);
            let ch_cb = ch;
            let built = match sup.sample_format() {
                cpal::SampleFormat::F32 => dev.build_input_stream(
                    cfg.clone(),
                    {
                        let tx = tx.clone();
                        let counter = Arc::clone(&counter_cb);
                        move |d: &[f32], _| {
                            counter.fetch_add((d.len() / ch_cb.max(1)) as u64, Ordering::Relaxed);
                            let _ = tx.send(downmix_f32(d, ch));
                        }
                    },
                    crate::audio::stream_error_counting(Arc::clone(&error_counter)),
                    None,
                ),
                cpal::SampleFormat::I16 => dev.build_input_stream(
                    cfg.clone(),
                    {
                        let tx = tx.clone();
                        let counter = Arc::clone(&counter_cb);
                        move |d: &[i16], _| {
                            counter.fetch_add((d.len() / ch_cb.max(1)) as u64, Ordering::Relaxed);
                            let _ = tx.send(downmix_i16(d, ch));
                        }
                    },
                    crate::audio::stream_error_counting(Arc::clone(&error_counter)),
                    None,
                ),
                cpal::SampleFormat::U16 => dev.build_input_stream(
                    cfg.clone(),
                    {
                        let tx = tx.clone();
                        let counter = Arc::clone(&counter_cb);
                        move |d: &[u16], _| {
                            counter.fetch_add((d.len() / ch_cb.max(1)) as u64, Ordering::Relaxed);
                            let _ = tx.send(downmix_u16(d, ch));
                        }
                    },
                    crate::audio::stream_error_counting(Arc::clone(&error_counter)),
                    None,
                ),
                f => bail!("Неподдерживаемый формат микрофона: {f:?}"),
            };
            match built {
                Ok(stream) => {
                    println!(
                        "🎙️ Микрофон: «{name}» ({}ch {}Hz {:?}, буфер {:?})",
                        ch,
                        declared_sr,
                        sup.sample_format(),
                        buffer
                    );
                    // Поток запускаем ЗДЕСЬ: без запуска cpal не вызывает
                    // колбэк, кадры не идут, и замер ниже всегда возвращал
                    // заявленную частоту (проверка была мёртвым кодом).
                    if let Err(e) = stream.play() {
                        last_error = Some(format!("«{name}»: не запустить поток: {e}"));
                        continue;
                    }
                    // Автоопределение фактической частоты: устройство может
                    // отдавать данные не на заявленной (встроенный микрофон
                    // Apple Silicon физически ~24k при потоке 48k), а голодный
                    // колбэк — рисовать фиктивно низкую. JARVIS_MIC_RATE
                    // отключает определение и фиксирует частоту вручную.
                    let forced = std::env::var("JARVIS_MIC_RATE")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|r| (1000..=256000).contains(r));
                    let sr = match forced {
                        Some(r) => {
                            println!("🎙️ Частота микрофона фиксирована вручную: {r} Гц (заявлена {declared_sr})");
                            r
                        }
                        None => match measure_stream_rate(&frame_counter, &error_counter, declared_sr) {
                            r if r == declared_sr => {
                                println!("🎙️ Частота микрофона: {r} Гц (совпадает с заявленной)");
                                r
                            }
                            r => {
                                println!(
                                    "⚠️ Устройство отдаёт {r} Гц вместо заявленных {declared_sr} — ресемплинг по факту"
                                );
                                r
                            }
                        },
                    };
                    println!("✅ Аудиопоток микрофона создан и запущен");
                    return Ok((stream, sr));
                }
                Err(e) => {
                    last_error = Some(format!("«{name}»: {e}"));
                }
            }
        }
    }

    bail!(
        "❌ Ни одно устройство ввода не открылось. Устройства: {}. Последняя причина: {}",
        names.join(", "),
        last_error.unwrap_or_else(|| "неизвестно".into())
    )
}

/// Блокирующее воспроизведение WAV-файла с проверкой флага остановки.
///
/// Используется разовыми сценариями (тесты, UI-проверки); плеер движка
/// работает через [`OutputPlayer`] — один постоянный стрим.
/// Один постоянный выходной поток: WAV декодируется в буфер, колбэк устройства
/// непрерывно его вычитывает. Никаких лагов на пересоздание стрима между фразами.
///
/// Поддерживает потоковую подачу: [`OutputPlayer::enqueue`] дописывает сэмплы
/// в очередь по мере готовности, [`OutputPlayer::wait_drain`] дожидается конца.
pub struct OutputPlayer {
    queue: Arc<Mutex<Vec<f32>>>,
    consumed: Arc<std::sync::atomic::AtomicUsize>,
    /// Сколько interleaved-сэмплов выдвинуто в очередь всего (монотонно).
    pushed: std::sync::atomic::AtomicUsize,
    /// Держим стрим живым: колбэк работает, пока существует структура.
    _stream: cpal::Stream,
    sample_rate: usize,
    channels: usize,
}

impl OutputPlayer {
    pub fn new() -> Result<Self> {
        // iOS: без активной сессии выходной стрим не играет (маршрут не выдан).
        #[cfg(target_os = "ios")]
        ios_session::activate().context("аудио-сессия iOS не активировалась")?;

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("Нет устройства вывода звука")?;
        let default_config = device.default_output_config()?;

        let sample_rate = default_config.sample_rate() as usize;
        let channels = default_config.channels() as usize;

        let queue: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let consumed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let cb_queue = Arc::clone(&queue);
        let cb_consumed = Arc::clone(&consumed);

        let config = cpal::StreamConfig {
            channels: default_config.channels(),
            sample_rate: default_config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        let stream = device.build_output_stream(
            config,
            move |data: &mut [f32], _| {
                // Zero-copy: пишем прямо в буфер устройства, без
                // промежуточного Vec на каждый вызов колбэка.
                let mut q = cb_queue.lock().unwrap_or_else(|e| e.into_inner());
                let take = data.len().min(q.len());
                data[..take].copy_from_slice(&q[..take]);
                q.drain(..take);
                drop(q);
                cb_consumed.fetch_add(take, Ordering::Relaxed);
                for c in data[take..].iter_mut() {
                    *c = 0.0;
                }
            },
            move |err| eprintln!("Output stream error: {err}"),
            None,
        )?;

        stream.play()?;

        Ok(Self {
            queue,
            consumed,
            pushed: std::sync::atomic::AtomicUsize::new(0),
            _stream: stream,
            sample_rate,
            channels,
        })
    }

    /// Частота выходного устройства (сэмплы для [`OutputPlayer::enqueue`]).
    pub fn device_sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Дописывает моно-сэмплы (уже в частоте устройства) в очередь вывода.
    /// Не блокируется: воспроизведение идёт параллельно через колбэк.
    pub fn enqueue(&self, mono: &[f32]) {
        if mono.is_empty() {
            return;
        }
        let out_channels = self.channels;
        let n = mono.len() * out_channels;
        {
            // Zero-copy для моно (наш случай): данные уходят в очередь
            // одним extend, без промежуточного interleaved-вектора.
            let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            if out_channels == 1 {
                q.extend_from_slice(mono);
            } else {
                q.reserve(n);
                for &s in mono {
                    q.extend(std::iter::repeat_n(s, out_channels));
                }
            }
        }
        self.pushed.fetch_add(n, Ordering::SeqCst);
    }

    /// Сбрасывает очередь вывода (используется при остановке речи).
    pub fn clear(&self) {
        let pushed = self.pushed.load(Ordering::SeqCst);
        self.queue.lock().unwrap_or_else(|e| e.into_inner()).clear();
        // Счётчики должны сойтись, иначе wait_drain решит, что всё уже сыграно.
        self.consumed.store(pushed, Ordering::SeqCst);
    }

    /// Ждёт, пока устройство вычитает всё выдвинутое в очередь, либо не придёт стоп.
    pub fn wait_drain(&self, stop_flag: &AtomicBool) {
        loop {
            if stop_flag.load(Ordering::SeqCst) {
                self.clear();
                return;
            }
            if self.consumed.load(Ordering::Relaxed) >= self.pushed.load(Ordering::SeqCst) {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Декодирует WAV, даунмиксит, ресемплирует под выходное устройство
    /// и проигрывает целиком.
    pub fn play_wav(&self, path: &str, stop_flag: &AtomicBool) -> Result<()> {
        if !Path::new(path).exists() {
            bail!("Файл звука не найден: {path}");
        }

        let mut reader = WavReader::open(path)?;
        let spec = reader.spec();

        let raw: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)
                .collect(),
            hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        };

        let mono = downmix_f32(&raw, spec.channels as usize);
        let final_samples = if spec.sample_rate as usize != self.sample_rate {
            resample_audio(&mono, spec.sample_rate as usize, self.sample_rate)
        } else {
            mono
        };

        self.enqueue(&final_samples);
        self.wait_drain(stop_flag);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_keeps_length_and_dc() {
        let src = vec![0.5f32; 1600]; // 0.1s @ 16k
        let out = resample_audio(&src, 16_000, 48_000);
        assert_eq!(out.len(), 4800);
        // sinc-фильтр имеет стартовый выброс (переходная зона окна), поэтому
        // постоянство проверяем в середине, а не по всему буферу.
        let mid = &out[2_000..2_800];
        let worst = mid.iter().map(|s| (s - 0.5).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "DC гуляет: {worst}");
    }

    #[test]
    fn resample_same_rate_is_clone() {
        let src = vec![0.1f32, 0.2, 0.3];
        let out = resample_audio(&src, 16_000, 16_000);
        assert_eq!(out, src);
    }

    #[test]
    fn downmix_stereo_averages() {
        let out = downmix_f32(&[0.0, 1.0, 0.5, 0.5], 2);
        assert_eq!(out, vec![0.5, 0.5]);
    }

    #[test]
    fn downmix_i16_normalizes() {
        let out = downmix_i16(&[i16::MAX], 1);
        assert!((out[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn rms_zero_and_nonzero() {
        assert_eq!(calculate_rms(&[]), 0.0);
        assert_eq!(calculate_rms(&[0.0; 100]), 0.0);
        assert!(calculate_rms(&[0.5; 100]) > 0.49);
    }

    /// Вердикт замера: совпадение с заявленной в допуске 5%.
    #[test]
    fn rate_verdict_matches_declared() {
        let mut rates = vec![47_900.0, 48_100.0, 48_050.0, 47_950.0, 48_000.0, 48_020.0];
        assert_eq!(rate_verdict(&mut rates, 48_000, false), RateVerdict::Declared);
    }

    /// Реальное отклонение (встроенный микрофон Apple Silicon: заявлено 48k,
    /// физически ~24k): сходимые под-окна без ошибок — верим измерению.
    #[test]
    fn rate_verdict_trusts_consistent_deviation() {
        let mut rates = vec![24_050.0, 23_980.0, 24_010.0, 24_020.0, 23_990.0, 24_005.0];
        assert_eq!(
            rate_verdict(&mut rates, 48_000, false),
            RateVerdict::Trusted(24_010)
        );
    }

    /// Голод колбэка на старте (репорт 2026-11-22: «9509 Гц вместо 48000»
    /// при прогреве моделей): все окна занижены и скачут — недостоверно,
    /// досматриваем.
    #[test]
    fn rate_verdict_starvation_inconclusive() {
        let mut rates = vec![9_500.0, 9_800.0, 9_100.0, 12_000.0, 8_900.0, 9_400.0];
        assert_eq!(
            rate_verdict(&mut rates, 48_000, false),
            RateVerdict::Inconclusive
        );
    }

    /// Половина окон голодная, половина полная: медиана честно попадает на
    /// заявленную — полные окна говорят правду, голод не переубеждает.
    #[test]
    fn rate_verdict_half_starved_still_declared() {
        let mut rates = vec![48_000.0, 48_000.0, 9_000.0, 9_500.0, 48_000.0, 9_100.0];
        assert_eq!(rate_verdict(&mut rates, 48_000, false), RateVerdict::Declared);
    }

    /// Ошибки потока в окне замера: отклонение не признаётся, даже если
    /// под-окна сошлись — голод рисует фиктивно низкую частоту.
    #[test]
    fn rate_verdict_with_errors_inconclusive() {
        let mut rates = vec![9_509.0, 9_510.0, 9_508.0, 9_511.0];
        assert_eq!(
            rate_verdict(&mut rates, 48_000, true),
            RateVerdict::Inconclusive
        );
    }

    /// Слишком низкая «частота» — артефакт замера, не свойство устройства.
    #[test]
    fn rate_verdict_implausibly_low_inconclusive() {
        let mut rates = vec![3_000.0, 3_010.0, 2_995.0, 3_005.0];
        assert_eq!(
            rate_verdict(&mut rates, 48_000, false),
            RateVerdict::Inconclusive
        );
    }

    /// Меньше трёх под-окон — вердикта нет.
    #[test]
    fn rate_verdict_needs_three_windows() {
        let mut rates = vec![24_000.0, 24_010.0];
        assert_eq!(
            rate_verdict(&mut rates, 48_000, false),
            RateVerdict::Inconclusive
        );
    }
}
