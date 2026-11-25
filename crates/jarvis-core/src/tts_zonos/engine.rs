//! Движок синтеза ZONOS3: текст → коды → (декодер) → звук.
//!
//! Генерация устроена как в эталоне: промпт (строка голоса, три строки
//! условий, текст, кадры тишины) прогоняется через бэкбон одним проходом,
//! затем идёт авторегрессия по кадрам.
//!
//! Delay-паттерн: строка потока `t` несёт коды `code[t][0]` … `code[t-9][9]` —
//! то есть книга `c` кадра `f` лежит в строке `f + c`. Поэтому шаг
//! генерации выдаёт сразу десять кодов (по одному на книгу), и каждый из
//! них принадлежит своему кадру. Кадр считается готовым, когда пришла его
//! десятая книга (строка `f + 9`).
//!
//! Cold start: пока «треугольник» задержки не заполнен, книги с номером
//! больше номера строки заполняются паддингом (`audio_pad_id`) — так же,
//! как в ggml-сервере и web-движке.

use super::backbone::Backbone;
use super::config::ZonosConfig;
use super::gguf::GgufFile;
use super::prompt::{PromptSpec, build_prompt};
use super::sampler::{Sampler, Sampling};
use super::tokenizer::TextTokenizer;
use anyhow::{Context, Result, bail};
use candle_core::{Device, IndexOp, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};

/// Готовый к декодированию блок кадров в потоковом синтезе.
pub struct StreamBlock {
    /// Срез кадров [slice_start..emit_end + гарант) — уже де-sheared,
    /// включает перенос контекста слева.
    pub frames: Vec<Vec<u32>>,
    /// Номер первого кадра среза (перенос — раньше emit_start).
    pub slice_start: usize,
    /// Номер первого отдаваемого кадра.
    pub emit_start: usize,
    /// Номер кадра за последним отдаваемым.
    pub emit_end: usize,
    /// Это последний блок синтеза.
    pub last: bool,
}

/// Настройки потоковой отдачи кадров.
pub struct StreamingCfg<'a> {
    /// Приёмник готовых блоков.
    pub on_block: &'a mut dyn FnMut(StreamBlock),
    /// Флаги отмены (внешний Stop/перебивание и отмена попытки). Проверяются
    /// между шагами генерации — остановка в пределах одного шага.
    pub cancel: &'a [&'a AtomicBool],
    /// Размер очередного блока, кадров.
    pub block_frames: usize,
    /// Размер первого блока, кадров (поменьше — ради быстрого старта речи).
    pub first_block: usize,
    /// Левый перенос контекста, кадров: стадия 0 MOSS видит 150 кадров
    /// назад — столько же берём в декод, чтобы блок совпадал с цельным.
    pub carry: usize,
    /// Правый гарант, кадров: iSTFT-хвост перекрывает ~1 кадр вперёд.
    pub guard: usize,
}

/// Размеры блоков и перенос контекста (кадров; кадр = 66.7 мс).
/// Левый перенос контекста считается декодером: `ZacDecoder::
/// required_left_context()` (окна стадий + семантика + ствол + запас) —
/// константы STREAM_CARRY больше нет: 150 кадров покрывали только стадию 0.
pub const STREAM_BLOCK_FRAMES: usize = 60;
pub const STREAM_FIRST_BLOCK: usize = 30;
pub const STREAM_GUARD: usize = 3;

/// Бюджет генерации: максимум кадров речи на одну фразу
/// (реестр генерации ZONOS3, generation.json «max_frames»).
pub const MAX_GENERATION_FRAMES: usize = 2000;

/// Декодер кодов в звук (ZAC-декодер).
pub trait AudioDecoder: Send + Sync {
    /// Коды кадров (де-sheared) → сэмплы на 44.1 кГц.
    fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>>;
    /// Как [`decode`], но срез начинается с абсолютного кадра `offset`:
    /// RoPE углы считаются от абсолютных позиций, кусочный декод
    /// трансляционно эквивалентен цельному (швы блоков сходятся).
    fn decode_from(&self, frames: &[Vec<u32>], offset: usize) -> Result<Vec<f32>>;
    /// Сэмплов на кадр (hop): для отбрасывания префилл-тишины.
    fn samples_per_frame(&self) -> usize;
}

/// Результат синтеза: коды кадров и, если есть декодер, звук.
pub struct Synthesis {
    /// Кадры (де-sheared): коды по книгам.
    pub frames: Vec<Vec<u32>>,
    /// Сэмплы 44.1 кГц (пусто, если декодера нет).
    pub audio: Vec<f32>,
    /// Частота дискретизации звука.
    pub sample_rate: u32,
}

/// Частота дискретизации на выходе ZONOS3.
pub const SAMPLE_RATE: u32 = 44100;

/// Устройство вычислений для TTS. По умолчанию CPU: Metal быстрее, но на
/// этом бэкбоне заметно портит генерацию (ASR теряет слова), поэтому он
/// включается только явно — `JARVIS_TTS_METAL=1`.
pub fn device() -> Device {
    #[cfg(target_os = "macos")]
    if std::env::var_os("JARVIS_TTS_METAL").is_some() {
        return Device::new_metal(0).unwrap_or(Device::Cpu);
    }
    Device::Cpu
}

/// Устройство ZAC-декодера. Декодер превращает уже выбранные коды в звук и
/// не влияет на текст, поэтому по умолчанию Metal: на нём декодер вдвое
/// быстрее CPU. `JARVIS_TTS_DECODER_CPU=1` возвращает CPU.
pub fn decoder_device() -> Device {
    #[cfg(target_os = "macos")]
    if std::env::var_os("JARVIS_TTS_DECODER_CPU").is_none() {
        return Device::new_metal(0).unwrap_or(Device::Cpu);
    }
    Device::Cpu
}

/// Движок синтеза.
pub struct ZonosEngine {
    pub cfg: ZonosConfig,
    pub tokenizer: TextTokenizer,
    pub backbone: Backbone,
    pub device: Device,
}

impl ZonosEngine {
    /// Загрузить движок из GGUF-бэкбона.
    pub fn load(backbone: &GgufFile, device: Device) -> Result<Self> {
        let cfg = ZonosConfig::from_gguf(backbone)?;
        let tokenizer = TextTokenizer::from_json(
            &cfg.tokenizer_json,
            cfg.text_offset,
            cfg.bos_id,
            cfg.eos_id,
            cfg.text_vocab.saturating_sub(cfg.text_offset as usize),
        )?;
        let backbone = Backbone::load(cfg.clone(), backbone, &device)?;
        Ok(Self {
            cfg,
            tokenizer,
            backbone,
            device,
        })
    }

    /// Синтез: текст → коды кадров (и звук, если передан декодер).
    ///
    /// `speaker` — 192-мерный эмбеддинг голоса (из спикер-энкодера);
    /// `buckets` — бакеты условий: громкость, f_max, темп речи;
    /// `cfg_scale` — офлайн-усилитель клонирования: guided = uncond +
    /// scale·(cond − uncond). В бою не используется (всегда 1.0): даёт
    /// +0.04–0.065 схожести голоса, но ~2× дороже и портит текст.
    pub fn synthesize(
        &mut self,
        text: &str,
        speaker: Option<&[f32]>,
        buckets: [usize; 3],
        sampling: &Sampling,
        cfg_scale: f32,
        decoder: Option<&dyn AudioDecoder>,
    ) -> Result<Synthesis> {
        self.synthesize_impl(text, speaker, buckets, sampling, cfg_scale, decoder, None, None, &[], None)
    }

    /// Как [`synthesize`], но с опорными кодами голоса (режим reference
    /// оригинального репозитория): промпт строится по кодам, спикер-строка
    /// не используется. Хвост 30 кадров кодов подаётся декодеру как левый
    /// контекст (decode_context_frames реестра) — вызовами voice.rs.
    pub fn synthesize_with_codes(
        &mut self,
        text: &str,
        speaker: Option<&[f32]>,
        buckets: [usize; 3],
        sampling: &Sampling,
        cfg_scale: f32,
        decoder: Option<&dyn AudioDecoder>,
        reference_codes: Option<&[u32]>,
        reference_tail: &[u32],
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Synthesis> {
        self.synthesize_impl(
            text,
            speaker,
            buckets,
            sampling,
            cfg_scale,
            decoder,
            None,
            reference_codes,
            reference_tail,
            cancel,
        )
    }

    /// Потоковый синтез: готовые кадры отдаются блоками по мере генерации,
    /// звук декодер собирает сам (параллельно с бэкбоном — декодер живёт
    /// на своём устройстве). Речь можно начинать воспроизводить до конца
    /// генерации: первый блок готов через ~2 с аудио вместо всего клипа.
    ///
    /// Блоки вырезаются из каузального декодера без швов: декод идёт с
    /// левым переносом контекста (стадия 0 MOSS видит 150 кадров назад) и
    /// правым гарантом на iSTFT-хвост; пограничные кадры, чей хвост ещё
    /// не полон, в выдачу не попадают.
    pub fn synthesize_streaming(
        &mut self,
        text: &str,
        speaker: Option<&[f32]>,
        buckets: [usize; 3],
        sampling: &Sampling,
        cfg_scale: f32,
        reference_codes: Option<&[u32]>,
        streaming: StreamingCfg<'_>,
    ) -> Result<Synthesis> {
        self.synthesize_impl(
            text,
            speaker,
            buckets,
            sampling,
            cfg_scale,
            None,
            Some(streaming),
            reference_codes,
            &[],
            None,
        )
    }

    fn synthesize_impl(
        &mut self,
        text: &str,
        speaker: Option<&[f32]>,
        buckets: [usize; 3],
        sampling: &Sampling,
        cfg_scale: f32,
        decoder: Option<&dyn AudioDecoder>,
        mut streaming: Option<StreamingCfg<'_>>,
        reference_codes: Option<&[u32]>,
        reference_tail: &[u32],
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Synthesis> {
        let cfg = self.cfg.clone();
        let ncb = cfg.n_codebooks;
        // CFG применим только к клонированию (как в эталоне --cfg-scale).
        let cfg_scale = if speaker.is_some() { cfg_scale } else { 1.0 };

        let text_ids = self.tokenizer.text_ids(text)?;
        if text_ids.is_empty() {
            bail!("Пустой текст для синтеза");
        }

        let silence_prefill = cfg.silence_prefill_frames.min(super::prompt::SILENCE_PREFILL_MAX);
        // Клонирование: опорные коды (как в референсе, --speaker-mode
        // reference) сильнее эмбеддинга и несовместимы с ним по промпту.
        let spec = PromptSpec {
            text_ids: &text_ids,
            speaker,
            buckets,
            silence_prefill,
            reference_codes,
        };
        let prompt = build_prompt(&cfg, &spec)?;
        let prefill_rows = prompt.ids.len();
        // Строка, с которой начинается аудиопоток (первый кадр тишины).
        let silence_start = prefill_rows - silence_prefill;

        // Twin для CFG: тот же текст/тишина, но без спикера и с неизвестными
        // условиями — «neutral»-ветка, от которой отталкивается усиление.
        let mut twin = None;
        if cfg_scale != 1.0 {
            let twin_spec = PromptSpec {
                text_ids: &text_ids,
                speaker: None,
                buckets: [
                    cfg.condition_bucket_counts[0],
                    cfg.condition_bucket_counts[1],
                    cfg.condition_bucket_counts[2],
                ],
                silence_prefill,
                reference_codes: None,
            };
            let twin_prompt = build_prompt(&cfg, &twin_spec)?;
            let mut twin_backbone = self.backbone.clone();
            twin_backbone.reset();
            let tl = twin_backbone.forward(&twin_prompt.ids, None, &twin_prompt.cond_rows)?;
            twin = Some((
                twin_backbone,
                tl.i((twin_prompt.ids.len() - 1) as usize)?.to_vec2::<f32>()?,
            ));
        }

        self.backbone.reset();
        let logits = self.backbone.forward(
            &prompt.ids,
            speaker.map(|s| (s, prompt.speaker_row.unwrap_or(0))),
            &prompt.cond_rows,
        )?;
        let mut last_logits = logits.i((prefill_rows - 1) as usize)?.to_vec2::<f32>()?;

        let mut sampler = Sampler::new(
            sampling.clone(),
            ncb,
            cfg.codebook_size,
            cfg.eoa_id,
        );
        // Dev-хак: JARVIS_ZONOS_MIN_FRAMES задаёт минимальное число кадров речи.
        if let Ok(v) = std::env::var("JARVIS_ZONOS_MIN_FRAMES") {
            if let Ok(n) = v.parse() {
                sampler.min_frames = n;
            }
        }
        // Поток кадров: сначала кадры тишины, потом сгенерированные строки.
        let mut stream_rows: Vec<Vec<u32>> = Vec::new();
        for k in 0..silence_prefill {
            stream_rows.push(prompt.ids[silence_start + k][..ncb].to_vec());
        }

        let mut eos_frame: Option<usize> = None;
        let mut countdown = 0usize;
        // Бюджет генерации — реестр генерации ZONOS3 (generation.json):
        // «max_frames: 2000», silence_prefill уже ограничен SILENCE_PREFILL_MAX.
        let max_frames = MAX_GENERATION_FRAMES;
        let gen_started = std::time::Instant::now();

        // Потоковое состояние: де-sheared кадры и граница уже отданного.
        // Отдача начинается после префилл-тишины: офлайн-путь эти кадры
        // вырезает (samples.drain(..silence_prefill * spf)), стрим — не отдает.
        let mut frames: Vec<Vec<u32>> = Vec::new();
        let mut emitted = silence_prefill;
        let mut first_block = true;
        let mut cancelled = false;

        for step in 0..max_frames {
            // Граница контекста: длина промпта проверена при сборке, но
            // генерация добавляет строку на каждый шаг — обрываем чинно
            // (как в референсе: бюджет генерации ограничен контекстом).
            if prefill_rows + step >= cfg.max_seqlen {
                break;
            }
            // Внешняя отмена офлайн-синтеза: обрываем генерацию, частичные
            // кадры уходят вызывающему (в бой — с флагом Stop в плеере).
            if cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
                cancelled = true;
                break;
            }
            let stream_row = silence_prefill + step;            let mut row = vec![cfg.audio_pad_id; ncb];
            let mut eoa_at: Option<usize> = None;
            // CFG-микс: guided = uncond + scale·(cond − uncond), по каждой книге.
            let step_logits: Vec<Vec<f32>> = match &twin {
                None => last_logits.clone(),
                Some((_, twin_logits)) => {
                    let mut g = Vec::with_capacity(ncb);
                    for cb in 0..ncb {
                        let c = &last_logits[cb];
                        let u = &twin_logits[cb];
                        g.push(
                            u.iter()
                                .zip(c)
                                .map(|(&un, &co)| un + cfg_scale * (co - un))
                                .collect(),
                        );
                    }
                    g
                }
            };
            for cb in 0..ncb {
                // Cold start: книги, чья «задержка» ещё не началась, — паддинг.
                if cb > stream_row {
                    row[cb] = cfg.audio_pad_id;
                    continue;
                }
                let token = sampler.sample(&step_logits[cb], cb);
                row[cb] = token;
                if token == cfg.eoa_id && eoa_at.is_none() {
                    eoa_at = Some(cb);
                }
            }
            sampler.accept(&row);
            stream_rows.push(row.clone());

            // Конец речи: EOA задаёт выровненный кадр и запускает обратный отсчёт.
            if let Some(cb) = eoa_at {
                let aligned = stream_row.saturating_sub(cb);
                if eos_frame.is_none() {
                    eos_frame = Some(aligned);
                    countdown = ncb + 1;
                }
            }
            if countdown > 0 {
                countdown -= 1;
                if countdown == 0 {
                    break;
                }
            }

            // Потоковая отдача: кадр f готов, когда пришла его последняя
            // книга (строка f + ncb − 1).
            if let Some(st) = streaming.as_mut() {
                if st.cancel.iter().any(|c| c.load(Ordering::SeqCst)) {
                    cancelled = true;
                    break;
                }
                while frames.len() + ncb <= stream_rows.len() {
                    let f = frames.len();
                    let mut frame = vec![cfg.audio_pad_id; ncb];
                    for cb in 0..ncb {
                        frame[cb] = stream_rows[f + cb][cb];
                    }
                    frames.push(frame);
                }
                let available = frames.len();
                let want = emitted + if first_block { st.first_block } else { st.block_frames };
                if available >= want + st.guard {
                    let e = want;
                    let from = emitted.saturating_sub(st.carry);
                    (st.on_block)(StreamBlock {
                        frames: frames[from..e + st.guard].to_vec(),
                        slice_start: from,
                        emit_start: emitted,
                        emit_end: e,
                        last: false,
                    });
                    emitted = e;
                    first_block = false;
                }
            }

            let mut input_row = row.clone();
            input_row.push(cfg.text_pad_id);
            let logits = self.backbone.forward(std::slice::from_ref(&input_row), None, &[])?;
            last_logits = logits.i(0)?.to_vec2::<f32>()?;
            if let Some((twin_bb, twin_logits)) = twin.as_mut() {
                let tl = twin_bb.forward(std::slice::from_ref(&input_row), None, &[])?;
                *twin_logits = tl.i(0)?.to_vec2::<f32>()?;
            }
        }

        let rows = stream_rows.len();
        let frame_count = rows.saturating_sub(ncb - 1);
        let limit = eos_frame.map(|f| f.min(frame_count)).unwrap_or(frame_count);
        let gen_secs = gen_started.elapsed().as_secs_f32();

        // Не-потоковый путь: собрать кадры разом (как раньше).
        if streaming.is_none() {
            frames.clear();
            frames.reserve(limit);
            for f in 0..limit {
                let mut frame = vec![cfg.audio_pad_id; ncb];
                for cb in 0..ncb {
                    let row_idx = f + cb;
                    if row_idx < rows {
                        frame[cb] = stream_rows[row_idx][cb];
                    }
                }
                frames.push(frame);
            }
        } else {
            // Потоковый: добить де-sheared буфер до `limit` и отдать хвост.
            while frames.len() < limit {
                let f = frames.len();
                let mut frame = vec![cfg.audio_pad_id; ncb];
                for cb in 0..ncb {
                    let row_idx = f + cb;
                    if row_idx < rows {
                        frame[cb] = stream_rows[row_idx][cb];
                    }
                }
                frames.push(frame);
            }
            if !cancelled && limit > emitted {
                let st = streaming.as_mut().expect("streaming есть");
                let from = emitted.saturating_sub(st.carry);
                (st.on_block)(StreamBlock {
                    frames: frames[from..limit].to_vec(),
                    slice_start: from,
                    emit_start: emitted,
                    emit_end: limit,
                    last: true,
                });
            }
        }

        let (audio, sample_rate) = match decoder {
            Some(dec) => {
                let dec_started = std::time::Instant::now();
                // Хвост опорных кодов (до 30 кадров, decode_context_frames
                // реестра) подмешивается слева декодеру и офлайн-путь тоже:
                // акустика старта берётся от реального голоса, офлайн и
                // стрим звучат одинаково. Число кадров — из длины массива,
                // с ограничением сверху.
                let ncb = cfg.n_codebooks;
                let tail_frames = (reference_tail.len() / ncb).min(30);
                let tail_cut = tail_frames * ncb;
                let mut decode_frames: Vec<Vec<u32>> =
                    Vec::with_capacity(tail_frames + limit);
                decode_frames.extend(reference_tail[..tail_cut].chunks(ncb).map(|c| c.to_vec()));
                decode_frames.extend(frames[..limit].iter().cloned());
                let mut samples = dec
                    .decode_from(&decode_frames, 0)
                    .context("ZAC-декодер не справился")?;
                let dec_secs = dec_started.elapsed().as_secs_f32();
                if std::env::var_os("JARVIS_VOICE_DEBUG").is_some() {
                    eprintln!(
                        "⏱ zonos: генерация {gen_secs:.1}s, декодер {dec_secs:.1}s, кадров {limit} + {tail_frames} опоры"
                    );
                }
                // Из начала уходят и сэмплы хвоста опоры, и префилл-тишина:
                // иначе съедается начало речи (декодированный хвост опоры
                // воспринимался бы как первые кадры фразы).
                let drop = (tail_frames + silence_prefill) * dec.samples_per_frame();
                let drop = drop.min(samples.len());
                samples.drain(..drop);
                studio_polish(&mut samples, SAMPLE_RATE);
                (samples, SAMPLE_RATE)
            }
            None => {
                if std::env::var_os("JARVIS_VOICE_DEBUG").is_some() && streaming.is_some() {
                    eprintln!("⏱ zonos: генерация {gen_secs:.1}s, кадров {limit}, стрим отменён: {cancelled}");
                }
                (Vec::new(), SAMPLE_RATE)
            }
        };

        Ok(Synthesis {
            frames,
            audio,
            sample_rate,
        })
    }

    /// Только коды — удобно для тестов и отладки.
    pub fn synthesize_codes(
        &mut self,
        text: &str,
        speaker: Option<&[f32]>,
        buckets: [usize; 3],
        sampling: &Sampling,
    ) -> Result<Vec<Vec<u32>>> {
        Ok(self
            .synthesize(text, speaker, buckets, sampling, 1.0, None)?
            .frames)
    }
}

/// Студийная доводка озвучки: обрезка тишины по краям, анти-щелчок (короткие
/// фейды) и выравнивание громкости до −16 дБFS по активной части — как в
/// чистой студийной записи. Пик ограничен 0.94, усиление — не более ×3.
fn studio_polish(samples: &mut Vec<f32>, rate: u32) {
    if samples.len() < rate as usize / 4 {
        return;
    }
    let peak = samples.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak <= 1e-3 {
        return;
    }
    // 1. Трим тишины по краям (порог −45 дБ от пика, поля 80 мс).
    let floor = peak * 10f32.powf(-45.0 / 20.0);
    let pad = (rate as usize * 80 / 1000).min(samples.len() / 2);
    let first = samples
        .iter()
        .position(|v| v.abs() > floor)
        .unwrap_or(0)
        .saturating_sub(pad);
    let last = samples
        .iter()
        .rposition(|v| v.abs() > floor)
        .map(|i| (i + pad).min(samples.len()))
        .unwrap_or(samples.len());
    samples.drain(last..);
    samples.drain(..first);

    // 2. Ровная громкость: −16 дБFS по активной части (выше порога).
    let floor_active = peak * 10f32.powf(-40.0 / 20.0);
    let active: Vec<&f32> = samples.iter().filter(|v| v.abs() > floor_active).collect();
    let rms = if active.len() > rate as usize / 10 {
        (active.iter().map(|v| **v).map(|v| v * v).sum::<f32>() / active.len() as f32).sqrt()
    } else {
        (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt()
    };
    if rms > 1e-5 {
        // Цель −16 дБFS; усиление ограничено 3× и пиком 0.94.
        let target = 10f32.powf(-16.0 / 20.0);
        let gain = (target / rms).min(3.0).min(0.94 / peak);
        if (gain - 1.0).abs() > 0.05 {
            for v in samples.iter_mut() {
                *v *= gain;
            }
        }
    }

    // 3. Анти-щелчок: фейды 8 мс по краям.
    let f = (rate as usize * 8 / 1000).min(samples.len() / 2);
    for (i, v) in samples[..f].iter_mut().enumerate() {
        *v *= i as f32 / f as f32;
    }
    let n = samples.len();
    for (i, v) in samples[n - f..].iter_mut().enumerate() {
        *v *= 1.0 - i as f32 / f as f32;
    }
}

/// Край блока в потоковой озвучке: первый тримит лидирующую тишину
/// (быстрее начинается речь), последний — хвостовую; средние не тримятся.
/// `Both` — вся фраза уместилась в один блок: оба края обрабатываются
/// независимо (и трим, и фейды).
pub(crate) enum BlockEdge {
    First,
    Both,
    Mid,
    Last,
}

/// Полировка одного блока потоковой озвучки. Громкость задаёт первый блок
/// (−16 дБFS по активной части, кап ×3); остальные пользуются тем же
/// усилением — без перекачки между блоками. Пик жёстко ограничен 0.94.
pub(crate) fn polish_block(samples: &mut Vec<f32>, rate: u32, gain: &mut f32, edge: BlockEdge) {
    if samples.is_empty() {
        return;
    }
    let head = matches!(edge, BlockEdge::First | BlockEdge::Both);
    let tail = matches!(edge, BlockEdge::Last | BlockEdge::Both);

    // 1. Трим тишины по краям (−45 дБ от пика, поля 80 мс).
    let peak = samples.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak > 1e-3 && (head || tail) {
        let floor = peak * 10f32.powf(-45.0 / 20.0);
        let pad = (rate as usize * 80 / 1000).min(samples.len() / 2);
        if head {
            let first = samples
                .iter()
                .position(|v| v.abs() > floor)
                .unwrap_or(0)
                .saturating_sub(pad);
            samples.drain(..first);
        }
        if tail {
            let last = samples
                .iter()
                .rposition(|v| v.abs() > floor)
                .map(|i| (i + pad).min(samples.len()))
                .unwrap_or(samples.len());
            samples.drain(last..);
        }
    }

    // 2. Громкость: первый блок нормируем к −16 дБFS, усиление запоминаем.
    // Как в studio_polish: кап ×3 и кап по пику 0.94 — иначе жёсткий clamp
    // срезал вершины (слышимые искажения) на громких блоках.
    let peak = samples.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if head && peak > 1e-3 {
        let floor_active = peak * 10f32.powf(-40.0 / 20.0);
        let active: Vec<&f32> = samples.iter().filter(|v| v.abs() > floor_active).collect();
        let rms = if active.len() > rate as usize / 10 {
            (active.iter().map(|v| **v * **v).sum::<f32>() / active.len() as f32).sqrt()
        } else {
            (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt()
        };
        if rms > 1e-5 {
            let target = 10f32.powf(-16.0 / 20.0);
            *gain = (target / rms).min(3.0).min(0.94 / peak.max(1e-6));
        }
    }
    // Ослабление применяется наравне с усилением: иначе громкий потоковый
    // путь клиппует, а офлайн (studio_polish) — нет.
    if (*gain - 1.0).abs() > 0.05 && peak > 1e-3 {
        for v in samples.iter_mut() {
            *v *= *gain;
        }
    }
    // Страховка: при усилении > капа пика на последующих блоках.
    for v in samples.iter_mut() {
        *v = v.clamp(-0.94, 0.94);
    }

    // 3. Анти-щелчок: фейды 8 мс только по внешним краям фразы (средние
    // блоки стыкуются в сплошной сигнал).
    let f = (rate as usize * 8 / 1000).min(samples.len() / 2);
    if head {
        for (i, v) in samples[..f].iter_mut().enumerate() {
            *v *= i as f32 / f as f32;
        }
    }
    if tail {
        let n = samples.len();
        for (i, v) in samples[n - f..].iter_mut().enumerate() {
            *v *= 1.0 - i as f32 / f as f32;
        }
    }
}

/// Бакеты условий по умолчанию: громкость 8, f_max 2, темп — «неизвестно».
///
/// Выбор по метрике похожести клонированного голоса (probe_buckets:
/// синтез → спикер-энкодер → косинус к референсу). На клоне «Джарвис»
/// косинус 0.82–0.84 против 0.75 на all-unknown, на «evgeny» 0.80–0.81
/// против 0.76 — стабильно лучше на двух фразах. Темп остаётся unknown:
/// конкретный бакет давал заикания (см. speed_bucket). Переопределяется
/// env: JARVIS_TTS_LOUDNESS_BUCKET / JARVIS_TTS_FMAX_BUCKET.
pub fn default_buckets(cfg: &ZonosConfig) -> [usize; 3] {
    [
        8.min(cfg.condition_bucket_counts[0].saturating_sub(1)),
        2.min(cfg.condition_bucket_counts[1].saturating_sub(1)),
        cfg.unknown_bucket(2),
    ]
}

/// Последняя строка логитов ранга 3 `[rows, n_codebooks, audio_vocab]`
/// → `[n_codebooks, audio_vocab]`.
fn last_logits_row(logits: &Tensor, rows: usize) -> Result<Vec<Vec<f32>>> {
    Ok(logits.i(rows - 1)?.to_vec2::<f32>()?)
}

/// Логиты последней строки промпта — для отладки.
pub fn last_logits(engine: &mut ZonosEngine, rows: usize) -> Result<Vec<Vec<f32>>> {
    let logits: Tensor = engine
        .backbone
        .forward(&vec![vec![engine.cfg.audio_pad_id; engine.cfg.n_codebooks + 1]; rows], None, &[])?;
    last_logits_row(&logits, rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts_zonos::prompt::SILENCE_PREFILL_MAX;
    use crate::tts_zonos::ZacDecoder;
    use std::f32::consts::PI;
    use std::sync::Arc;

    const RATE: usize = 44100;

    /// Последняя строка логитов ранга 3 [rows, ncb, vocab]: берётся по
    /// индексу rows-1 (раньше to_vec2 на ранге 3 падал ошибкой ранга).
    #[test]
    fn last_logits_row_takes_last() {
        let device = Device::Cpu;
        // [2, 3, 4]: строка r заполнена значениями 100r + 10cb + v.
        let data: Vec<f32> = (0..2usize)
            .flat_map(|r| {
                (0..3usize)
                    .flat_map(move |cb| (0..4usize).map(move |v| (100 * r + 10 * cb + v) as f32))
            })
            .collect();
        let t = Tensor::from_vec(data, &[2, 3, 4], &device).unwrap();
        let row = last_logits_row(&t, 2).unwrap();
        assert_eq!(row.len(), 3);
        assert_eq!(row[0], vec![100.0, 101.0, 102.0, 103.0]);
        assert_eq!(row[2], vec![120.0, 121.0, 122.0, 123.0]);
    }

    /// Сквозной тест (критерий приёмки №3): при одинаковом сиде кадры
    /// стрима идентичны офлайну, а склеенный из блоков PCM совпадает с
    /// офлайн-PCM (без полировки) с точностью 1e-4. Первый блок начинается
    /// после префилл-тишины: 3 кадра в эфир не идут.
    #[test]
    #[ignore = "нужны веса ZONOS3 (~1 ГБ): cargo test -p jarvis-core --lib -- --ignored streaming_pcm_matches_offline"]
    fn streaming_pcm_matches_offline() {
        let dir = crate::model_fetch::zonos3_dir();
        assert!(
            dir.join("backbone-ptq.gguf").is_file() && dir.join("zac-keep2.gguf").is_file(),
            "весов нет в {} — тест запускается только с весами",
            dir.display()
        );
        let backbone = GgufFile::open(&dir.join("backbone-ptq.gguf")).unwrap();
        let decoder = Arc::new(
            ZacDecoder::open(&dir.join("zac-keep2.gguf"), &Device::Cpu).unwrap(),
        );
        let sp = Sampling {
            seed: 7,
            temperature: 0.6,
            ..Sampling::default()
        };
        let text = "Курс установлен, скорость максимальная.";
        let buckets = [8, 2, 30];

        // Офлайн: только кадры (звук декодируем сами — без полировки).
        let offline_frames = {
            let mut engine = ZonosEngine::load(&backbone, Device::Cpu).unwrap();
            let syn = engine
                .synthesize(text, None, buckets, &sp, 1.0, None)
                .unwrap();
            syn.frames
        };

        // Стрим: собираем блоки, восстанавливаем кадры и склеиваем PCM.
        let (stream_frames, stream_pcm) = {
            let mut engine = ZonosEngine::load(&backbone, Device::Cpu).unwrap();
            let mut blocks: Vec<StreamBlock> = Vec::new();
            {
                let mut sink = |blk: StreamBlock| blocks.push(blk);
                engine
                    .synthesize_streaming(
                        text,
                        None,
                        buckets,
                        &sp,
                        1.0,
                        None,
                        StreamingCfg {
                            on_block: &mut sink,
                            cancel: &[],
                            block_frames: STREAM_BLOCK_FRAMES,
                            first_block: STREAM_FIRST_BLOCK,
                            carry: decoder.required_left_context(),
                            guard: STREAM_GUARD,
                        },
                    )
                    .unwrap();
            }
            let spf = decoder.samples_per_frame();
            let mut frames: Vec<Vec<u32>> = Vec::new();
            let mut pcm: Vec<f32> = Vec::new();
            for blk in &blocks {
                for f in blk.emit_start..blk.emit_end {
                    let i = f - blk.slice_start;
                    assert!(i < blk.frames.len(), "кадр {f} вне среза блока");
                    frames.push(blk.frames[i].clone());
                }
                let audio = decoder.decode_from(&blk.frames, blk.slice_start).unwrap();
                let from = ((blk.emit_start - blk.slice_start) * spf).min(audio.len());
                let to = ((blk.emit_end - blk.slice_start) * spf).min(audio.len());
                pcm.extend_from_slice(&audio[from..to]);
            }
            (frames, pcm)
        };

        // Число кадров префилл-тишины ( офлайн включает их, стрим — нет).
        let silence = {
            let probe = ZonosEngine::load(&backbone, Device::Cpu).unwrap();
            probe.cfg.silence_prefill_frames.min(SILENCE_PREFILL_MAX)
        };

        // 1. Кадры идентичны (офлайн включает префилл-тишину, стрим — нет).
        let speech_frames = &offline_frames[silence..];
        assert_eq!(
            stream_frames.len(),
            speech_frames.len(),
            "стрим отдал {} кадров, офлайн собал {} речевых",
            stream_frames.len(),
            speech_frames.len()
        );
        for (f, (s, o)) in stream_frames.iter().zip(speech_frames).enumerate() {
            assert_eq!(s, o, "кадр {f} разошёлся");
        }

        // 2. PCM совпадает: офлайн без полировки минус префилл-тишина.
        let expected = decoder.decode(&offline_frames).unwrap();
        let expected = &expected[(silence * decoder.samples_per_frame()).min(expected.len())..];
        assert_eq!(
            stream_pcm.len(),
            expected.len(),
            "длины PCM разошлись: стрим {}, офлайн {}",
            stream_pcm.len(),
            expected.len()
        );
        let max_diff = stream_pcm
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, |m, d| m.max(d));
        assert!(
            max_diff < 1e-4,
            "склеенный PCM стрима разошёлся с офлайном: {max_diff}"
        );
    }

    /// Косинус: ненулевые края, чтобы фейды были заметны на границах.
    fn tone(len: usize, amp: f32) -> Vec<f32> {
        (0..len)
            .map(|i| amp * (2.0 * PI * 220.0 * i as f32 / RATE as f32).cos())
            .collect()
    }

    /// Вся фраза одним блоком: фейд-ин и фейд-аут есть с обеих сторон,
    /// иначе на концах короткой реплики щелчки.
    #[test]
    fn both_edge_fades_both_sides() {
        let mut buf = tone(RATE, 0.5);
        let mut gain = 1.0f32;
        polish_block(&mut buf, RATE as u32, &mut gain, BlockEdge::Both);
        let f = RATE * 8 / 1000;
        assert!(buf[0].abs() < 0.01, "нет фейд-ина: {}", buf[0]);
        assert!(buf[f - 1].abs() < 0.5, "фейд-ин не растёт: {}", buf[f - 1]);
        let n = buf.len();
        assert!(buf[n - 1].abs() < 0.01, "нет фейд-аута: {}", buf[n - 1]);
    }

    /// `Both` тримит тишину с обеих сторон (короткая реплика «Да, сэр.»).
    #[test]
    fn both_edge_trims_silence_on_both_sides() {
        let mut buf = vec![0.0001f32; RATE / 5];
        buf.extend(tone(3 * RATE / 5, 0.5));
        buf.extend(vec![0.0001f32; RATE / 5]);
        let pre = buf.len();
        let speech = 3 * RATE / 5;
        let mut gain = 1.0f32;
        polish_block(&mut buf, RATE as u32, &mut gain, BlockEdge::Both);
        // Тишину сняли с обеих сторон — остались речь и поля по 80 мс.
        assert!(buf.len() < pre - RATE / 5, "хвост не тримнут: {} против {pre}", buf.len());
        assert!(
            buf.len() > speech + RATE / 20,
            "отрезали речь: {} из {pre}", buf.len()
        );
    }

    /// Средний блок без краёв: ни фейдов, ни трима — блоки стыкуются в сплошной сигнал.
    #[test]
    fn mid_edge_has_no_fades_or_trim() {
        let mut buf = tone(RATE, 0.5);
        let mut gain = 1.0f32;
        polish_block(&mut buf, RATE as u32, &mut gain, BlockEdge::Mid);
        assert_eq!(buf.len(), RATE, "средний блок не должен триматься");
        assert!(buf[0].abs() > 0.4, "лишний фейд-ин: {}", buf[0]);
        assert!(
            buf[buf.len() - 1].abs() > 0.4,
            "лишний фейд-аут: {}",
            buf[buf.len() - 1]
        );
    }

    /// `Both` задаёт громкость как `First`: первый блок нормируется к −16 дБFS
    /// (тихий сигнал — усиление, кап ×3).
    #[test]
    fn both_edge_sets_gain_like_first() {
        let mut buf = tone(RATE, 0.1);
        let mut gain = 1.0f32;
        polish_block(&mut buf, RATE as u32, &mut gain, BlockEdge::Both);
        let peak = buf.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let target = 10f32.powf(-16.0 / 20.0);
        // RMS косинуса = amp/√2; усиление должно подвести rms к цели.
        assert!(
            (peak - target * 2.0f32.sqrt()).abs() < 0.1,
            "громкость не нормирована: пик {peak}"
        );
        assert!((gain - 1.0).abs() > 0.05, "усиление не запомнилось: {gain}");
    }

    /// Громкий блок: усиление ограничено пиком, а не жёстким clamp —
    /// вершины не срезаются «полками», искажений нет.
    #[test]
    fn loud_block_gain_limited_no_clipping() {
        let mut buf = tone(RATE, 2.0);
        let mut gain = 1.0f32;
        polish_block(&mut buf, RATE as u32, &mut gain, BlockEdge::Both);
        // Ослабление применилось (раньше `if *gain > 1.0` его пропускал).
        assert!(gain < 0.9, "ослабление не применилось: gain {gain}");
        let peak = buf.iter().fold(0f32, |m, &v| m.max(v.abs()));
        assert!(peak <= 0.94 + 1e-3, "пик уехал: {peak}");
        // Полки: подряд ≥3 сэмплов на жёстком пределе — признак клипа.
        let shelves = buf
            .windows(3)
            .filter(|w| w.iter().all(|v| v.abs() >= 0.939_999))
            .count();
        assert_eq!(shelves, 0, "вершины срезаны clamp'ом: {shelves} полок");
    }

    /// Потоковый и офлайн-пути ограничивают громкость одинаково:
    /// усиление = min(цель/rms, 3, 0.94/пик) в обоих.
    #[test]
    fn polish_matches_studio_gain() {
        // Один и тот же сигнал через оба полировщика.
        let mk = || tone(RATE, 1.2);
        let mut streaming = mk();
        let mut gain = 1.0f32;
        polish_block(&mut streaming, RATE as u32, &mut gain, BlockEdge::Both);
        let mut offline = mk();
        studio_polish(&mut offline, RATE as u32);
        let peak_s = streaming.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let peak_o = offline.iter().fold(0f32, |m, &v| m.max(v.abs()));
        // Стрим фейдит края, поэтому сравниваем пики с допуском на фейд.
        assert!(
            (peak_s - peak_o).abs() < 0.05,
            "пики разошлись: стрим {peak_s}, офлайн {peak_o}"
        );
    }
}
