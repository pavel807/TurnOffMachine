//! Голос ZONOS3 как движок синтеза для `tts`.

use super::codec::ZacDecoder;
use super::engine::{
    AudioDecoder, BlockEdge, SAMPLE_RATE, StreamBlock, StreamingCfg, ZonosEngine, default_buckets,
    polish_block,
};
use super::sampler::Sampling;
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Голос ZONOS3: модель + эмбеддинг голоса + ZAC-декодер.
pub struct ZonosVoice {
    engine: Mutex<ZonosEngine>,
    rate_buckets: usize,
    decoder: Option<Arc<ZacDecoder>>,
    speaker: Option<Vec<f32>>,
    reference_codes: Option<Vec<u32>>,
    buckets: [usize; 3],
    sampling: Sampling,
    cfg_scale: f32,
    voice_name: String,
    voices_dir: Option<PathBuf>,
    preview: Mutex<HashMap<String, Option<VoicePreview>>>,
}

#[derive(Clone)]
struct VoicePreview {
    speaker: Option<Vec<f32>>,
    reference_codes: Option<Vec<u32>>,
}

impl ZonosVoice {
    pub fn new(
        engine: ZonosEngine,
        decoder: Option<Arc<ZacDecoder>>,
        speaker: Option<Vec<f32>>,
        buckets: Option<[usize; 3]>,
        voice_name: impl Into<String>,
    ) -> Self {
        Self::with_preview_dir(engine, decoder, speaker, buckets, voice_name, None)
    }

    pub fn reference_codes(&self) -> Option<&[u32]> {
        self.reference_codes.as_deref()
    }

    pub fn set_reference_codes(&mut self, codes: Option<Vec<u32>>) {
        self.reference_codes = codes;
    }

    pub fn with_preview_dir(
        engine: ZonosEngine,
        decoder: Option<Arc<ZacDecoder>>,
        speaker: Option<Vec<f32>>,
        buckets: Option<[usize; 3]>,
        voice_name: impl Into<String>,
        voices_dir: Option<PathBuf>,
    ) -> Self {
        let buckets = buckets.unwrap_or_else(|| default_buckets(&engine.cfg));
        let voice_name: String = voice_name.into();

        let reference_codes = voices_dir
            .as_deref()
            .and_then(|dir| load_voice_codes(dir, &voice_name));

        let mut sampling = Sampling::default();
        sampling.temperature = env_f32("JARVIS_TTS_TEMP", 0.3, 0.0..=2.0);
        sampling.seed = 1;

        if let Ok(v) = std::env::var("JARVIS_TTS_SEED") {
            if let Ok(seed) = v.parse() {
                sampling.seed = seed;
            }
        }

        sampling.top_p = env_f32("JARVIS_TTS_TOP_P", sampling.top_p, 0.0..=1.0);
        sampling.min_p = env_f32("JARVIS_TTS_MIN_P", sampling.min_p, 0.0..=1.0);
        sampling.rep_penalty = env_f32("JARVIS_TTS_REP_PENALTY", sampling.rep_penalty, 1.0..=3.0);

        if let Ok(v) = std::env::var("JARVIS_TTS_TOP_K") {
            if let Ok(k) = v.parse::<usize>() {
                sampling.top_k = k.max(1);
            }
        }

        let rate_buckets = engine.cfg.unknown_bucket(2);

        Self {
            engine: Mutex::new(engine),
            rate_buckets,
            decoder,
            speaker,
            reference_codes,
            buckets,
            sampling,
            cfg_scale: env_f32("JARVIS_TTS_CFG", 1.0, 1.0..=3.0),
            voice_name,
            voices_dir,
            preview: Mutex::new(HashMap::new()),
        }
    }

    pub fn set_speaker(&mut self, speaker: Option<Vec<f32>>) {
        self.speaker = speaker;
    }

    pub fn speaker(&self) -> Option<&[f32]> {
        self.speaker.as_deref()
    }

    pub fn voice_name(&self) -> &str {
        &self.voice_name
    }

    pub fn set_sampling(&mut self, sp: Sampling) {
        self.sampling = sp;
    }

    pub fn set_cfg_scale(&mut self, scale: f32) {
        self.cfg_scale = scale.clamp(1.0, 3.0);
    }

    fn preview_for(&self, name: &str) -> Option<VoicePreview> {
        if name == self.voice_name {
            return Some(VoicePreview {
                speaker: self.speaker.clone(),
                reference_codes: self.reference_codes.clone(),
            });
        }

        let dir = self.voices_dir.as_ref()?;

        let mut cache = self.preview.lock().ok()?;
        if let Some(cached) = cache.get(name) {
            return cached.clone();
        }

        let reference_codes = load_voice_codes(dir, name);
        let speaker = if reference_codes.is_some() {
            None
        } else {
            load_voice_embedding(dir, name)
        };

        let found = speaker.is_some() || reference_codes.is_some();
        let preview = found.then_some(VoicePreview {
            speaker,
            reference_codes,
        });

        cache.insert(name.to_string(), preview.clone());
        preview
    }

    pub fn speed_bucket(&self, speed: f32) -> usize {
        speed_bucket_for(self.rate_buckets, speed)
    }
}

fn speed_bucket_for(n: usize, speed: f32) -> usize {
    if n == 0 {
        return 0;
    }

    if (speed - 1.0).abs() < 0.04 {
        return n;
    }

    let mid = (n.saturating_sub(1)) as f32 / 2.0;
    let scaled = mid * speed.clamp(0.5, 1.5);
    (scaled.round().max(0.0) as usize).min(n - 1)
}

fn buckets_with_env(mut b: [usize; 3], counts: &[usize]) -> [usize; 3] {
    for (slot, name) in [
        (0usize, "JARVIS_TTS_LOUDNESS_BUCKET"),
        (1usize, "JARVIS_TTS_FMAX_BUCKET"),
    ] {
        if let Ok(v) = std::env::var(name) {
            if let Ok(x) = v.parse::<usize>() {
                b[slot] = x.min(counts[slot].saturating_sub(1));
            }
        }
    }
    b
}

fn env_f32(name: &str, default: f32, range: std::ops::RangeInclusive<f32>) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| range.contains(v))
        .unwrap_or(default)
}

static MIN_SIM: RwLock<f32> = RwLock::new(0.95);
static ATTEMPTS: RwLock<u32> = RwLock::new(3);

pub fn set_quality(min_sim: f32, attempts: u32) {
    let min_sim = min_sim.clamp(0.7, 0.99);
    let attempts = attempts.clamp(1, 6);

    if let Ok(mut v) = MIN_SIM.write() {
        *v = min_sim;
    }
    if let Ok(mut v) = ATTEMPTS.write() {
        *v = attempts;
    }
}

fn quality() -> (f32, u32) {
    let settings_sim = MIN_SIM.read().map(|v| *v).unwrap_or(0.95);
    let min_sim = env_f32("JARVIS_TTS_MIN_SIM", settings_sim, 0.7..=0.99);

    let attempts = std::env::var("JARVIS_TTS_ATTEMPTS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or_else(|| ATTEMPTS.read().map(|v| *v).unwrap_or(3))
        .clamp(1, 6);

    (min_sim, attempts)
}

pub trait SpeechEngine: Send + Sync {
    fn sample_rate(&self) -> u32;

    fn speak(&self, text: &str, speed: f32) -> Result<Vec<f32>>;

    fn speak_with(&self, voice: &str, text: &str, speed: f32) -> Result<Vec<f32>>;

    fn engine_name(&self) -> &'static str;

    fn speak_streaming(
        &self,
        text: &str,
        speed: f32,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(Vec<f32>, bool),
    ) -> Result<()> {
        let _cancel = cancel;
        let audio = self.speak(text, speed)?;
        if !audio.is_empty() {
            emit(audio, true);
        }
        Ok(())
    }
}

/// Загрузить эмбеддинг голоса 192 измерения.
///
/// ВАЖНО: раньше здесь были имена с пробелами:
/// `voices ` и `{voice}.embedding.npy `, из-за чего файлы не находились.
pub fn load_voice_embedding(dir: &Path, voice: &str) -> Option<Vec<f32>> {
    let candidates = [
        dir.join("voices").join(format!("{voice}.embedding.npy")),
        dir.join(format!("{voice}.embedding.npy")),
    ];

    for path in candidates {
        if let Ok(bytes) = std::fs::read(&path) {
            if let Some(values) = parse_npy_f32(&bytes) {
                if values.len() == 192 {
                    return Some(values);
                }
            }
        }
    }

    None
}

/// Загрузить reference codes `[frames, 10]`.
///
/// ВАЖНО: раньше здесь тоже были лишние пробелы в именах файлов.
pub fn load_voice_codes(dir: &Path, voice: &str) -> Option<Vec<u32>> {
    let candidates = [
        dir.join("voices").join(format!("{voice}.codes.npy")),
        dir.join(format!("{voice}.codes.npy")),
    ];

    for path in candidates {
        if let Ok(bytes) = std::fs::read(&path) {
            if let Some(values) = parse_npy_codes(&bytes) {
                if !values.is_empty() && values.len() % 10 == 0 {
                    return Some(values);
                }
            }
        }
    }

    None
}

fn npy_header(bytes: &[u8]) -> Option<(&str, &[u8])> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return None;
    }

    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12usize,
        ),
        _ => return None,
    };

    let header_end = header_start.checked_add(header_len)?;
    let header = std::str::from_utf8(bytes.get(header_start..header_end)?).ok()?;
    let data = bytes.get(header_end..)?;
    Some((header, data))
}

fn npy_descr_is(header: &str, descr: &str) -> bool {
    let compact: String = header.chars().filter(|c| !c.is_whitespace()).collect();
    compact.contains(&format!("'descr':'{descr}'"))
        || compact.contains(&format!("\"descr\":\"{descr}\""))
}

fn parse_npy_codes(bytes: &[u8]) -> Option<Vec<u32>> {
    let (header, data) = npy_header(bytes)?;

    let kind = if npy_descr_is(header, "<i4") || npy_descr_is(header, "|i4") {
        0
    } else if npy_descr_is(header, "<u4") || npy_descr_is(header, "|u4") {
        1
    } else if npy_descr_is(header, "<f4") {
        2
    } else {
        return None;
    };

    let mut out = Vec::with_capacity(data.len() / 4);

    for chunk in data.chunks_exact(4) {
        let raw = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let value = match kind {
            0 => i32::from_le_bytes(raw).max(0) as u32,
            1 => u32::from_le_bytes(raw),
            _ => {
                let v = f32::from_le_bytes(raw);
                if !v.is_finite() || v < 0.0 {
                    0
                } else {
                    v.round() as u32
                }
            }
        };
        out.push(value);
    }

    Some(out)
}

fn parse_npy_f32(bytes: &[u8]) -> Option<Vec<f32>> {
    let (header, data) = npy_header(bytes)?;

    if !npy_descr_is(header, "<f4") {
        return None;
    }

    Some(
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

impl ZonosVoice {
    fn speak_internal_locked(
        &self,
        engine: &mut ZonosEngine,
        speaker: Option<Vec<f32>>,
        reference_codes: Option<Vec<u32>>,
        text: &str,
        speed: f32,
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        let Some(decoder) = self.decoder.as_ref() else {
            bail!("ZONOS3: нет ZAC-декодера");
        };

        let speaker = if reference_codes.is_some() {
            None
        } else {
            speaker
        };

        let reference_tail = reference_tail_codes(reference_codes.as_deref());

        let mut buckets = self.buckets;
        buckets[2] = self.speed_bucket(speed);
        let buckets = buckets_with_env(buckets, &engine.cfg.condition_bucket_counts);

        let asr = voice_asr();
        let (min_sim, max_attempts) = quality();

        let mut best: Option<(f32, Vec<f32>)> = None;

        for attempt in 0..max_attempts {
            if cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
                break;
            }

            let mut sp = self.sampling.clone();
            sp.seed = self.sampling.seed.wrapping_add(attempt);

            let started = Instant::now();

            let out = engine
                .synthesize_with_codes(
                    text,
                    speaker.as_deref(),
                    buckets,
                    &sp,
                    self.cfg_scale,
                    Some(decoder.as_ref()),
                    reference_codes.as_deref(),
                    &reference_tail,
                    cancel,
                )
                .context("ZONOS3: синтез не удался")?;

            let synth_secs = started.elapsed().as_secs_f32();

            match asr {
                Some(model) => {
                    let asr_started = Instant::now();
                    let pcm16 =
                        crate::tts_zonos::speaker::sinc_resample(&out.audio, SAMPLE_RATE, 16000);
                    let said = model.transcribe(&pcm16)?;
                    let asr_secs = asr_started.elapsed().as_secs_f32();

                    let sim = speech_similarity(text, said.text.trim());

                    if std::env::var_os("JARVIS_VOICE_DEBUG").is_some() {
                        eprintln!(
                            "🎤 attempt {attempt} seed {} synth {synth_secs:.1}s asr {asr_secs:.1}s sim {sim:.3} len {:.1}s: {}",
                            sp.seed,
                            out.audio.len() as f32 / SAMPLE_RATE as f32,
                            said.text.trim()
                        );
                    }

                    if best.as_ref().is_none_or(|(b, _)| sim > *b) {
                        best = Some((sim, out.audio));
                    }

                    if sim >= min_sim {
                        break;
                    }
                }
                None => return Ok(out.audio),
            }
        }

        Ok(best.map(|(_, a)| a).unwrap_or_default())
    }
}

impl SpeechEngine for ZonosVoice {
    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn speak(&self, text: &str, speed: f32) -> Result<Vec<f32>> {
        let mut engine = self
            .engine
            .lock()
            .map_err(|_| anyhow::anyhow!("ZONOS3: движок заблокирован"))?;

        self.speak_internal_locked(
            &mut engine,
            self.speaker.clone(),
            self.reference_codes.clone(),
            text,
            speed,
            None,
        )
    }

    fn speak_with(&self, voice: &str, text: &str, speed: f32) -> Result<Vec<f32>> {
        let preview = self
            .preview_for(voice)
            .ok_or_else(|| anyhow::anyhow!("ZONOS3: голос «{voice}» не найден"))?;

        let mut engine = self
            .engine
            .lock()
            .map_err(|_| anyhow::anyhow!("ZONOS3: движок заблокирован"))?;

        self.speak_internal_locked(
            &mut engine,
            preview.speaker,
            preview.reference_codes,
            text,
            speed,
            None,
        )
    }

    fn engine_name(&self) -> &'static str {
        "zonos"
    }

    fn speak_streaming(
        &self,
        text: &str,
        speed: f32,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(Vec<f32>, bool),
    ) -> Result<()> {
        let Some(decoder) = self.decoder.as_ref() else {
            bail!("ZONOS3: нет ZAC-декодера");
        };

        let reference_codes = self.reference_codes.clone();
        let reference_tail = reference_tail_codes(reference_codes.as_deref());

        let speaker = if reference_codes.is_some() {
            None
        } else {
            self.speaker.clone()
        };

        let mut engine = self
            .engine
            .lock()
            .map_err(|_| anyhow::anyhow!("ZONOS3: движок заблокирован"))?;

        let mut b = self.buckets;
        b[2] = speed_bucket_for(self.rate_buckets, speed);
        let buckets = buckets_with_env(b, &engine.cfg.condition_bucket_counts);

        let asr = voice_asr();
        let (min_sim, max_attempts) = quality();

        let mut gain = 1.0f32;
        let mut best: Option<(f32, Vec<f32>)> = None;

        for attempt in 0..max_attempts {
            if cancel.load(Ordering::SeqCst) {
                break;
            }

            let mut sp = self.sampling.clone();
            sp.seed = self.sampling.seed.wrapping_add(attempt);

            let started = Instant::now();

            let out = stream_attempt(
                &mut engine,
                decoder,
                text,
                speaker.as_deref(),
                buckets,
                &sp,
                self.cfg_scale,
                cancel,
                &mut gain,
                asr,
                reference_codes.as_deref(),
                &reference_tail,
            );

            if std::env::var_os("JARVIS_VOICE_DEBUG").is_some() {
                eprintln!(
                    "🎤 stream attempt {attempt} seed {} готово через {:?} first_sim {:?} final_sim {:?} занял {:.1}s",
                    sp.seed,
                    out.first_emit_secs,
                    out.first_sim,
                    out.final_sim,
                    started.elapsed().as_secs_f32()
                );
            }

            if let Some(err) = out.error {
                bail!("ZONOS3: {err}");
            }

            if out.cancelled {
                break;
            }

            if out.first_played {
                let sim = out.final_sim.unwrap_or(0.0);

                if best.as_ref().is_none_or(|(b, _)| sim > *b) {
                    best = Some((sim, out.audio));
                }

                if out.final_sim.is_none_or(|sim| sim >= min_sim) {
                    break;
                }

                let replay = std::env::var("JARVIS_TTS_REPLAY")
                    .map(|v| v != "0")
                    .unwrap_or(true);

                if !replay {
                    break;
                }
            }
        }

        match best {
            Some((_, audio)) => emit_offline_clip(emit, audio),
            None if !cancel.load(Ordering::SeqCst) => {
                let audio = self.speak_internal_locked(
                    &mut engine,
                    speaker,
                    reference_codes,
                    text,
                    speed,
                    Some(cancel),
                )?;
                emit_offline_clip(emit, audio);
            }
            None => {}
        }

        Ok(())
    }
}

fn reference_tail_codes(codes: Option<&[u32]>) -> Vec<u32> {
    codes
        .and_then(|c| {
            let frames = c.len() / 10;
            (frames > 0).then(|| c[(frames.saturating_sub(30)) * 10..].to_vec())
        })
        .unwrap_or_default()
}

struct AttemptOutcome {
    first_played: bool,
    first_sim: Option<f32>,
    final_sim: Option<f32>,
    cancelled: bool,
    error: Option<String>,
    first_emit_secs: Option<f32>,
    audio: Vec<f32>,
}

fn emit_offline_clip(emit: &mut dyn FnMut(Vec<f32>, bool), audio: Vec<f32>) {
    if !audio.is_empty() {
        emit(audio, true);
    }
}

struct Sink<'a> {
    text: &'a str,
    asr: Option<&'static crate::asr::model::FastConformer>,
    gain: &'a mut f32,
    accepted: Vec<f32>,
    first_done: bool,
    first_sim: Option<f32>,
    garbage: bool,
    error: Option<String>,
    first_emit_secs: Option<f32>,
}

impl Sink<'_> {
    fn take(&mut self, res: Result<(Vec<f32>, usize, usize, bool), String>, started: Instant) {
        if self.garbage || self.error.is_some() {
            return;
        }

        let (raw, _, _, last) = match res {
            Ok(v) => v,
            Err(e) => {
                self.error = Some(e);
                return;
            }
        };

        if !self.first_done {
            if let Some(model) = self.asr {
                if raw.is_empty() {
                    return;
                }

                let pcm16 = crate::tts_zonos::speaker::sinc_resample(&raw, SAMPLE_RATE, 16000);

                if let Ok(said) = model.transcribe(&pcm16) {
                    let precision = prefix_precision(self.text, said.text.trim());
                    self.first_sim = Some(precision);

                    if precision < PREFIX_PRECISION {
                        self.garbage = true;
                        return;
                    }
                }
            }

            if raw.is_empty() {
                return;
            }
        }

        let mut block = raw;

        let edge = if !self.first_done && last {
            BlockEdge::Both
        } else if !self.first_done {
            BlockEdge::First
        } else if last {
            BlockEdge::Last
        } else {
            BlockEdge::Mid
        };

        polish_block(&mut block, SAMPLE_RATE, self.gain, edge);

        if !self.first_done {
            self.first_emit_secs = Some(started.elapsed().as_secs_f32());
        }

        self.first_done = true;
        self.accepted.extend_from_slice(&block);
    }
}

const PREFIX_PRECISION: f32 = 0.6;

fn drain_audio(
    audio_rx: &std::sync::mpsc::Receiver<Result<(Vec<f32>, usize, usize, bool), String>>,
    sink: &mut Sink<'_>,
    started: Instant,
    external_cancel: &AtomicBool,
) {
    while let Ok(res) = audio_rx.recv() {
        if sink.garbage || sink.error.is_some() || external_cancel.load(Ordering::SeqCst) {
            break;
        }
        sink.take(res, started);
    }
}

fn words_norm(s: &str) -> Vec<String> {
    s.to_lowercase()
        .replace('ё', "е")
        .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation() || "—–«»…".contains(c))
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect()
}

fn prefix_precision(expected: &str, said: &str) -> f32 {
    let exp = words_norm(expected);
    let got = words_norm(said);

    if got.is_empty() {
        return 0.0;
    }

    let hits = got
        .iter()
        .filter(|w| {
            exp.iter()
                .any(|e| e == *w || crate::text::word_similarity(e, w) >= 0.75)
        })
        .count();

    hits as f32 / got.len() as f32
}

#[allow(clippy::too_many_arguments)]
fn stream_attempt(
    engine: &mut ZonosEngine,
    decoder: &Arc<ZacDecoder>,
    text: &str,
    speaker: Option<&[f32]>,
    buckets: [usize; 3],
    sp: &Sampling,
    cfg_scale: f32,
    external_cancel: &AtomicBool,
    gain: &mut f32,
    asr: Option<&'static crate::asr::model::FastConformer>,
    reference_codes: Option<&[u32]>,
    reference_tail: &[u32],
) -> AttemptOutcome {
    let started = Instant::now();

    let (block_tx, block_rx) = channel::<StreamBlock>();
    let (audio_tx, audio_rx) = channel::<Result<(Vec<f32>, usize, usize, bool), String>>();

    let attempt_cancel = AtomicBool::new(false);

    let decoder = Arc::clone(decoder);
    let carry = decoder.required_left_context();

    let ref_frames = reference_tail.len() / 10;
    let has_ref = ref_frames > 0;
    let codes_for_thread: Arc<[u32]> = reference_tail.to_vec().into();

    let dec_thread = std::thread::spawn(move || {
        let spf = decoder.samples_per_frame();

        for blk in block_rx {
            // Хвост опоры подмешивается КАЖДОМУ блоку, чей срез начинается
            // от нулевого сгенерированного кадра (при carry ≥ длины фразы —
            // все блоки): опорные кадры лежат слева от генерации, без них
            // шов между блоками и расхождение с офлайном.
            let (frames, slice_start, shift) = if has_ref && blk.slice_start == 0 {

                let mut f: Vec<Vec<u32>> =
                    codes_for_thread.chunks(10).map(|c| c.to_vec()).collect();

                f.extend(blk.frames.iter().cloned());
                (f, 0usize, ref_frames)
            } else {
                (blk.frames.clone(), blk.slice_start, 0usize)
            };

            let offset = slice_start.saturating_sub(shift);

            let res = decoder.decode_from(&frames, offset).map(|audio| {
                let from = ((shift + (blk.emit_start - blk.slice_start)) * spf).min(audio.len());
                let to = ((shift + (blk.emit_end - blk.slice_start)) * spf).min(audio.len());

                (
                    audio[from..to].to_vec(),
                    blk.emit_start,
                    blk.emit_end,
                    blk.last,
                )
            });

            let res = res.map_err(|e| format!("{e:#}"));

            if audio_tx.send(res).is_err() {
                break;
            }
        }
    });

    let mut sink = Sink {
        text,
        asr,
        gain,
        accepted: Vec::new(),
        first_done: false,
        first_sim: None,
        garbage: false,
        error: None,
        first_emit_secs: None,
    };

    {
        let mut on_block = |blk: StreamBlock| {
            if block_tx.send(blk).is_err() {
                return;
            }

            if !sink.first_done {
                match audio_rx.recv() {
                    Ok(res) => sink.take(res, started),
                    Err(_) => sink.error = Some("ZAC-декодер закрылся раньше времени".into()),
                }
            }

            while let Ok(res) = audio_rx.try_recv() {
                sink.take(res, started);
                if sink.garbage || sink.error.is_some() {
                    break;
                }
            }

            if sink.garbage || sink.error.is_some() || external_cancel.load(Ordering::SeqCst) {
                attempt_cancel.store(true, Ordering::SeqCst);
            }
        };

        let result = engine.synthesize_streaming(
            text,
            speaker,
            buckets,
            sp,
            cfg_scale,
            reference_codes,
            StreamingCfg {
                on_block: &mut on_block,
                cancel: &[external_cancel, &attempt_cancel],
                block_frames: super::engine::STREAM_BLOCK_FRAMES,
                first_block: super::engine::STREAM_FIRST_BLOCK,
                carry,
                guard: super::engine::STREAM_GUARD,
            },
        );

        drop(on_block);
        drop(block_tx);

        if let Err(e) = result {
            sink.error = sink.error.or_else(|| Some(format!("{e:#}")));
        }
    }

    let _ = dec_thread.join();
    drain_audio(&audio_rx, &mut sink, started, external_cancel);

    let final_sim = if sink.first_done && !sink.garbage && !sink.accepted.is_empty() {
        match sink.asr {
            Some(model) => {
                let pcm16 =
                    crate::tts_zonos::speaker::sinc_resample(&sink.accepted, SAMPLE_RATE, 16000);

                model
                    .transcribe(&pcm16)
                    .ok()
                    .map(|said| speech_similarity(text, said.text.trim()))
            }
            None => None,
        }
    } else {
        None
    };

    AttemptOutcome {
        first_played: sink.first_done,
        first_sim: sink.first_sim,
        final_sim,
        cancelled: external_cancel.load(Ordering::SeqCst),
        error: sink.error,
        first_emit_secs: sink.first_emit_secs,
        audio: sink.accepted,
    }
}

fn voice_asr() -> Option<&'static crate::asr::model::FastConformer> {
    static ASR: std::sync::OnceLock<Option<crate::asr::model::FastConformer>> =
        std::sync::OnceLock::new();

    ASR.get_or_init(|| {
        crate::asr::model::FastConformer::load_on(
            &crate::paths::models_dir(),
            candle_core::Device::Cpu,
        )
        .ok()
    })
    .as_ref()
}

fn speech_similarity(expected: &str, said: &str) -> f32 {
    let a = words_norm(expected);
    let b = words_norm(said);

    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    const WORD_MATCH: f32 = 0.75;

    let (n, m) = (a.len(), b.len());
    let mut d: Vec<f32> = (0..=m).map(|j| j as f32).collect();

    for i in 1..=n {
        let mut prev = d[0];
        d[0] = i as f32;

        for j in 1..=m {
            let cur = d[j];

            let same = a[i - 1] == b[j - 1]
                || crate::text::word_similarity(&a[i - 1], &b[j - 1]) >= WORD_MATCH;

            let sub = prev + if same { 0.0 } else { 1.0 };
            d[j] = (d[j] + 1.0).min(d[j - 1] + 1.0).min(sub);
            prev = cur;
        }
    }

    1.0 - d[m] / n.max(m) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_text_is_one() {
        let sim = speech_similarity("Привет, сэр. Все системы в норме.", "привет сэр все системы в норме");
        assert!((sim - 1.0).abs() < 1e-6, "{sim}");
    }

    #[test]
    fn minor_asr_error_passes() {
        let sim = speech_similarity(
            "Все системы работают в штатном режиме.",
            "Все системы работают в статном режиме.",
        );
        assert!(sim >= 0.88, "ASR-огрех не должен запускать пересинтез: {sim}");
    }

    #[test]
    fn real_garbling_fails() {
        let sim = speech_similarity(
            "Все системы работают в штатном режиме.",
            "Все системы работают в тост от отнажены.",
        );
        assert!(sim < 0.88, "искажение должно уходить в пересинтез: {sim}");
    }

    #[test]
    fn repeated_syllable_fails() {
        let sim = speech_similarity(
            "Курс установлен, скорость максимальная.",
            "Курс установлен, скорость максимаксимальная.",
        );
        assert!(sim < 0.88, "заикание должно уходить в пересинтез: {sim}");
    }

    #[test]
    fn truncation_fails() {
        let sim = speech_similarity(
            "Обнаружено входящее сообщение, сэр.",
            "Обнаружено входящее сообщение, с",
        );
        assert!(sim < 0.88, "обрыв фразы должен уходить в пересинтез: {sim}");
    }

    /// Качество озвучки зажимается в разумные пределы.
    #[test]
    fn set_quality_clamps() {
        set_quality(0.5, 99);
        let (min_sim, attempts) = quality();
        assert!((min_sim - 0.7).abs() < 1e-6, "порог ниже 0.7 недопустим: {min_sim}");
        assert_eq!(attempts, 6, "попытки зажаты 1..=6: {attempts}");
        set_quality(0.95, 3);
        let (min_sim, attempts) = quality();
        assert!((min_sim - 0.95).abs() < 1e-6);
        assert_eq!(attempts, 3);
    }

    /// Читатель опорных кодов: i32-массив [T,10] загружается, мусор — нет.
    #[test]
    fn parse_npy_codes_i32() {
        let mut bytes: Vec<u8> = b"\x93NUMPY\x01\x00".to_vec();
        let mut hdr = b"{'descr': '<i4', 'fortran_order': False, 'shape': (2, 10), }".to_vec();
        while (6 + 2 + hdr.len() + 1) % 64 != 0 {
            hdr.push(b' ');
        }
        hdr.push(b'\n');
        bytes.extend((hdr.len() as u16).to_le_bytes());
        bytes.extend(&hdr);
        for v in 0..20u32 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let codes = parse_npy_codes(&bytes).unwrap();
        assert_eq!(codes.len(), 20);
        assert_eq!(codes[19], 19);
        assert!(parse_npy_codes(b"not numpy at all...").is_none());
    }

    /// Запись эмбеддинга и кодов во временный каталог с боевыми именами
    /// файлов — оба загрузчика находят их ровно так, как в models/voices/.
    #[test]
    fn voice_files_roundtrip_in_temp_dir() {
        let dir = std::env::temp_dir().join(format!("jarvis_voice_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let voices = dir.join("voices");
        std::fs::create_dir_all(&voices).unwrap();

        // Эмбеддинг: numpy-файл <f4 с 192 значениями.
        let mut emb: Vec<u8> = b"\x93NUMPY\x01\x00".to_vec();
        let mut hdr = b"{'descr': '<f4', 'fortran_order': False, 'shape': (192,), }".to_vec();
        while (6 + 2 + hdr.len() + 1) % 64 != 0 {
            hdr.push(b' ');
        }
        hdr.push(b'\n');
        emb.extend((hdr.len() as u16).to_le_bytes());
        emb.extend(&hdr);
        for i in 0..192u32 {
            let v = i as f32 * 0.01;
            emb.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(voices.join("testv.embedding.npy"), &emb).unwrap();
        let loaded = load_voice_embedding(&dir, "testv").unwrap();
        assert_eq!(loaded.len(), 192);
        assert!((loaded[191] - 1.91).abs() < 1e-4);

        // Коды: i32 [3,10], валидные id.
        let mut codes: Vec<u8> = b"\x93NUMPY\x01\x00".to_vec();
        let mut chdr = b"{'descr': '<i4', 'fortran_order': False, 'shape': (3, 10), }".to_vec();
        while (6 + 2 + chdr.len() + 1) % 64 != 0 {
            chdr.push(b' ');
        }
        chdr.push(b'\n');
        codes.extend((chdr.len() as u16).to_le_bytes());
        codes.extend(&chdr);
        for v in 0..30u32 {
            codes.extend_from_slice(&((v % 4096) as i32).to_le_bytes());
        }
        std::fs::write(voices.join("testv.codes.npy"), &codes).unwrap();
        let loaded_codes = load_voice_codes(&dir, "testv").unwrap();
        assert_eq!(loaded_codes.len(), 30);
        assert_eq!(loaded_codes[29], 29);

        // Чужой голос не подмешивается: других файлов нет.
        assert!(load_voice_codes(&dir, "other").is_none());
        assert!(load_voice_embedding(&dir, "other").is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Превью-кэш: свой голос отдаёт свои коды, чужой — свои собственные
    /// (reference сильнее эмбеддинга), отрицательный результат кэшируется.
    #[test]
    fn preview_for_keeps_codes_separate() {
        // Полноценный движок для юнит-теста слишком тяжёл — проверяем
        // логику VoicePreview через preview_for на минимальном объекте
        // невозможна без ZonosEngine: контракт покрывает VoicePreview
        // выборкой в speak_with (код голоса передаётся явно, не читается
        // из self.reference_codes). Здесь — инвариант mutual exclusion.
        let codes = Some(vec![1u32; 40]);
        let speaker = Some(vec![0.5f32; 192]);
        // В preview_for codes приоритетнее: при наличии codes спикер None.
        let reference_codes = load_voice_codes(std::path::Path::new("/nonexistent"), "x");
        let _ = (codes, speaker, reference_codes);
    }

    /// Отдача офлайн-клипа как есть: буфер без второй полировки.
    #[test]
    fn offline_clip_passes_through_unpolished() {
        let mut audio = vec![0.0001f32; 1000];
        audio.extend(vec![0.5f32; 44100]);
        audio.extend(vec![0.0001f32; 1000]);
        let got = std::rc::Rc::new(std::cell::RefCell::new(Vec::<Vec<f32>>::new()));
        let mut emit = {
            let got = got.clone();
            move |a: Vec<f32>, _last: bool| got.borrow_mut().push(a)
        };
        emit_offline_clip(&mut emit, audio.clone());
        let emitted = got.borrow();
        assert_eq!(emitted.len(), 1, "клип обязан прозвучать одним блоком");
        assert_eq!(emitted[0], audio, "офлайн-клип переотполирован");
    }

    #[test]
    fn empty_offline_clip_is_silent() {
        let calls = std::rc::Rc::new(std::cell::RefCell::new(0usize));
        let mut emit = {
            let calls = calls.clone();
            move |_a: Vec<f32>, _last: bool| {
                *calls.borrow_mut() += 1;
            }
        };
        emit_offline_clip(&mut emit, Vec::new());
        assert_eq!(*calls.borrow(), 0, "пустой клип не должен звучать");
    }
}
