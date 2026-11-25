//! ZAC-декодер: кадры кодов → звук 44.1 кГц.
//!
//! Порт ggml-реализации `servers/ggml/src/zac.cpp` из Zyphra/ZONOS3-Mini
//! (архитектура `mossdual_istft`, «ZONOS3 Mini MR-iSTFT decoder») на candle:
//!
//! 1. RVQ: код книги 0 — семантика (`sem.table`), коды 1..9 — остаток
//!    (`res.{i}.table`); таблицы уже свёрнуты с out_proj конвертером,
//!    поверх прибавляется `res.bias` (сумма смещений всех остаточных книг);
//! 2. семантическая голова: причинные ConvNeXt-блоки и линейный выход в
//!    латент 1024;
//! 3. пять MOSS-стадий: линейный вход, слои каузального внимания с RoPE и
//!    окном (`stage.context` кадров), линейный выход, patch-up ×2;
//! 4. iSTFT-голова: ConvNeXt-ствол, LayerNorm, две проекции в спектры
//!    (n_fft 736 и 2208, hop 184), обратное STFT с окном Ханна, нормировка
//!    на энергию окна и FIR-микшер 2→1;
//! 5. точный sinc-ресемплер 44160 → 44100 (ядро лежит в GGUF),
//!    обрезка до `кадры × 2940` и ослабление на headroom дБ.
//!
//! Активации каналы-мажор: `[каналы, время]`, как в ggml. Все матрицы —
//! f32 (в GGUF они F16, деквантизация на загрузке).

use super::gguf::GgufFile;
use anyhow::{Context, Result, bail};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use rustfft::{FftPlanner, FftDirection};
use std::path::Path;

/// Конфигурация декодера из метаданных GGUF.
#[derive(Debug, Clone)]
pub struct ZacConfig {
    pub sample_rate: u32,
    pub internal_rate: u32,
    pub frame_rate: u32,
    /// Внешних сэмплов на кадр (2940).
    pub hop: u32,
    /// Внутренних сэмплов на кадр (2944).
    pub internal_hop: u32,
    pub n_codebooks: usize,
    pub codebook_size: usize,
    pub latent_dim: usize,
    pub semantic_dim: usize,
    pub sem_depth: usize,
    pub ln_eps: f32,
    pub convnext_eps: f32,
    /// Стадии: вход, выход, модель, головы, слои, FFN, окно внимания, patch.
    pub stages: Vec<StageHp>,
    pub head_depth: usize,
    pub head_hidden: usize,
    pub head_hop: usize,
    pub fir_kernel: usize,
    pub n_fft: [usize; 2],
    pub max_log_mag: f32,
    pub headroom_db: f32,
    pub rs_orig: usize,
    pub rs_new: usize,
    pub rs_width: usize,
}

/// Параметры одной MOSS-стадии.
#[derive(Debug, Clone)]
pub struct StageHp {
    pub in_dim: usize,
    pub out_dim: usize,
    pub d_model: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_ff: usize,
    /// Окно каузального внимания в кадрах стадии.
    pub context: usize,
    pub patch: usize,
}

/// ConvNeXt-блок (семантическая голова и iSTFT-ствол).
struct ConvNext {
    /// Глубинные отводы [C, K].
    dw: Tensor,
    dw_bias: Tensor,
    norm_w: Tensor,
    norm_b: Tensor,
    /// [hidden, C].
    pw1: Tensor,
    pw1_bias: Tensor,
    /// [C, hidden].
    pw2: Tensor,
    pw2_bias: Tensor,
}

/// Слой MOSS-трансформера.
struct MossLayerW {
    norm1_w: Tensor,
    norm1_b: Tensor,
    /// [3d, d].
    qkv: Tensor,
    /// [d, d].
    wo: Tensor,
    attn_scale: Tensor,
    norm2_w: Tensor,
    norm2_b: Tensor,
    /// [n_ff, d].
    ffn1: Tensor,
    /// [d, n_ff].
    ffn2: Tensor,
    ffn_scale: Tensor,
}

/// Одна MOSS-стадия.
pub struct Stage {
    pub hp: StageHp,
    /// [d_model, in_dim].
    input: Tensor,
    /// [out_dim, d_model], если out_dim > 0.
    output: Option<Tensor>,
    pub layers: Vec<MossLayerW>,
}

/// Декодер ZAC: вес и конфигурация, готовые к проходу по кадрам.
pub struct ZacDecoder {
    pub cfg: ZacConfig,
    device: Device,
    sem_table: Tensor,
    sem_bias: Tensor,
    res_tables: Vec<Tensor>,
    res_bias: Tensor,
    sem_head: Vec<ConvNext>,
    sem_out: (Tensor, Tensor),
    pub stages: Vec<Stage>,
    head: Vec<ConvNext>,
    head_norm: (Tensor, Tensor),
    head_out: Vec<(Tensor, Tensor)>,
    /// FIR-микшер: [2 × K] отводов (сначала ветка 0).
    mixer: Vec<f32>,
    windows: [Vec<f32>; 2],
    /// Ресемплер 44160 → 44100: [rs_new × KL].
    rs_kernel: Vec<f32>,
}

const ARCH: &str = "mossdual_istft";
const ROPE_THETA: f32 = 10000.0;
/// Граница блока запросов в внимании (как в эталоне: 512-кадровые блоки).
const ATTN_BLOCK: usize = 512;

impl ZacDecoder {
    /// Открыть `zac.gguf` и поднять все веса.
    pub fn open(path: &Path, device: &Device) -> Result<Self> {
        let file = GgufFile::open(path)?;
        let kv = &file.kv;

        let arch = file.kv_str("general.architecture").unwrap_or_default();
        if arch != ARCH {
            bail!("Ожидалась архитектура {ARCH}, получено «{arch}»");
        }
        let u32v = |key: &str| -> Result<u32> {
            kv.get(&format!("{ARCH}.{key}"))
                .and_then(|v| v.as_i64())
                .map(|v| v as u32)
                .with_context(|| format!("Нет {ARCH}.{key}"))
        };
        let f32v = |key: &str| -> Result<f32> {
            kv.get(&format!("{ARCH}.{key}"))
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .with_context(|| format!("Нет {ARCH}.{key}"))
        };
        let arr = |key: &str, n: usize| -> Result<Vec<usize>> {
            let v = kv
                .get(&format!("{ARCH}.{key}"))
                .and_then(|v| v.as_array())
                .with_context(|| format!("Нет {ARCH}.{key}"))?;
            if v.len() != n {
                bail!("{ARCH}.{key}: {} значений вместо {n}", v.len());
            }
            Ok(v.iter().map(|v| v.as_i64().unwrap_or(0).max(0) as usize).collect())
        };

        let sem_depth = u32v("sem_head.depth")? as usize;
        let head_depth = u32v("head.depth")? as usize;
        let n_stages = 5;
        let cfg = ZacConfig {
            sample_rate: u32v("sample_rate")?,
            internal_rate: u32v("internal_sample_rate")?,
            frame_rate: u32v("frame_rate")?,
            hop: u32v("hop")?,
            internal_hop: u32v("internal_hop")?,
            n_codebooks: u32v("n_codebooks")? as usize,
            codebook_size: u32v("codebook_size")? as usize,
            latent_dim: u32v("latent_dim")? as usize,
            semantic_dim: u32v("semantic_dim")? as usize,
            sem_depth,
            ln_eps: f32v("ln_eps")?,
            convnext_eps: f32v("convnext_eps")?,
            stages: {
                let in_dim = arr("stage.in_dim", n_stages)?;
                let out_dim = arr("stage.out_dim", n_stages)?;
                let d_model = arr("stage.d_model", n_stages)?;
                let n_head = arr("stage.n_head", n_stages)?;
                let n_layer = arr("stage.n_layer", n_stages)?;
                let n_ff = arr("stage.n_ff", n_stages)?;
                let context = arr("stage.context", n_stages)?;
                let patch = arr("stage.patch", n_stages)?;
                (0..n_stages)
                    .map(|s| StageHp {
                        in_dim: in_dim[s],
                        out_dim: out_dim[s],
                        d_model: d_model[s],
                        n_head: n_head[s],
                        n_layer: n_layer[s],
                        n_ff: n_ff[s],
                        context: context[s],
                        patch: patch[s],
                    })
                    .collect()
            },
            head_depth,
            head_hidden: u32v("head.hidden")? as usize,
            head_hop: u32v("head.hop")? as usize,
            fir_kernel: u32v("head.fir_kernel")? as usize,
            n_fft: {
                let v = arr("head.n_fft", 2)?;
                [v[0], v[1]]
            },
            max_log_mag: f32v("head.max_log_magnitude")?,
            headroom_db: f32v("headroom_db")?,
            rs_orig: u32v("resample.orig")? as usize,
            rs_new: u32v("resample.new")? as usize,
            rs_width: u32v("resample.width")? as usize,
        };
        let c = &cfg;
        // Согласованность размеров — как в zac.cpp.
        if c.sample_rate != 44100
            || c.internal_rate != 44160
            || c.frame_rate != 15
            || c.hop != 2940
            || c.internal_hop != 2944
            || c.n_codebooks != 10
            || c.codebook_size != 4096
        {
            bail!("Неожиданные частоты/кадры в {path:?}");
        }
        let mut up = c.head_hop;
        for hp in &c.stages {
            up *= hp.patch;
        }
        if up != c.internal_hop as usize || 16 * c.head_hop != c.internal_hop as usize {
            bail!("Patch-лестница не сходится с internal_hop");
        }
        for (s, hp) in c.stages.iter().enumerate() {
            if hp.d_model % hp.n_head != 0 || (hp.d_model / hp.n_head) % 2 != 0 {
                bail!("Стадия {s}: d_model {} не делится на головы", hp.d_model);
            }
            // Guard: окно внимания нулевой длины невалидно — moss_layer
            // считает context − 1, ноль обернулся бы underflow-паникой.
            if hp.context == 0 {
                bail!("Стадия {s}: context 0 — окно внимания пустое");
            }
            if s + 1 < c.stages.len() && c.stages[s + 1].in_dim * hp.patch != hp.out_dim {
                bail!("Стадия {s}: выход и вход следующей не стыкуются");
            }
        }

        let t32 = |name: &str| -> Result<Tensor> {
            file.tensor_f32(name, device)
                .with_context(|| format!("Нет тензора «{name}»"))
        };

        let sem_table = t32("sem.table")?;
        let sem_bias = t32("sem.bias")?;
        let mut res_tables = Vec::with_capacity(c.n_codebooks - 1);
        for i in 0..c.n_codebooks - 1 {
            res_tables.push(t32(&format!("res.{i}.table"))?);
        }
        let res_bias = t32("res.bias")?;

        let sem_head = (0..c.sem_depth)
            .map(|b| load_convnext(&file, &format!("sem_head.{b}"), device))
            .collect::<Result<Vec<_>>>()?;
        let sem_out = (t32("sem_head.out.weight")?, t32("sem_head.out.bias")?);

        let mut stages = Vec::with_capacity(n_stages);
        for (s, hp) in c.stages.iter().enumerate() {
            let input = t32(&format!("stage.{s}.in.weight"))?;
            let output = if hp.out_dim > 0 {
                Some(t32(&format!("stage.{s}.out.weight"))?)
            } else {
                None
            };
            let mut layers = Vec::with_capacity(hp.n_layer);
            for l in 0..hp.n_layer {
                let p = format!("stage.{s}.l{l}");
                layers.push(MossLayerW {
                    norm1_w: t32(&format!("{p}.norm1.weight"))?,
                    norm1_b: t32(&format!("{p}.norm1.bias"))?,
                    qkv: t32(&format!("{p}.qkv.weight"))?,
                    wo: t32(&format!("{p}.wo.weight"))?,
                    attn_scale: t32(&format!("{p}.attn_scale"))?,
                    norm2_w: t32(&format!("{p}.norm2.weight"))?,
                    norm2_b: t32(&format!("{p}.norm2.bias"))?,
                    ffn1: t32(&format!("{p}.ffn1.weight"))?,
                    ffn2: t32(&format!("{p}.ffn2.weight"))?,
                    ffn_scale: t32(&format!("{p}.ffn_scale"))?,
                });
            }
            stages.push(Stage {
                hp: hp.clone(),
                input,
                output,
                layers,
            });
        }

        let head = (0..c.head_depth)
            .map(|b| load_convnext(&file, &format!("head.{b}"), device))
            .collect::<Result<Vec<_>>>()?;
        let head_norm = (t32("head.norm.weight")?, t32("head.norm.bias")?);
        let head_out = (0..2)
            .map(|i| {
                Ok((
                    t32(&format!("head.out.{i}.weight"))?,
                    t32(&format!("head.out.{i}.bias"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        // Хостовые тензоры: микшер, окна, ядро ресемплера.
        let mixer = flat(&file.tensor_f32("head.mixer.weight", &Device::Cpu)?)?;
        let mut windows = [Vec::new(), Vec::new()];
        for i in 0..2 {
            windows[i] = flat(&file.tensor_f32(&format!("head.window.{i}"), &Device::Cpu)?)?;
        }
        let rs_kernel = flat(&file.tensor_f32("resample.kernel", &Device::Cpu)?)?;
        if rs_kernel.len() != c.rs_new * (2 * c.rs_width + c.rs_orig) {
            bail!("Ядро ресемплера неожиданного размера");
        }
        if mixer.len() != 2 * c.fir_kernel {
            bail!("Микшер неожиданного размера");
        }

        Ok(Self {
            cfg: cfg.clone(),
            device: device.clone(),
            sem_table,
            sem_bias,
            res_tables,
            res_bias,
            sem_head,
            sem_out,
            stages,
            head,
            head_norm,
            head_out,
            mixer,
            windows,
            rs_kernel,
        })
    }

    /// Минимальный левый контекст кусочного декода (кадров кодов), при
    /// котором склейка блоков совпадает с цельным декодом. Сумма окон стадий,
    /// пересчитанных в кадры кодов (стадия s оперирует на rate_s = ∏ patch
    /// предыдущих стадий), плюс причинная ConvNeXt-семантика и ствол iSTFT.
    /// Референс zac.cpp: контексты стадий — из GGUF (stage.context),
    /// ConvNeXt — причинная свёртка 7×7 (sem_head и head).
    ///
    /// Формула даёт ТОЧНУЮ границу достижения краевого загрязнения среза
    /// (замер на zac-keep2: 535.06 кадра — тест chunked_decode_matches_full
    /// и покомпонентный seam_component_dump). Хвост загрязнения затухает
    /// через окна внимания медленно: на самой границе расхождение ~1.3e-4,
    /// поэтому сверху добавлен запас SEAM_MARGIN (замер: +32…+64 кадра
    /// дают 3–6e-5; порог не подгонялся).
    pub fn required_left_context(&self) -> usize {
        /// Запас сверх точной границы краевого загрязнения, кадров кодов.
        const SEAM_MARGIN: usize = 64;
        let c = &self.cfg;
        // Ядро глубинной свёртки ConvNeXt — из формы весов, не константа.
        let k_dw = self
            .head
            .first()
            .map(|b| b.dw.dim(1).unwrap_or(7))
            .unwrap_or(7);
        let mut frames = 0.0f64;
        let mut rate = 1.0f64;
        for hp in &c.stages {
            frames += (hp.context as f64 - 1.0) / rate;
            rate *= hp.patch as f64;
        }
        // Семантика работает на частоте кадров кодов: (k_dw − 1) кадров.
        frames += c.sem_depth as f64 * (k_dw - 1) as f64;
        // Ствол iSTFT — на 16 кадров головы на кадр кодов.
        frames += c.head_depth as f64 * (k_dw - 1) as f64 / 16.0;
        frames.ceil() as usize + SEAM_MARGIN
    }

    /// Спектры головы для диагностики: [logmag(bins), phase(bins)] на кадр.
    pub fn debug_spectra(&self, frames: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        let c = &self.cfg;
        let dev = &self.device;
        let nf = frames.len();
        let idx = |cb: usize| -> Result<Tensor> {
            let ids: Vec<u32> = frames.iter().map(|f| f[cb].min(c.codebook_size as u32 - 1)).collect();
            Ok(Tensor::from_vec(ids, &[nf], dev)?)
        };
        let sem = self.sem_table.index_select(&idx(0)?, 0)?.t()?.contiguous()?.broadcast_add(&self.sem_bias.unsqueeze(1)?)?;
        let mut x = convnext_chain(&sem, &self.sem_head, c.convnext_eps)?;
        x = self.sem_out.0.matmul(&x)?.broadcast_add(&self.sem_out.1.unsqueeze(1)?)?;
        for (cb, table) in self.res_tables.iter().enumerate() {
            let part = table.index_select(&idx(cb + 1)?, 0)?.t()?.contiguous()?;
            x = x.broadcast_add(&part)?;
        }
        x = x.broadcast_add(&self.res_bias.unsqueeze(1)?)?;
        for stage in &self.stages {
            let hp = &stage.hp;
            x = stage.input.matmul(&x)?;
            for layer in &stage.layers {
                x = moss_layer(x, layer, hp, c.ln_eps, 0, &self.device)?;
            }
            if let Some(out) = &stage.output {
                x = out.matmul(&x)?;
            }
            if hp.patch > 1 {
                x = patch_up(&x, hp.patch)?;
            }
        }
        for block in &self.head {
            x = convnext(&x, block, c.convnext_eps)?;
        }
        x = layer_norm_channels(&x, &self.head_norm.0, &self.head_norm.1, c.convnext_eps)?;
        let mut out = Vec::new();
        for (w, b) in &self.head_out {
            out.push(flat(&w.matmul(&x)?.broadcast_add(&b.unsqueeze(1)?)?.t()?.contiguous()?)?);
        }
        Ok(out)
    }

    /// Постатейная статистика пайплайна — диагностика (cargo run --example probe_stage_stats).
    pub fn dump_pipeline(&self, frames: &[Vec<u32>]) -> Result<()> {
        println!("KV eps: ln={:?} convnext={:?}", self.cfg.ln_eps, self.cfg.convnext_eps);
        println!("KV: head.depth={} head.hidden={} sem_head.depth={}", self.cfg.head_depth, self.cfg.head_hidden, self.cfg.sem_depth);
        println!(
            "стадии: {:?}",
            self.stages.iter().map(|s| (s.hp.in_dim, s.hp.out_dim, s.hp.d_model, s.hp.n_head, s.hp.n_layer, s.hp.context, s.hp.patch)).collect::<Vec<_>>()
        );
        let c = &self.cfg;
        let dev = &self.device;
        let nf = frames.len();
        let idx = |cb: usize| -> Result<Tensor> {
            let ids: Vec<u32> = frames.iter().map(|f| f[cb].min(c.codebook_size as u32 - 1)).collect();
            Ok(Tensor::from_vec(ids, &[nf], dev)?)
        };
        let st = |t: &Tensor, label: &str| -> Result<()> {
            let v = t.flatten_all()?.to_vec1::<f32>()?;
            let n = v.len();
            let mean = v.iter().sum::<f32>() / n as f32;
            let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
            let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
            for &x in &v {
                mn = mn.min(x);
                mx = mx.max(x);
            }
            // Временная корреляция соседних кадров (структура сигнала).
            let dims = t.shape().dims();
            let mut corr = 0.0f32;
            if dims.len() == 2 && dims[1] > 1 {
                let cols: Vec<Vec<f32>> = (0..dims[1])
                    .map(|ti| t.i((.., ti)).map(|x| x.to_vec1::<f32>().unwrap_or_default()).unwrap_or_default())
                    .collect();
                let (mut num, mut d1, mut d2) = (0.0f64, 0.0f64, 0.0f64);
                for w in cols.windows(2) {
                    let (a, b) = (&w[0], &w[1]);
                    num += a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum::<f64>();
                    d1 += a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
                    d2 += b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
                }
                corr = (num / (d1.sqrt() * d2.sqrt() + 1e-12)) as f32;
            }
            println!("{label:<30} shape={:?} mean={mean:+.4} std={:.4} min={mn:+.3} max={mx:+.3} corr={corr:.3}", t.shape().dims(), var.sqrt());
            Ok(())
        };
        let sem = self.sem_table.index_select(&idx(0)?, 0)?.t()?.contiguous()?.broadcast_add(&self.sem_bias.unsqueeze(1)?)?;
        st(&self.sem_table, "sem.table")?;
        st(&sem, "book0 → sem [C,T]")?;
        let mut x = convnext_chain(&sem, &self.sem_head, c.convnext_eps)?;
        st(&x, "sem_head ConvNeXt")?;
        x = self.sem_out.0.matmul(&x)?.broadcast_add(&self.sem_out.1.unsqueeze(1)?)?;
        st(&x, "sem_out → латент")?;
        for (cb, table) in self.res_tables.iter().enumerate() {
            let part = table.index_select(&idx(cb + 1)?, 0)?.t()?.contiguous()?;
            x = x.broadcast_add(&part)?;
        }
        x = x.broadcast_add(&self.res_bias.unsqueeze(1)?)?;
        st(&x, "+ остаточные книги (z)")?;
        for (si, stage) in self.stages.iter().enumerate() {
            let hp = &stage.hp;
            x = stage.input.matmul(&x)?;
            st(&x, &format!("stage{si}.in"))?;
            for (li, layer) in stage.layers.iter().enumerate() {
                x = moss_layer(x, layer, hp, c.ln_eps, 0, &self.device)?;
                if li == 0 || li + 1 == stage.layers.len() {
                    st(&x, &format!("stage{si}.l{li}"))?;
                }
            }
            if let Some(out) = &stage.output {
                x = out.matmul(&x)?;
            }
            if hp.patch > 1 {
                x = patch_up(&x, hp.patch)?;
            }
            st(&x, &format!("stage{si} → (patch {})", hp.patch))?;
        }
        for (bi, block) in self.head.iter().enumerate() {
            x = convnext(&x, block, c.convnext_eps)?;
            if bi < 2 || bi + 1 == self.head.len() {
                st(&x, &format!("head.{bi}"))?;
            }
        }
        x = layer_norm_channels(&x, &self.head_norm.0, &self.head_norm.1, c.convnext_eps)?;
        st(&x, "head.norm")?;
        let mut specs = Vec::new();
        for (i, (w, b)) in self.head_out.iter().enumerate() {
            let y = flat(&w.matmul(&x)?.broadcast_add(&b.unsqueeze(1)?)?.t()?.contiguous()?)?;
            // logmag и фазы отдельно.
            let bins = c.n_fft[i] / 2 + 1;
            let lm: Vec<f32> = y.chunks_exact(c.n_fft[i] + 2).flat_map(|r| &r[..bins]).copied().collect();
            let ph: Vec<f32> = y.chunks_exact(c.n_fft[i] + 2).flat_map(|r| &r[bins..]).copied().collect();
            let (lm_m, lm_s) = (lm.iter().sum::<f32>() / lm.len() as f32, {
                let m = lm.iter().sum::<f32>() / lm.len() as f32;
                (lm.iter().map(|x| (x - m).powi(2)).sum::<f32>() / lm.len() as f32).sqrt()
            });
            // logmag в нижней и верхней части спектра.
            let low: Vec<f32> = y.chunks_exact(c.n_fft[i] + 2).flat_map(|r| &r[..bins / 6]).copied().collect();
            let high: Vec<f32> = y.chunks_exact(c.n_fft[i] + 2).flat_map(|r| &r[bins * 3 / 4..bins]).copied().collect();
            let avg = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
            println!("head.out.{i}: logmag mean={lm_m:+.3} std={lm_s:.3} низ={:+.3} верх={:+.3} | фаза mean={:+.3} std={:.3}",
                avg(&low), avg(&high), ph.iter().sum::<f32>() / ph.len() as f32,
                (ph.iter().map(|x| x.powi(2)).sum::<f32>() / ph.len() as f32).sqrt());
            specs.push(y);
        }
        let internal = self.istft_mix(&specs, nf * 16)?;
        st(&Tensor::from_vec(internal.clone(), &[internal.len()], dev)?, "istft+mixer")?;
        let audio = self.resample(&internal)?;
        st(&Tensor::from_vec(audio.clone(), &[audio.len()], dev)?, "resample → 44100")?;
        Ok(())
    }

    /// Декодировать кадры (де-sheared): `frames[f]` — 10 кодов кадра f.
    /// Возвращает моно-сэмплы 44.1 кГц.
    pub fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        self.decode_from(frames, 0)
    }

    /// Как [`decode`], но срез `frames` начинается с абсолютного кадра
    /// `offset` потока: RoPE стадий считает углы от абсолютных позиций
    /// (`pos0 = offset · rate_s`), поэтому кусочный декод побитово совпадает
    /// с цельным — швы блоков сходятся (см. тест chunked_decode_matches_full).
    pub fn decode_from(&self, frames: &[Vec<u32>], offset: usize) -> Result<Vec<f32>> {
        self.decode_books_from(frames, self.cfg.n_codebooks, offset)
    }

    /// Декодирование с усечением до `books` первых кодовых книг
    /// (как decode_tokens в эталоне: 1..n_codebooks — для диагностики).
    pub fn decode_books(&self, frames: &[Vec<u32>], books: usize) -> Result<Vec<f32>> {
        self.decode_books_from(frames, books, 0)
    }

    fn decode_books_from(
        &self,
        frames: &[Vec<u32>],
        books: usize,
        offset: usize,
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let dev = &self.device;
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let nf = frames.len();
        let books = books.clamp(1, c.n_codebooks);
        // Накопленный patch-фактор стадии: позиция стадии s = offset · rate_s.
        let mut rate = 1usize;

        // --- 1. RVQ → латент [latent, nf] -----------------------------------
        let idx = |cb: usize| -> Result<Tensor> {
            let ids: Vec<u32> = frames
                .iter()
                .map(|f| f[cb].min(c.codebook_size as u32 - 1))
                .collect();
            Ok(Tensor::from_vec(ids, &[nf], dev)?)
        };
        let sem = self
            .sem_table
            .index_select(&idx(0)?, 0)?
            .t()?
            .contiguous()?
            .broadcast_add(&self.sem_bias.unsqueeze(1)?)?; // [sem_dim, nf]
        let mut x = convnext_chain(&sem, &self.sem_head, c.convnext_eps)?;
        x = self
            .sem_out
            .0
            .matmul(&x)?
            .broadcast_add(&self.sem_out.1.unsqueeze(1)?)?; // [latent, nf]
        if books > 1 {
            for cb in 0..books - 1 {
                let part = self.res_tables[cb]
                    .index_select(&idx(cb + 1)?, 0)?
                    .t()?
                    .contiguous()?;
                x = x.broadcast_add(&part)?;
            }
            x = x.broadcast_add(&self.res_bias.unsqueeze(1)?)?; // [latent, nf]
        }

        // --- 2. MOSS-стадии ---------------------------------------------------
        for stage in &self.stages {
            let hp = &stage.hp;
            // Абсолютная позиция среза в единицах стадии: RoPE углы обязаны
            // совпадать с цельным декодом, иначе швы кусочного декода расходятся.
            let pos0 = offset * rate;
            rate *= hp.patch;
            x = stage.input.matmul(&x)?; // [d_model, nf]
            for layer in &stage.layers {
                x = moss_layer(x, layer, hp, c.ln_eps, pos0, &self.device)?;
            }
            if let Some(out) = &stage.output {
                x = out.matmul(&x)?;
            }
            if hp.patch > 1 {
                x = patch_up(&x, hp.patch)?;
            }
        }

        // --- 3. iSTFT-голова ---------------------------------------------------
        for block in &self.head {
            x = convnext(&x, block, c.convnext_eps)?;
        }
        x = layer_norm_channels(&x, &self.head_norm.0, &self.head_norm.1, c.convnext_eps)?;
        // head_out[i].0 [n_fft+2, head_frames]
        let head_frames = x.dim(1)?;
        let mut specs = Vec::with_capacity(2);
        for (w, b) in &self.head_out {
            specs.push(flat(&w.matmul(&x)?.broadcast_add(&b.unsqueeze(1)?)?.t()?.contiguous()?)?);
        }

        // --- 4. Хост: iSTFT, микшер, ресемплер ---------------------------------
        let internal = self.istft_mix(&specs, head_frames)?;
        let audio = self.resample(&internal)?;
        // Обрезка до ожидаемой длины и ослабление на headroom.
        let expected = nf * c.hop as usize;
        let gain = 10f32.powf(-c.headroom_db / 20.0);
        Ok(audio
            .into_iter()
            .take(expected)
            .map(|v| v * gain)
            .collect())
    }

    /// Две ветки iSTFT → FIR-микшер → сигнал 44160 Гц.
    fn istft_mix(&self, specs: &[Vec<f32>], head_frames: usize) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let hop = c.head_hop;
        let out_len = head_frames * hop;
        let mut branches = Vec::with_capacity(2);
        for (i, spec) in specs.iter().enumerate() {
            branches.push(self.istft_branch(i, spec, head_frames, out_len)?);
        }

        // Микшер: y[m] = Σ_c Σ_k w[c,k]·s_c[m + k - h], нули за краями сигнала.
        let h = (c.fir_kernel - 1) / 2;
        let mut out = vec![0.0f32; out_len];
        for (m, y) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (ci, s) in branches.iter().enumerate() {
                let w = &self.mixer[ci * c.fir_kernel..(ci + 1) * c.fir_kernel];
                for (k, wk) in w.iter().enumerate() {
                    // m + k - h может уйти ниже нуля: считаем в i64.
                    let idx = m as i64 + k as i64 - h as i64;
                    if idx >= 0 {
                        let idx = idx as usize;
                        if idx < out_len {
                            acc += wk * s[idx];
                        }
                    }
                }
            }
            *y = acc;
        }
        Ok(out)
    }

    /// Одна ветка: спектры [n_fft+2 на кадр] → сэмплы 44160 Гц (OLA по окну).
    fn istft_branch(
        &self,
        branch: usize,
        spec: &[f32],
        head_frames: usize,
        out_len: usize,
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let n = c.n_fft[branch];
        let bins = n / 2 + 1;
        let hop = c.head_hop;
        let win = &self.windows[branch];
        if win.len() != n {
            bail!("Окно ветки {branch}: {} значений вместо {n}", win.len());
        }

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft(n, FftDirection::Inverse);
        let mut buf: Vec<rustfft::num_complex::Complex<f32>> =
            vec![rustfft::num_complex::Complex::new(0.0, 0.0); n];
        let mut time = vec![0.0f32; n];

        // ola/env — добавка кадра t в позиции [t·hop, t·hop + n).
        let ola_len = (head_frames - 1) * hop + n;
        let mut ola = vec![0.0f32; ola_len];
        let mut env = vec![0.0f32; ola_len];

        for t in 0..head_frames {
            let f = &spec[t * (n + 2)..(t + 1) * (n + 2)];
            // logmag[0..bins], phase[bins..2*bins] → полный спектр с сопряжённой
            // симметрией (в GGUF спектр лежит в halfcomplex-раскладке pocketfft).
            buf[0] = rustfft::num_complex::Complex::new(
                f[0].min(c.max_log_mag).exp() * f[bins].cos(),
                0.0,
            );
            for k in 1..n / 2 {
                let a = f[k].min(c.max_log_mag).exp();
                let re = a * f[bins + k].cos();
                let im = a * f[bins + k].sin();
                buf[k] = rustfft::num_complex::Complex::new(re, im);
                buf[n - k] = rustfft::num_complex::Complex::new(re, -im);
            }
            buf[n / 2] = rustfft::num_complex::Complex::new(
                f[n / 2].min(c.max_log_mag).exp() * f[bins + n / 2].cos(),
                0.0,
            );
            fft.process(&mut buf);
            // Обратное БПФ rustfft не нормирует — делим на n.
            for (j, v) in buf.iter().enumerate() {
                time[j] = v.re / n as f32;
            }
            let base = t * hop;
            for j in 0..n {
                ola[base + j] += win[j] * time[j];
                env[base + j] += win[j] * win[j];
            }
        }

        // center=True: позиция m сигнала — паддированная m + n/2.
        let pad = n / 2;
        Ok((0..out_len)
            .map(|m| {
                let j = m + pad;
                let e = env[j];
                if e > 0.0 {
                    ola[j] / e
                } else {
                    0.0
                }
            })
            .collect())
    }

    /// Точный sinc-ресемплер 44160 → 44100 (блоки: 736 входных → 735 выходных).
    fn resample(&self, internal: &[f32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let o = c.rs_orig;
        let n = c.rs_new;
        let w = c.rs_width;
        let kl = 2 * w + o;
        let rs_in = internal.len() as i64;
        let total = ((n as i64 * rs_in) + o as i64 - 1) / o as i64;
        let mut audio = Vec::with_capacity(total as usize);
        let mut block = 0i64;
        loop {
            if block * n as i64 >= total {
                break;
            }
            let x0 = block * o as i64 - w as i64;
            for i in 0..n {
                let kernel = &self.rs_kernel[i * kl..(i + 1) * kl];
                let mut acc = 0.0f32;
                for (k, ki) in kernel.iter().enumerate() {
                    let j = x0 + k as i64;
                    if j >= 0 && j < rs_in {
                        acc += ki * internal[j as usize];
                    }
                }
                audio.push(acc);
            }
            block += 1;
        }
        Ok(audio)
    }
}

/// Свести тензор к плоскому Vec<f32>.
fn flat(t: &Tensor) -> Result<Vec<f32>> {
    let n = t.elem_count();
    Ok(t.reshape(&[n])?.to_vec1::<f32>()?)
}

/// Загрузить ConvNeXt-блок из GGUF.
fn load_convnext(file: &GgufFile, p: &str, device: &Device) -> Result<ConvNext> {
    let t32 = |name: &str| -> Result<Tensor> {
        file.tensor_f32(name, device)
            .with_context(|| format!("Нет тензора «{name}»"))
    };
    Ok(ConvNext {
        dw: t32(&format!("{p}.dw.weight"))?,
        dw_bias: t32(&format!("{p}.dw.bias"))?,
        norm_w: t32(&format!("{p}.norm.weight"))?,
        norm_b: t32(&format!("{p}.norm.bias"))?,
        pw1: t32(&format!("{p}.pw1.weight"))?,
        pw1_bias: t32(&format!("{p}.pw1.bias"))?,
        pw2: t32(&format!("{p}.pw2.weight"))?,
        pw2_bias: t32(&format!("{p}.pw2.bias"))?,
    })
}

/// Цепочка ConvNeXt-блоков.
fn convnext_chain(x: &Tensor, blocks: &[ConvNext], eps: f32) -> Result<Tensor> {
    let mut x = x.clone();
    for b in blocks {
        x = convnext(&x, b, eps)?;
    }
    Ok(x)
}

/// Причинный ConvNeXt-блок: глубинная свёртка k7, LayerNorm, pw1, GELU, pw2, вычет.
fn convnext(x: &Tensor, blk: &ConvNext, eps: f32) -> Result<Tensor> {
    let (_c, t) = x.dims2()?;
    let k = blk.dw.dim(1)?;
    // Слева нули (паддинг k-1): h[:, t] = Σ_k x[:, t+k-(K-1)]·w[:, k].
    let xp = x.pad_with_zeros(1, k - 1, 0)?; // [C, T+K-1]
    let mut h: Option<Tensor> = None;
    for j in 0..k {
        let shifted = xp.narrow(1, j, t)?; // [C, T]
        let col = blk.dw.narrow(1, j, 1)?; // [C, 1]
        let term = shifted.broadcast_mul(&col)?;
        h = Some(match h {
            None => term,
            Some(prev) => prev.add(&term)?,
        });
    }
    let h = h
        .context("Свёртка не собрана")?
        .broadcast_add(&blk.dw_bias.unsqueeze(1)?)?;
    let h = layer_norm_channels(&h, &blk.norm_w, &blk.norm_b, eps)?;
    let h = blk
        .pw1
        .matmul(&h)?
        .broadcast_add(&blk.pw1_bias.unsqueeze(1)?)?;
    let h = h.gelu_erf()?;
    let h = blk
        .pw2
        .matmul(&h)?
        .broadcast_add(&blk.pw2_bias.unsqueeze(1)?)?;
    Ok(x.add(&h)?)
}

/// LayerNorm по каналам: x [C, T] → каждый столбец нормируется по C.
fn layer_norm_channels(x: &Tensor, w: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor> {
    let (c, t) = x.dims2()?;
    let xt = x.transpose(0, 1)?.contiguous()?; // [T, C]
    let out = candle_nn::ops::layer_norm(&xt, w, b, eps)?;
    Ok(out.transpose(0, 1)?.reshape(&[c, t])?.contiguous()?)
}

/// Patch-up ×h: out[ch, t·h + j] = in[ch·h + j, t].
fn patch_up(x: &Tensor, h: usize) -> Result<Tensor> {
    let (c, t) = x.dims2()?;
    let out_c = c / h;
    Ok(x
        .reshape(&[out_c, h, t])?
        .transpose(1, 2)?
        .contiguous()?
        .reshape(&[out_c, t * h])?
        .contiguous()?)
}

/// Один слой MOSS: преднормированное окновое каузальное внимание + GELU-FFN,
/// с по-канальными layer scale. eps обеих норм — cfg.ln_eps (zac.cpp:754/808).
/// `pos0` — абсолютная позиция первого столбца x (в единицах стадии):
/// RoPE обязана считать углы от абсолютных позиций, иначе кусочный декод
/// получает другие f32-углы, чем цельный, и швы расходятся ~1e-4.
#[allow(clippy::too_many_arguments)]
fn moss_layer(
    x: Tensor,
    w: &MossLayerW,
    hp: &StageHp,
    ln_eps: f32,
    pos0: usize,
    device: &Device,
) -> Result<Tensor> {
    let (d, t) = x.dims2()?;
    let nh = hp.n_head;
    let hd = d / nh;

    let h = layer_norm_channels(&x, &w.norm1_w, &w.norm1_b, ln_eps)?;
    let qkv = w.qkv.matmul(&h)?; // [3d, T]
    let q = qkv.narrow(0, 0, d)?;
    let k = qkv.narrow(0, d, d)?;
    let v = qkv.narrow(0, 2 * d, d)?;

    // [nh, hd, T] → [T, nh, hd] → RoPE → обратно [nh, hd, T].
    let rope = |z: &Tensor| -> Result<Tensor> {
        let zt = z
            .transpose(0, 1)?
            .contiguous()?
            .reshape(&[t, nh, hd])?
            .contiguous()?;
        let roped = rope_apply(&zt, pos0, ROPE_THETA, device)?;
        Ok(roped
            .reshape(&[t, nh * hd])?
            .transpose(0, 1)?
            .contiguous()?
            .reshape(&[d, t])?
            .contiguous()?)
    };
    let q = rope(&q)?; // [d, T]
    let k = rope(&k)?;

    // Головы: [nh, hd, T] — строки d нарезаны по nh групп по hd.
    let split_heads = |z: &Tensor| -> Result<Tensor> {
        Ok(z.reshape(&[nh, hd, t])?.contiguous()?)
    };
    let qh = split_heads(&q)?; // [nh, hd, T]
    let kh = split_heads(&k)?;
    let vh = split_heads(&v)?;

    // Окновое каузальное внимание блоками по 512 запросов.
    let mut outs: Vec<Tensor> = Vec::new();
    let mut a = 0usize;
    while a < t {
        let end = (a + ATTN_BLOCK).min(t);
        let nq = end - a;
        let left = a.saturating_sub(hp.context - 1);
        let nk = end - left;

        let qb = qh.narrow(2, a, nq)?.transpose(1, 2)?.contiguous()?; // [nh, nq, hd]
        let kb = kh.narrow(2, left, nk)?.transpose(1, 2)?.contiguous()?; // [nh, nk, hd]
        let vb = vh.narrow(2, left, nk)?.transpose(1, 2)?.contiguous()?; // [nh, nk, hd]

        let scores = qb.matmul(&kb.transpose(1, 2)?)?; // [nh, nq, nk]
        // Масштаб внимания 1/√hd перед softmax (ggml_soft_max_ext).
        let scale = 1.0 / (hd as f32).sqrt();
        let scores = scores.affine(scale as f64, 0.0)?;
        let mask = attn_mask(nq, nk, a, left, hp.context, device)?;
        let probs = softmax(&scores.broadcast_add(&mask.unsqueeze(0)?)?, 2)?;
        let ob = probs.matmul(&vb)?; // [nh, nq, hd]

        // Головы обратно в d: [T, nh, hd] → [T, d] → [d, T] (головы-мажор).
        let ob = ob
            .transpose(0, 1)?
            .contiguous()?
            .reshape(&[nq, d])?
            .transpose(0, 1)?
            .contiguous()?;
        outs.push(ob);
        a = end;
    }
    let mut o = outs[0].clone();
    for part in outs.iter().skip(1) {
        o = Tensor::cat(&[&o, part], 1)?;
    }
    let o = w
        .wo
        .matmul(&o)?
        .broadcast_mul(&w.attn_scale.unsqueeze(1)?)?; // [d, T]
    let x = x.add(&o)?;

    let h2 = layer_norm_channels(&x, &w.norm2_w, &w.norm2_b, ln_eps)?;
    let f1 = w.ffn1.matmul(&h2)?.gelu_erf()?; // [n_ff, T]
    let f2 = w
        .ffn2
        .matmul(&f1)?
        .broadcast_mul(&w.ffn_scale.unsqueeze(1)?)?; // [d, T]
    Ok(x.add(&f2)?)
}

/// Аддитивная маска [nq, nk]: 0 для ключей в каузальном окне, −∞ вне.
fn attn_mask(nq: usize, nk: usize, q0: usize, k0: usize, context: usize, device: &Device) -> Result<Tensor> {
    let mut mask = vec![0.0f32; nq * nk];
    for i in 0..nq {
        let qp = (q0 + i) as i64;
        for j in 0..nk {
            let kp = (k0 + j) as i64;
            if kp > qp || qp - kp >= context as i64 {
                mask[i * nk + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(mask, &[nq, nk], device)?)
}

/// RoPE соседними парами на абсолютных позициях: x [T, nh, hd].
fn rope_apply(x: &Tensor, pos0: usize, theta: f32, device: &Device) -> Result<Tensor> {
    let (t, nh, hd) = x.dims3()?;
    let pairs = hd / 2;
    let x4 = x.reshape(&[t, nh, pairs, 2])?.contiguous()?;
    let a = x4.narrow(3, 0, 1)?.squeeze(3)?; // [T, nh, pairs]
    let b = x4.narrow(3, 1, 1)?.squeeze(3)?;

    let mut cos = vec![0.0f32; t * pairs];
    let mut sin = vec![0.0f32; t * pairs];
    for p in 0..t {
        for i in 0..pairs {
            let angle = (pos0 + p) as f32 * theta.powf(-2.0 * i as f32 / hd as f32);
            cos[p * pairs + i] = angle.cos();
            sin[p * pairs + i] = angle.sin();
        }
    }
    let cos = Tensor::from_vec(cos, &[t, 1, pairs], device)?;
    let sin = Tensor::from_vec(sin, &[t, 1, pairs], device)?;
    let ra = (a.broadcast_mul(&cos)? - b.broadcast_mul(&sin)?)?;
    let rb = (a.broadcast_mul(&sin)? + b.broadcast_mul(&cos)?)?;
    let out = Tensor::cat(&[ra.unsqueeze(3)?, rb.unsqueeze(3)?], 3)?;
    Ok(out.reshape(&[t, nh, hd])?.contiguous()?)
}

impl super::engine::AudioDecoder for ZacDecoder {
    fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        ZacDecoder::decode(self, frames)
    }

    fn decode_from(&self, frames: &[Vec<u32>], offset: usize) -> Result<Vec<f32>> {
        ZacDecoder::decode_from(self, frames, offset)
    }

    fn samples_per_frame(&self) -> usize {
        self.cfg.hop as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ДИАГНОСТИКА (временно): покомпонентное сравнение цельного декода
    /// [0..800] и среза [164..723]+offset на позициях ≥ 700.
    #[test]
    #[ignore = "нужны веса ZAC: cargo test -- --ignored seam_component_dump"]
    fn seam_component_dump() {
        let dir = crate::model_fetch::zonos3_dir();
        let dec = ZacDecoder::open(
            &dir.join("zac-keep2.gguf"),
            &crate::tts_zonos::engine::decoder_device(),
        )
        .unwrap();
        let nf = 800usize;
        let mut seed = 0x2545F491u32;
        let frames: Vec<Vec<u32>> = (0..nf)
            .map(|_| {
                (0..dec.cfg.n_codebooks)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        seed % dec.cfg.codebook_size as u32
                    })
                    .collect()
            })
            .collect();
        let from = 164usize;

        // Шаг 1: RVQ+сем для полного и среза.
        let step1 = |frames: &[Vec<u32>]| -> Tensor {
            let c = &dec.cfg;
            let dev = &dec.device;
            let n = frames.len();
            let idx = |cb: usize| -> Tensor {
                let ids: Vec<u32> = frames
                    .iter()
                    .map(|f| f[cb].min(c.codebook_size as u32 - 1))
                    .collect();
                Tensor::from_vec(ids, &[n], dev).unwrap()
            };
            let sem = dec
                .sem_table
                .index_select(&idx(0), 0)
                .unwrap()
                .t()
                .unwrap()
                .contiguous()
                .unwrap()
                .broadcast_add(&dec.sem_bias.unsqueeze(1).unwrap())
                .unwrap();
            let mut x = convnext_chain(&sem, &dec.sem_head, c.convnext_eps).unwrap();
            x = dec
                .sem_out
                .0
                .matmul(&x)
                .unwrap()
                .broadcast_add(&dec.sem_out.1.unsqueeze(1).unwrap())
                .unwrap();
            for cb in 0..c.n_codebooks - 1 {
                let part = dec.res_tables[cb]
                    .index_select(&idx(cb + 1), 0)
                    .unwrap()
                    .t()
                    .unwrap()
                    .contiguous()
                    .unwrap();
                x = x.broadcast_add(&part).unwrap();
            }
            x.broadcast_add(&dec.res_bias.unsqueeze(1).unwrap()).unwrap()
        };
        let lat_full = step1(&frames).contiguous().unwrap();
        let lat_slice = step1(&frames[from..]).contiguous().unwrap();
        {
            let a = lat_full
                .i((.., 700..710))
                .unwrap()
                .contiguous()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let b = lat_slice
                .i((.., 700 - from..710 - from))
                .unwrap()
                .contiguous()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let max = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            println!("латент [700..710]: max diff {max}");
        }

        // Шаг 2: стадии по очереди (полный против среза с pos0).
        let run_stages = |lat: Tensor, offset: usize| -> Vec<Tensor> {
            let c = &dec.cfg;
            let mut x = lat;
            let mut rate = 1usize;
            let mut outs = Vec::new();
            for stage in &dec.stages {
                let hp = &stage.hp;
                let pos0 = offset * rate;
                rate *= hp.patch;
                x = stage.input.matmul(&x).unwrap();
                for layer in &stage.layers {
                    x = moss_layer(x, layer, hp, c.ln_eps, pos0, &dec.device).unwrap();
                }
                if let Some(out) = &stage.output {
                    x = out.matmul(&x).unwrap();
                }
                if hp.patch > 1 {
                    x = patch_up(&x, hp.patch).unwrap();
                }
                outs.push(x.contiguous().unwrap());
            }
            outs
        };
        let stages_full = run_stages(lat_full, 0);
        let stages_slice = run_stages(lat_slice, from);
        for (s, (a, b)) in stages_full.iter().zip(&stages_slice).enumerate() {
            let t = a.dim(1).unwrap();
            // Выход стадии снят ПОСЛЕ patch-up: позиция код-кадра f равна
            // f·rate_s·patch_s (для стадий 0..3 это 2^(s+1), у ствола 16).
            let per_frame = if s < 4 { 1usize << (s + 1) } else { 16 };
            let pos = (700usize * per_frame).min(t.saturating_sub(9));
            let aa = a
                .i((.., pos..pos + 8))
                .unwrap()
                .contiguous()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let pos_slice = ((700 - from) * per_frame).min(t.saturating_sub(9));
            let bb = b
                .i((.., pos_slice..pos_slice + 8))
                .unwrap()
                .contiguous()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let max = aa.iter().zip(&bb).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            println!("стадия {s} @код700 (полная позиция {pos}): max diff {max}");
        }
    }

    /// Швы кусочного декода (критерий приёмки №2): цельный декод массива
    /// кадров против покусочного с carry = required_left_context() и правым
    /// гарантом, как режет блоки движок (последний блок — без гаранта).
    /// Максимальное расхождение склейки — < 1e-4. Если падает — увеличивать
    /// carry в формуле required_left_context, а не подгонять порог.
    ///
    /// carry по факту ≈ 536 кадров (стадии 150/300/480/480/480 с патчами
    /// 2/2/2/2/1 дают накопленные окна 149+149.5+119.75+59.9+29.9 ≈ 508,
    /// плюс семантика 24 и ствол 3): STREAM_CARRY=150 покрывал только
    /// стадию 0 — швы длиннее ~10 с фразы расходились. Массив длиннее
    /// carry, иначе все срезы стартуют от нуля и швы не проверяются.
    #[test]
    #[ignore = "нужны веса ZAC (~430 МБ): cargo test -p jarvis-core --lib -- --ignored chunked_decode_matches_full"]
    fn chunked_decode_matches_full() {
        let dir = crate::model_fetch::zonos3_dir();
        if !dir.join("zac-keep2.gguf").is_file() {
            panic!("весов нет в {} — тест запускается только с весами", dir.display());
        }
        // Metal, как в бою: декод длинного массива на CPU — минуты.
        let dec = ZacDecoder::open(&dir.join("zac-keep2.gguf"), &crate::tts_zonos::engine::decoder_device())
            .unwrap();
        let nf = 800usize; // ~53 с аудио — заведомо длиннее carry
        let mut seed = 0x2545F491u32;
        let frames: Vec<Vec<u32>> = (0..nf)
            .map(|_| {
                (0..dec.cfg.n_codebooks)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        seed % dec.cfg.codebook_size as u32
                    })
                    .collect()
            })
            .collect();

        let carry = dec.required_left_context();
        let guard = 3usize; // как STREAM_GUARD в engine.rs
        println!("required_left_context = {carry} кадров");
        assert!(carry < nf, "массив {nf} должен быть длиннее carry {carry}");

        let spf = dec.cfg.hop as usize;
        let full = dec.decode(&frames).unwrap();

        /// Срез [from..to], эмит [emit_start..emit_end] — сравнить с цельным.
        /// Срез декодируется с абсолютным офсетом (decode_from): RoPE углы
        /// совпадают с цельным декодом, остаётся только шум устройства.
        let check = |dec: &ZacDecoder, full: &[f32], from: usize, to: usize, emit_start: usize, emit_end: usize| {
            let audio = dec.decode_from(&frames[from..to], from).unwrap();
            let a = ((emit_start - from) * spf).min(audio.len());
            let b = ((emit_end - from) * spf).min(audio.len());
            let piece = &audio[a..b];
            let expect = &full[emit_start * spf..emit_end * spf];
            assert_eq!(piece.len(), expect.len(), "длина куска [{emit_start}..{emit_end}]");
            piece
                .iter()
                .zip(expect)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };

        // 1. Срез от нуля (блоки первой половины фразы): обязан совпасть точно.
        let d0 = check(&dec, &full, 0, 600 + guard, 200, 600);
        println!("шов от нуля: {d0}");
        assert!(d0 < 1e-4, "срез от нуля разошёлся: {d0}");

        // 2. Настоящий шов на границе формулы: carry покрывает краевое
        // загрязнение (сем 24 + окна стадий 508 + ствол 3 = 535.06 кадров),
        // абсолютный офсет RoPE убирает угловой разброс — остаётся шум.
        let emit_start = 700usize;
        let from = emit_start - carry;
        let d1 = check(&dec, &full, from, 720 + guard, emit_start, 720);
        println!("шов на границе формулы (emit {emit_start}, from {from}): {d1}");
        assert!(d1 < 1e-4, "шов на границе формулы разошёлся: {d1}");

        // 3. Последний блок без гаранта, как в движке: тоже точный.
        let d2 = check(&dec, &full, 780usize.saturating_sub(carry), nf, 780, nf);
        println!("хвост без гаранта: {d2}");
        assert!(d2 < 1e-4, "хвостовой блок разошёлся: {d2}");
    }

    /// Ресемплер 44160→44100 на чистой синусоиде: частота обязана
    /// сохраниться, амплитуда — остаться в пределах пары процентов.
    #[test]
    fn resampler_keeps_sine() {
        let dir = crate::model_fetch::zonos3_dir();
        if !dir.join("zac-keep2.gguf").is_file() {
            return;
        }
        let dec = ZacDecoder::open(&dir.join("zac-keep2.gguf"), &Device::Cpu).unwrap();
        // 1.5 с синуса 440 Гц на 44160 Гц.
        let rate = 44160.0f32;
        let freq = 440.0f32;
        let n_in = 66_240; // 1.5 с
        let x: Vec<f32> = (0..n_in)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * freq * i as f32 / rate).sin())
            .collect();
        let y = dec.resample(&x).unwrap();
        assert_eq!(y.len(), 66_150, "длина ресемпла");
        // Частота: нули пересекаются 2·freq раз за секунду.
        let mut crossings = 0usize;
        for w in y.windows(2) {
            if w[0] <= 0.0 && w[1] > 0.0 {
                crossings += 1;
            }
        }
        let hz = crossings as f32 / (y.len() as f32 / 44100.0);
        println!("частота на выходе: {hz:.1} Гц (ожидаю ~440)");
        assert!((hz - freq).abs() < 8.0, "частота уехала: {hz}");
        // Амплитуда: пик в установившемся режиме.
        let peak = y[4000..y.len() - 4000]
            .iter()
            .fold(0f32, |m, v| m.max(v.abs()));
        println!("пик амплитуды: {peak:.4} (ожидаю ~0.5)");
        assert!((peak - 0.5).abs() < 0.05, "амплитуда уехала: {peak}");
    }

    /// Синтетический спектр: только DC с модулем A и нулевой фазой →
    /// после iSTFT+OLA+микшера сигнал обязан быть константой (плюс краевые
    /// эффекты первых сэмплов). Проверяет всю аудио-цепочку хоста.
    #[test]
    fn dc_spectrum_decodes_constant() {
        let dir = crate::model_fetch::zonos3_dir();
        if !dir.join("zac-keep2.gguf").is_file() {
            return; // весов нет на этой машине — тест не о них
        }
        let dec = ZacDecoder::open(&dir.join("zac-keep2.gguf"), &Device::Cpu).unwrap();
        let head_frames = 12usize;
        let mk_spec = |branch: usize| -> Vec<f32> {
            let n = dec.cfg.n_fft[branch];
            let bins = n / 2 + 1;
            let mut spec = vec![0.0f32; head_frames * (n + 2)];
            for t in 0..head_frames {
                let f = &mut spec[t * (n + 2)..(t + 1) * (n + 2)];
                f[0] = 2f32.ln(); // DC: модуль 2, фаза 0
                f[bins] = 0.0;
                for k in 1..bins {
                    f[k] = -20.0; // e^-20 ≈ 0
                    f[bins + k] = 0.0;
                }
            }
            spec
        };
        {
            let specs = [mk_spec(0), mk_spec(1)];
            let mixed = dec.istft_mix(&specs, head_frames).unwrap();
            // Хвост исключаем: там правое паддинговое окно микшера.
            let body = &mixed[64..mixed.len() - 64];
            let mean: f32 = body.iter().sum::<f32>() / body.len() as f32;
            let max_dev = body.iter().fold(0f32, |m, v| m.max((v - mean).abs()));
            println!(
                "микс: len {}, среднее {mean:.4}, макс. отклонение {max_dev:.4}",
                mixed.len()
            );
            assert!(
                max_dev < 0.05,
                "DC-спектр дал неконстантный сигнал: среднее {mean}, отклонение {max_dev}"
            );
        }
    }


    use super::*;

    /// RoPE: поворот соседней пары на угол θ = pos·ω.
    #[test]
    fn rope_rotates_adjacent_pairs() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 0.0, 2.0, 0.0], &[1, 1, 4], &device).unwrap();
        let out = rope_apply(&x, 3, 10000.0, &device).unwrap();
        let v = out.to_vec3::<f32>().unwrap();
        // hd=4, пары (1,0) и (2,0), угол pos·ω_i, ω_i = θ^(-2i/4).
        let w0 = 10000f32.powf(0.0); // 1
        let w1 = 10000f32.powf(-0.5);
        let a = 3.0f32 * w0;
        let b = 3.0f32 * w1;
        assert!((v[0][0][0] - a.cos()).abs() < 1e-5);
        assert!((v[0][0][1] - a.sin()).abs() < 1e-5);
        assert!((v[0][0][2] - 2.0 * b.cos()).abs() < 1e-5);
        assert!((v[0][0][3] - 2.0 * b.sin()).abs() < 1e-5);
    }

    /// Patch-up: out[ch, t·h + j] = in[ch·h + j, t].
    #[test]
    fn patch_up_interleaves() {
        let device = Device::Cpu;
        // 4 канала × 2 кадра: канал c кадра t = 10c + t.
        let data: Vec<f32> = (0..4)
            .flat_map(|c| (0..2).map(move |t| (10 * c + t) as f32))
            .collect();
        let x = Tensor::from_vec(data, &[4, 2], &device).unwrap();
        let out = patch_up(&x, 2).unwrap().to_vec2::<f32>().unwrap();
        // 2 канала × 4 позиции.
        assert_eq!(out.len(), 2);
        // Канал 0: in[0,0], in[1,0], in[0,1], in[1,1] = 0, 10, 1, 11.
        assert_eq!(out[0], vec![0.0, 10.0, 1.0, 11.0]);
        // Канал 1: in[2,0], in[3,0], in[2,1], in[3,1] = 20, 30, 21, 31.
        assert_eq!(out[1], vec![20.0, 30.0, 21.0, 31.0]);
    }

    /// Маска: каузальное окно из context ключей.
    #[test]
    fn mask_is_windowed_causal() {
        let device = Device::Cpu;
        let m = attn_mask(3, 4, 2, 0, 3, &device)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        // Запросы на позициях 2..5, ключи 0..4. Окно: kp <= qp && qp - kp < 3.
        let fin = |v: f32| v.is_finite();
        assert!(fin(m[0][0]) && fin(m[0][1]) && fin(m[0][2]) && !fin(m[0][3]));
        assert!(fin(m[1][1]) && fin(m[1][2]) && fin(m[1][3]));
        assert!(!fin(m[1][0]));
        assert!(fin(m[2][2]) && fin(m[2][3]));
        assert!(!fin(m[2][0]) && !fin(m[2][1]));
    }

    /// LayerNorm по каналам совпадает с ручным расчётом.
    #[test]
    fn channel_layer_norm() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[2, 2], &device).unwrap();
        let w = Tensor::from_vec(vec![1.0f32, 1.0], &[2], &device).unwrap();
        let b = Tensor::from_vec(vec![0.0f32, 0.0], &[2], &device).unwrap();
        let out = layer_norm_channels(&x, &w, &b, 1e-5)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        // Столбец [1, 5]: среднее 3, std 2 → [-1, 1].
        assert!((out[0][0] - -1.0).abs() < 1e-4 && (out[1][0] - 1.0).abs() < 1e-4);
        assert!((out[0][1] - -1.0).abs() < 1e-4 && (out[1][1] - 1.0).abs() < 1e-4);
    }
}
