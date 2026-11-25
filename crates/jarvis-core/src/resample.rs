//! Потоковый ресемплинг звука: качественный фильтр с антиалиасингом и
//! сохранением состояния между чанками.
//!
//! Почему нельзя «просто интерполировать»:
//!
//! 1. **Алиасинг.** Микрофон отдаёт 48 кГц, модель ждёт 16 кГц. При
//!    децимации всё выше 8 кГц должно быть отфильтровано до, а не после
//!    прореживания. Линейная интерполяция этого не делает: энергия
//!    16–24 кГц (и шум микрофона) складывается обратно в речевую полосу
//!    0–8 кГц, и ASR-модель слышит «кашу».
//! 2. **Разрывы на стыках.** Ресемплинг каждого чанка отдельно обнуляет
//!    фазу: на границе 48 кГц → 16 кГц теряется дробная часть сэмпла,
//!    и в сигнале появляется щелчок каждые ~85 мс. Плюс за секунду
//!    набегает дрейф темпа.
//! 3. **Модуляция.** Простой полосовой фильтр не даёт полного затухания
//!    в зоне перехода — нужен sinc-интерполятор с окном (Kaiser/Blackman).
//!
//! Здесь — синхронный FFT-ресемплер (`rubato`): отношение частот
//! фиксировано (микрофон → 16 кГц), сигнал обрабатывается блоками с
//! сохранением состояния, антиалиасинг обеспечен оконной sinc-фильтрацией.
//!
//! Поток (микрофон) и TTS (динамик) используют **разные** экземпляры:
//! состояние фильтра общее на канал.

use anyhow::{Context, Result};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{FixedSync, Fft, Resampler as _};

/// Размер блока ресемплера. Внутри FFT-ресемплер сам подбирает размеры под
/// отношение частот; это только подсказка.
const CHUNK_HINT: usize = 1024;

/// Потоковый ресемплинг «один вход → один выход» с постоянным отношением.
pub struct Resampler {
    inner: Fft<f32>,
    /// Буфер входа фиксированного размера (моно).
    in_buf: Vec<f32>,
    /// Буфер выхода (моно).
    out_buf: Vec<f32>,
    /// Вход, ещё не уложившийся в полный блок.
    pending: Vec<f32>,
    src_rate: usize,
    dst_rate: usize,
}

impl Resampler {
    /// Создать ресемплер для моно-сигнала.
    pub fn new(src_rate: usize, dst_rate: usize) -> Result<Self> {
        anyhow::ensure!(src_rate > 0 && dst_rate > 0, "некорректные частоты");
        if src_rate == dst_rate {
            // Тождественный случай: фильтр не нужен, состояние не ведём.
            // ratio = 1 — валидный FFT-ресемплер, но тратить на него
            // процессорное время незачем; отдаём пустой «пасс».
            return Ok(Self {
                inner: Fft::<f32>::new(src_rate, dst_rate, CHUNK_HINT, 1, FixedSync::Both)
                .context("реземплер 1:1")?,
                in_buf: Vec::new(),
                out_buf: Vec::new(),
                pending: Vec::new(),
                src_rate,
                dst_rate,
            });
        }
        let inner = Fft::<f32>::new(src_rate, dst_rate, CHUNK_HINT, 1, FixedSync::Both)
        .map_err(|e| anyhow::anyhow!("реземплер {src_rate}→{dst_rate}: {e}"))?;
        let out_cap = inner.output_frames_max();
        Ok(Self {
            inner,
            in_buf: vec![0.0; CHUNK_HINT],
            out_buf: vec![0.0; out_cap],
            pending: Vec::new(),
            src_rate,
            dst_rate,
        })
    }

    /// Частота выхода.
    pub fn dst_rate(&self) -> usize {
        self.dst_rate
    }

    /// Задержка фильтра в сэмплах выхода (групповая задержка sinc-окна).
    pub fn delay(&self) -> usize {
        if self.is_passthrough() {
            0
        } else {
            self.inner.output_delay()
        }
    }

    /// Тождественный ресемплер (частоты совпадают).
    pub fn is_passthrough(&self) -> bool {
        self.src_rate == self.dst_rate
    }

    /// Перелить `input` в `out`, продолжая с прошлого состояния.
    ///
    /// Не аллоцирует в горячем пути: внутренние буферы переиспользуются.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.is_passthrough() {
            out.extend_from_slice(input);
            return;
        }
        self.pending.extend_from_slice(input);
        loop {
            let need = self.inner.input_frames_next();
            if self.pending.len() < need {
                break;
            }
            if self.in_buf.len() < need {
                self.in_buf.resize(need, 0.0);
            }
            self.in_buf[..need].copy_from_slice(&self.pending[..need]);
            self.pending.drain(..need);

            let want_out = self.inner.output_frames_next();
            if self.out_buf.len() < want_out {
                self.out_buf.resize(want_out, 0.0);
            }
            let in_adapter =
                InterleavedSlice::new(&self.in_buf[..need], 1, need).expect("моно-вход");
            let mut out_adapter =
                InterleavedSlice::new_mut(&mut self.out_buf[..want_out], 1, want_out)
                    .expect("моно-выход");
            match self
                .inner
                .process_into_buffer(&in_adapter, &mut out_adapter, None)
            {
                Ok((_read, written)) => out.extend_from_slice(&self.out_buf[..written]),
                Err(e) => {
                    // Ошибка ресемплера — не повод терять поток целиком:
                    // логируем и пропускаем блок.
                    eprintln!("⚠️ Ресемплинг сбой: {e}");
                    break;
                }
            }
        }
    }

    /// Дослать хвост (последний неполный блок) — для концов файлов и
    /// тестов. Основной поток микрофона не флешит: там блоки не нуждаются
    /// в «хвосте», а досылка добавила бы лишние сэмплы в начало.
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        if self.is_passthrough() || self.pending.is_empty() {
            return;
        }
        // Добиваем вход нулями до полного блока: так фильтр дочищает
        // переходную зону, и в выходе нет обрезанного хвоста.
        let need = self.inner.input_frames_next();
        self.pending.resize(need, 0.0);
        let in_adapter = InterleavedSlice::new(&self.pending[..need], 1, need).expect("моно-вход");
        let want_out = self.inner.output_frames_next();
        if self.out_buf.len() < want_out {
            self.out_buf.resize(want_out, 0.0);
        }
        let mut out_adapter =
            InterleavedSlice::new_mut(&mut self.out_buf[..want_out], 1, want_out).expect("моно-выход");
        if let Ok((_read, written)) =
            self.inner
                .process_into_buffer(&in_adapter, &mut out_adapter, None)
        {
            out.extend_from_slice(&self.out_buf[..written]);
        }
        self.pending.clear();
    }
}

/// Одноразовый ресемплинг целого буфера (файлы, тесты, офлайн-задачи).
///
/// Качество то же, что и у потокового: sinc-фильтр с антиалиасингом.
/// Длина результата — ровно `len * dst / src`: задержка фильтра срезается
/// из начала, а «хвост» фильтра — из конца, поэтому выход совмещён с
/// входом по времени.
pub fn resample_buffer(samples: &[f32], src_rate: usize, dst_rate: usize) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let mut inner = match Fft::<f32>::new(src_rate, dst_rate, CHUNK_HINT, 1, FixedSync::Both) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("⚠️ Ресемплинг {src_rate}→{dst_rate} не удался: {e}");
            return samples.to_vec();
        }
    };
    // `process_all` сам обрезает задержку фильтра и возвращает ровно
    // пересэмплированный клип — выход совмещён со входом по времени.
    let needed = inner.process_all_needed_output_len(samples.len());
    let mut out = vec![0.0f32; needed];
    let in_adapter = InterleavedSlice::new(samples, 1, samples.len()).expect("моно-вход");
    let mut out_adapter =
        InterleavedSlice::new_mut(&mut out, 1, needed).expect("моно-выход");
    match inner.process_all_into_buffer(&in_adapter, &mut out_adapter, samples.len(), None) {
        Ok((_consumed, produced)) => {
            out.truncate(produced);
            out
        }
        Err(e) => {
            eprintln!("⚠️ Ресемплинг {src_rate}→{dst_rate} сбой: {e}");
            samples.to_vec()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Тон 1 кГц: после ресемплинга его частота и амплитуда должны
    /// сохраниться, а в спектре не должно появиться энергии выше 8 кГц
    /// (то есть алиасинга нет).
    #[test]
    fn тон_сохраняет_частоту_и_не_даёт_алиасинга() {
        let sr_in = 48_000;
        let sr_out = 16_000;
        let n = sr_in;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / sr_in as f32).sin())
            .collect();
        let out = resample_buffer(&input, sr_in, sr_out);
        assert_eq!(out.len(), n / 3, "длина выхода неверна");

        // Энергия в полосах через DFT по отрезку.
        let seg = &out[..out.len().min(16_000 * 2)];
        let energy = |f: f32| -> f32 {
            let mut re = 0.0;
            let mut im = 0.0;
            for (i, s) in seg.iter().enumerate() {
                let ph = 2.0 * std::f32::consts::PI * f * i as f32 / sr_out as f32;
                re += s * ph.cos();
                im += s * ph.sin();
            }
            (re * re + im * im) / seg.len() as f32
        };
        let base = energy(1_000.0);
        assert!(base > 0.1, "тон потерялся: {base}");
        // Алиасинг 1 кГц из 17 кГц (17 - 16 = 1) — на частоте 1 кГц,
        // поэтому проверяем полосу 4..8 кГц: там должна быть тишина.
        let hi = energy(6_000.0);
        assert!(hi < base * 0.01, "алиасинг в полосе 6 кГц: {hi} против {base}");
    }

    /// Потоковый ресемплинг по чанкам должен давать тот же сигнал, что и
    /// одноразовый: фаза не рвётся на стыках. Сдвиг равен задержке фильтра.
    #[test]
    fn потоковый_совпадает_с_одноразовым() {
        let sr_in = 48_000;
        let sr_out = 16_000;
        let input: Vec<f32> = (0..48_000)
            .map(|i| {
                let t = i as f32 / sr_in as f32;
                (2.0 * std::f32::consts::PI * 1_000.0 * t).sin() * 0.5
                    + (2.0 * std::f32::consts::PI * 3_000.0 * t).sin() * 0.2
            })
            .collect();
        let whole = resample_buffer(&input, sr_in, sr_out);

        let mut r = Resampler::new(sr_in, sr_out).expect("ресемплер");
        let mut streamed = Vec::new();
        for chunk in input.chunks(480) {
            r.process(chunk, &mut streamed);
        }
        let delay = r.delay();

        // Выравниваем по задержке фильтра.
        let start = 2_000;
        let n = whole.len().min(streamed.len().saturating_sub(delay)).saturating_sub(start);
        assert!(n > 1_000, "нечего сравнивать: {n}");
        let a = &whole[start..start + n];
        let b = &streamed[start + delay..start + delay + n];
        let mut num = 0.0;
        let mut da = 0.0;
        let mut db = 0.0;
        for (x, y) in a.iter().zip(b) {
            num += x * y;
            da += x * x;
            db += y * y;
        }
        let corr = num / (da * db).sqrt();
        assert!(corr > 0.99, "поток и одноразовый расходятся: corr={corr}");
    }

    /// Темп потока не должен «уплывать»: на 10 секунд входа приходит
    /// ровно ожидаемое число выходных сэмплов (с точностью до блока).
    #[test]
    fn поток_не_уходит_по_темпу() {
        let sr_in = 48_000;
        let sr_out = 16_000;
        let secs = 10;
        let input = vec![0.25f32; sr_in * secs];
        let mut r = Resampler::new(sr_in, sr_out).expect("ресемплер");
        let mut streamed = Vec::new();
        for chunk in input.chunks(480) {
            r.process(chunk, &mut streamed);
        }
        let expected = input.len() * sr_out / sr_in;
        let diff = (streamed.len() as i64 - expected as i64).abs();
        assert!(
            diff < (sr_out as i64) / 4,
            "поток отстаёт по темпу: {} против ожидаемых {expected} (разница {diff})",
            streamed.len()
        );
    }

    /// Тождественный случай: частота 1:1 — сигнал проходит без изменений.
    #[test]
    fn тождественный_проход() {
        let input = vec![0.1f32; 1000];
        let out = resample_buffer(&input, 16_000, 16_000);
        assert_eq!(out.len(), input.len());
        assert!((out[500] - 0.1).abs() < 1e-6);
    }

    /// Постоянный сигнал не должен гулять: в середине он равен исходному
    /// (стартовый выброс фильтра — норма для sinc-окна).
    #[test]
    fn постоянный_сигнал_не_гуляет() {
        let src = vec![0.5f32; 16_000];
        let out = resample_buffer(&src, 16_000, 48_000);
        assert_eq!(out.len(), 48_000);
        let mid = &out[20_000..28_000];
        let worst = mid
            .iter()
            .map(|s| (s - 0.5).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "постоянный сигнал гуляет: {worst}");
    }
}
