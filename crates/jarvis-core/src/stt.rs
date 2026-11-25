//! STT: NVIDIA FastConformer-Hybrid Large RU (CC-BY-4.0) — собственная
//! реализация на candle (см. [`crate::asr`]).
//!
//! Фасад для движка: тот же контракт, что был у старого GigaAM-бэкенда
//! (`load` + `transcribe`), чтобы интеграция не менялась.

use crate::asr::model::FastConformer;
use anyhow::Result;
use std::path::Path;

pub struct Stt {
    model: FastConformer,
}

impl Stt {
    /// Скачивает модель при необходимости и загружает её (Metal на macOS).
    pub fn load(models_dir: &Path) -> Result<Self> {
        let model = FastConformer::load(models_dir)?;
        Ok(Self { model })
    }

    /// Загрузить модель на заданном устройстве: wake-детектору нужен свой
    /// CPU-экземпляр — непрерывный подсчёт окон не выживает на Metal
    /// (~0.7 с диспетчизации на окно, воркер захлёбывается; см.
    /// wake::create_engine).
    pub fn load_on(models_dir: &Path, device: candle_core::Device) -> Result<Self> {
        let model = FastConformer::load_on(models_dir, device)?;
        Ok(Self { model })
    }

    /// Загрузить модель на CPU — для wake-детектора и для проверки фразы
    /// из интерфейса. Своя обёртка, чтобы вызывающий код не зависел от
    /// candle: выбор устройства — деталь реализации ядра.
    pub fn load_cpu(models_dir: &Path) -> Result<Self> {
        Self::load_on(models_dir, candle_core::Device::Cpu)
    }

    /// Распознавание готового буфера 16 кГц mono f32 (RNN-T greedy).
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        let transcript = self.model.transcribe(samples)?;
        Ok(transcript.text.trim().to_string())
    }

    /// Доступ к полной модели (декодеры, словарь Jarvis, тайминги).
    pub fn model(&self) -> &FastConformer {
        &self.model
    }
}
