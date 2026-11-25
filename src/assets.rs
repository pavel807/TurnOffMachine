//! Ресурсы приложения: фоновые сцены и иконки в формате SVG.
//!
//! Все картинки хранятся вектором и растеризуются под фактический размер
//! интерфейса — см. [`crate::vector`]. Стандартный набор gpui-kit при этом
//! остаётся доступен: источник ресурсов приложения отдаёт и его иконки.

use std::borrow::Cow;

use gpui_kit::assets::IconNamed;
use gpui_kit::{AssetSource, Result, SharedString};
use rust_embed::RustEmbed;

/// SVG-файлы приложения: сцены и иконки.
#[derive(RustEmbed)]
#[folder = "assets"]
pub struct AppFiles;

/// Прочитать SVG-файл приложения по пути относительно `assets`.
pub fn svg(path: &str) -> Option<Cow<'static, [u8]>> {
    AppFiles::get(path).map(|file| file.data)
}

/// Иконки приложения.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppIcon {
    /// Выключение питания.
    Power,
    /// Перезагрузка.
    Restart,
    /// Сон.
    Sleep,
    /// Гибернация.
    Hibernate,
    /// Рассвет.
    Sunrise,
    /// День.
    Sun,
    /// Закат.
    Sunset,
    /// Ночь.
    CloudMoon,
}

impl AppIcon {
    /// Все иконки приложения.
    pub const ALL: [Self; 8] = [
        Self::Power,
        Self::Restart,
        Self::Sleep,
        Self::Hibernate,
        Self::Sunrise,
        Self::Sun,
        Self::Sunset,
        Self::CloudMoon,
    ];

    /// Путь к SVG-файлу иконки относительно `assets`.
    pub fn asset_path(self) -> &'static str {
        match self {
            Self::Power => "icons/power.svg",
            Self::Restart => "icons/restart.svg",
            Self::Sleep => "icons/sleep.svg",
            Self::Hibernate => "icons/hibernate.svg",
            Self::Sunrise => "icons/sunrise.svg",
            Self::Sun => "icons/sun.svg",
            Self::Sunset => "icons/sunset.svg",
            Self::CloudMoon => "icons/cloud-moon.svg",
        }
    }

    /// Содержимое SVG-иконки.
    pub fn svg(self) -> Cow<'static, [u8]> {
        svg(self.asset_path()).unwrap_or(Cow::Borrowed(b""))
    }
}

impl IconNamed for AppIcon {
    fn path(self) -> SharedString {
        self.asset_path().into()
    }
}

/// Единый источник ресурсов приложения.
pub struct AppAssets;

impl AssetSource for AppAssets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        if let Some(file) = svg(path) {
            return Ok(Some(file));
        }
        gpui_kit::assets::Assets.load(path)
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        let mut listed = AppFiles::iter()
            .filter(|name| name.starts_with(path))
            .map(SharedString::from)
            .collect::<Vec<_>>();
        listed.extend(gpui_kit::assets::Assets.list(path)?);
        Ok(listed)
    }
}
