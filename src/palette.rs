//! Продуктовые токены оформления: палитра времени суток и экранные размеры.
//!
//! Акцент каждого периода выведен из семейства его неба, а не взят отдельно:
//! кнопка действия обязана принадлежать сцене, а не спорить с ней. Поэтому
//! утро — розовый по сливовому небу, день — лазурный по синему, вечер —
//! алый по багровому, ночь — фиалковый по глубокому фиолетовому.

use gpui_kit::{Hsla, Pixels, px, rgb, rgba};

use crate::scene::TimeOfDay;

/// Кегль крупных часов.
pub const CLOCK_FONT_SIZE: Pixels = px(68.);
/// Диаметр круглой кнопки питания.
pub const POWER_BUTTON_SIZE: Pixels = px(86.);
/// Радиус круглой кнопки питания.
pub const POWER_BUTTON_RADIUS: Pixels = px(43.);
/// Диаметр кольца загрузки — из него раскрывается интерфейс.
pub const RING_SIZE: Pixels = px(252.);

/// Цветовая схема периода суток.
#[derive(Clone, Copy, Debug)]
pub struct Palette {
    /// Акцент: кнопка действия, выбранная строка, свечение.
    pub accent: Hsla,
    /// Тёмный оттенок акцента.
    pub accent_dark: Hsla,
    /// Светлый оттенок акцента.
    pub accent_light: Hsla,
    /// Текст и значки на сплошном акценте.
    pub on_accent: Hsla,
    /// Фон окна.
    pub background: Hsla,
    /// Полупрозрачная поверхность чипов поверх сцены.
    pub surface: Hsla,
    /// Непрозрачное стекло: круг запуска, меню, диалоги.
    pub glass: Hsla,
    /// Стекло, подкрашенное акцентом: кнопка выбора действия.
    pub tint: Hsla,
    /// Тонкая граница поверхностей.
    pub hairline: Hsla,
    /// Цвет затемняющих вуалей.
    pub scrim: Hsla,
    /// Текст поверх сцены.
    pub on_scene: Hsla,
    /// Приглушённый текст.
    pub on_scene_muted: Hsla,
}

impl Palette {
    /// Палитра для периода суток.
    pub fn for_time_of_day(time: TimeOfDay) -> Self {
        let (accent, background) = match time {
            // Сливовое небо рассвета → розовый акцент.
            TimeOfDay::Morning => (rgb(0xf472b6), rgb(0x231a33)),
            // Синее небо дня → лазурный акцент.
            TimeOfDay::Day => (rgb(0x38bdf8), rgb(0x0f2231)),
            // Багровое небо заката → алый акцент.
            TimeOfDay::Evening => (rgb(0xfb7185), rgb(0x1f1226)),
            // Фиолетовое небо ночи → фиалковый акцент.
            TimeOfDay::Night => (rgb(0xa78bfa), rgb(0x0c0a1a)),
        };

        let accent: Hsla = accent.into();
        let background: Hsla = background.into();
        let on_scene: Hsla = rgb(0xffffff).into();
        let glass = background.blend(on_scene.opacity(0.07));

        Self {
            accent,
            accent_dark: shade(accent, 0.58),
            accent_light: shade(accent, 1.32),
            on_accent: background,
            background,
            surface: rgba(0x2a2d35cc).into(),
            glass,
            tint: glass.blend(accent.opacity(0.12)),
            hairline: on_scene.opacity(0.14),
            scrim: background,
            on_scene,
            on_scene_muted: rgba(0xb3ffffff).into(),
        }
    }
}

/// Изменить светлоту цвета.
fn shade(color: Hsla, factor: f32) -> Hsla {
    Hsla {
        h: color.h,
        s: color.s,
        l: (color.l * factor).clamp(0.0, 1.0),
        a: color.a,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accent_shares_hue_family_with_background() {
        for period in TimeOfDay::ALL {
            let palette = Palette::for_time_of_day(period);
            let delta = (palette.accent.h - palette.background.h).abs();
            let distance = delta.min(1.0 - delta);
            assert!(
                distance < 0.22,
                "{period:?}: акцент {:.3} слишком далёк от фона {:.3}",
                palette.accent.h,
                palette.background.h
            );
        }
    }

    #[test]
    fn surfaces_are_opaque_where_text_sits_on_them() {
        for period in TimeOfDay::ALL {
            let palette = Palette::for_time_of_day(period);
            assert_eq!(palette.glass.a, 1.0);
            assert_eq!(palette.tint.a, 1.0);
        }
    }
}
