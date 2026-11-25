//! Тема приложения, следующая за палитрой сцены.
//!
//! Компоненты gpui-kit (меню, диалог, тултип) красятся семантическими токенами
//! темы. Пока эти токены системные, любое меню выглядит чужеродным рядом со
//! сценой. Здесь палитра периода суток проецируется в токены, поэтому выбор
//! действия и подтверждение наследуют стиль интерфейса без своей отрисовки.

use gpui_kit::App;
use gpui_kit::component::{Theme, ThemeColor};

use crate::palette::Palette;
use crate::scene::TimeOfDay;

/// Применить палитру периода к теме. Вызывать только при смене периода.
pub fn apply(period: TimeOfDay, cx: &mut App) {
    let palette = Palette::for_time_of_day(period);

    // Тёмная база даёт корректные значения всем токенам, которые мы не трогаем.
    let mut colors = *ThemeColor::dark();

    colors.background = palette.background;
    colors.foreground = palette.on_scene;
    colors.border = palette.on_scene.opacity(0.14);
    colors.input = palette.on_scene.opacity(0.18);

    // Поверхности всплывающих слоёв: меню, диалог, тултип.
    colors.popover = palette.surface;
    colors.popover_foreground = palette.on_scene;
    colors.overlay = gpui_kit::black().opacity(0.45);

    // Строки меню и списков.
    colors.list = palette.surface;
    colors.list_hover = palette.accent.opacity(0.16);
    colors.list_active = palette.accent.opacity(0.24);
    colors.list_active_border = palette.accent.opacity(0.45);
    colors.accent = palette.accent.opacity(0.18);
    colors.accent_foreground = palette.on_scene;

    colors.muted = palette.surface;
    colors.muted_foreground = palette.on_scene_muted;
    colors.secondary = palette.surface;
    colors.secondary_foreground = palette.on_scene;
    colors.secondary_hover = palette.accent.opacity(0.14);

    // Основное действие — акцент сцены.
    colors.primary = palette.accent;
    colors.primary_foreground = palette.background;
    colors.primary_hover = palette.accent_light;
    colors.primary_active = palette.accent_dark;
    colors.button_primary = palette.accent;
    colors.button_primary_foreground = palette.background;
    colors.button_primary_hover = palette.accent_light;
    colors.button_primary_active = palette.accent_dark;

    colors.button = palette.surface;
    colors.button_foreground = palette.on_scene;
    colors.button_hover = palette.accent.opacity(0.14);
    colors.button_active = palette.accent.opacity(0.22);

    colors.title_bar = palette.background.opacity(0.0);
    colors.title_bar_border = palette.background.opacity(0.0);
    colors.ring = palette.accent;
    colors.selection = palette.accent.opacity(0.28);
    colors.scrollbar_thumb = palette.on_scene.opacity(0.22);
    colors.scrollbar_thumb_hover = palette.on_scene.opacity(0.34);

    Theme::update(cx, |theme| {
        theme.colors = colors;
        theme.radius = gpui_kit::px(10.);
        theme.radius_lg = gpui_kit::px(14.);
    });
}
