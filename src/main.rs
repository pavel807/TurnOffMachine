//! Точка входа: окно, клавиатурные команды и политика закрытия.

use gpui_kit::component::TitleBar;
use gpui_kit::{AppContext as _, KeyBinding, WindowBounds, WindowOptions, px, size};

use turnoff_gpui::app::{
    self, CancelScheduledShutdown, CommitAction, NextAction, PreviousAction, Quit, ToggleScene,
};
use turnoff_gpui::assets::AppAssets;

fn main() {
    gpui_kit::application().with_assets(AppAssets).run(|cx| {
        gpui_kit::init(cx);

        cx.bind_keys([
            KeyBinding::new("tab", ToggleScene, Some("TurnOff")),
            KeyBinding::new("right", NextAction, Some("TurnOff")),
            KeyBinding::new("left", PreviousAction, Some("TurnOff")),
            KeyBinding::new("enter", CommitAction, Some("TurnOff")),
            KeyBinding::new("escape", CancelScheduledShutdown, Some("TurnOff")),
            #[cfg(target_os = "macos")]
            KeyBinding::new("cmd-q", Quit, Some("TurnOff")),
            #[cfg(not(target_os = "macos"))]
            KeyBinding::new("ctrl-q", Quit, Some("TurnOff")),
        ]);

        // База от TitleBar: включает app_owns_titlebar_drag, иначе macOS
        // сам обрабатывает клики по заголовку и задерживает их.
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::centered(None, size(px(440.), px(680.)), cx)),
            window_min_size: Some(size(px(app::MIN_WINDOW_WIDTH), px(app::MIN_WINDOW_HEIGHT))),
            app_id: Some("turnoff-machine".into()),
            ..TitleBar::window_options()
        };

        gpui_kit::open_window(options, cx, |window, cx| {
            cx.new(|cx| app::RootView::new(window, cx))
        })
        .expect("не удалось открыть окно");

        // Однооконное приложение: закрытие окна завершает процесс.
        cx.on_window_closed(|cx, _| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();
    });
}
