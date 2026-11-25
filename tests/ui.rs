//! UI-интеграционные проверки: выбор действия питания и подтверждение.

use gpui_kit::test::{TestQueryExt, TestWindowExt};
use gpui_kit::{
    AppContext, Bounds, Entity, KeyBinding, Role, TestAppContext, WindowBounds, WindowOptions, px,
    size,
};
use turnoff_gpui::app::{NextAction, PreviousAction, RootView};
use turnoff_gpui::power::PowerAction;

/// Открывает окно с корневым экраном приложения.
fn open_root_view(cx: &mut TestAppContext) -> (gpui_kit::AnyWindowHandle, Entity<RootView>) {
    open_root_view_sized(cx, 460., 740.)
}

/// Открывает окно заданного размера — для проверки узкой раскладки.
fn open_root_view_sized(
    cx: &mut TestAppContext,
    width: f32,
    height: f32,
) -> (gpui_kit::AnyWindowHandle, Entity<RootView>) {
    cx.update(|cx| {
        gpui_kit::open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds::new(
                    Default::default(),
                    size(px(width), px(height)),
                ))),
                ..Default::default()
            },
            cx,
            |window, cx| cx.new(|cx| RootView::new(window, cx)),
        )
        .expect("не удалось открыть тестовое окно")
    })
}

/// Есть ли видимая кнопка с таким доступным именем.
fn visible_button(window: &gpui_kit::Window, label: &str) -> bool {
    window
        .find_all_by_label(label)
        .iter()
        .any(|element| element.role() == Some(Role::Button) && element.visible())
}

#[gpui_kit::test]
fn действие_переключается_с_клавиатуры(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_kit::init(cx);
        cx.bind_keys([
            KeyBinding::new("down", NextAction, None),
            KeyBinding::new("up", PreviousAction, None),
        ]);
    });
    let (handle, view) = open_root_view(cx);

    cx.update_window(handle.into(), |_, window, cx| {
        view.update(cx, |view, cx| view.skip_loading(cx));
        window.render_frame(cx);

        // Изначально выбрано выключение — оно видно на кнопке-триггере.
        assert!(visible_button(window, "Выключить"));

        // Стрелка вниз переключает действие: кнопка показывает следующее.
        window.press("down", cx);
        window.render_frame(cx);
        assert!(visible_button(window, "Перезагрузить"));

        // И обратно — стрелка вверх.
        window.press("up", cx);
        window.render_frame(cx);
        assert!(visible_button(window, "Выключить"));
    })
    .unwrap();

    // Проверяем и результат на стороне приложения.
    cx.update(|cx| {
        assert_eq!(view.read(cx).selected_action(), PowerAction::Shutdown);
    });
}

#[gpui_kit::test]
fn меню_выбора_показывает_все_действия(cx: &mut TestAppContext) {
    cx.update(gpui_kit::init);
    let (handle, view) = open_root_view(cx);

    cx.update_window(handle.into(), |_, window, cx| {
        view.update(cx, |view, cx| view.skip_loading(cx));
        window.render_frame(cx);

        // Меню открывается по клику на триггер.
        window.click("action-select", cx);
        window.render_frame(cx);

        // В меню видны все четыре действия.
        for label in ["Выключить", "Перезагрузить", "Сон", "Гибернация"]
        {
            let item = window.within("popup-menu").find_by_label(label);
            assert_eq!(item.role(), Some(Role::MenuItem));
            assert!(item.visible());
        }
    })
    .unwrap();
}

#[gpui_kit::test]
fn пункт_меню_выбирает_режим_питания(cx: &mut TestAppContext) {
    cx.update(gpui_kit::init);
    let (handle, view) = open_root_view(cx);

    cx.update_window(handle.into(), |_, window, cx| {
        view.update(cx, |view, cx| view.skip_loading(cx));
        window.render_frame(cx);

        // Открываем меню и выбираем «Сон».
        window.click("action-select", cx);
        window.render_frame(cx);

        let item = window.within("popup-menu").find_by_label("Сон");
        let id = item.path().last().cloned().expect("у пункта меню есть id");
        window.click(id, cx);
        window.render_frame(cx);

        // Триггер показывает выбранный режим.
        assert!(visible_button(window, "Сон"));
    })
    .unwrap();

    cx.update(|cx| {
        assert_eq!(view.read(cx).selected_action(), PowerAction::Sleep);
    });
}

#[gpui_kit::test]
fn узкое_окно_не_выпускает_содержимое_за_края(
    cx: &mut TestAppContext,
) {
    cx.update(gpui_kit::init);
    let (handle, view) = open_root_view_sized(cx, 420., 660.);

    cx.update_window(handle.into(), |_, window, cx| {
        view.update(cx, |view, cx| view.skip_loading(cx));
        window.render_frame(cx);

        window.click("action-select", cx);
        window.render_frame(cx);

        let viewport = window.viewport_size();
        for label in ["Выключить", "Перезагрузить", "Сон", "Гибернация"]
        {
            let item = window.within("popup-menu").find_by_label(label);
            let bounds = item.bounds();
            assert!(
                bounds.origin.x >= px(0.)
                    && bounds.origin.y >= px(0.)
                    && bounds.origin.x + bounds.size.width <= viewport.width
                    && bounds.origin.y + bounds.size.height <= viewport.height,
                "пункт «{label}» выходит за окно: {:?} при размере {:?}",
                bounds,
                viewport
            );
        }
    })
    .unwrap();
}

#[gpui_kit::test]
fn необратимое_действие_просит_подтверждения(
    cx: &mut TestAppContext,
) {
    cx.update(gpui_kit::init);
    let (handle, view) = open_root_view(cx);

    cx.update_window(handle.into(), |_, window, cx| {
        view.update(cx, |view, cx| view.skip_loading(cx));
        window.render_frame(cx);

        // По умолчанию выбрано выключение — необратимое действие.
        assert!(visible_button(window, "Выключить"));

        // Кнопка питания открывает подтверждение, а не выполняет действие.
        window.click("power-commit", cx);
        window.render_frame(cx);

        let dialogs = window.find_all_by_role(Role::AlertDialog);
        assert_eq!(dialogs.len(), 1, "ожидалось одно окно подтверждения");

        // В диалоге есть отмена и действие с точным названием.
        assert!(window.within("dialog").find_by_label("Отмена").visible());
        assert!(window.within("dialog").find_by_label("Выключить").visible());
    })
    .unwrap();
}
