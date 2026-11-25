//! Проверка векторного рендеринга: SVG растеризуется под запрошенный размер.

use gpui_kit::{
    AppContext, Bounds, DevicePixels, ImageSource, IntoElement, Render, Size, TestAppContext,
    WindowBounds, WindowOptions, div, px, size,
};
use turnoff_gpui::vector::VectorImage;

/// Пустой экран — нужна только площадка для рендера.
struct Empty;

impl Render for Empty {
    fn render(
        &mut self,
        _window: &mut gpui_kit::Window,
        _cx: &mut gpui_kit::Context<Self>,
    ) -> impl IntoElement {
        div()
    }
}

#[gpui_kit::test]
fn svg_растеризуется_под_запрошенный_размер(
    cx: &mut TestAppContext,
) {
    cx.update(gpui_kit::init);

    let handle = cx.update(|cx| {
        gpui_kit::open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds::new(
                    Default::default(),
                    size(px(400.), px(400.)),
                ))),
                ..Default::default()
            },
            cx,
            |_window, cx| cx.new(|_| Empty),
        )
        .expect("не удалось открыть тестовое окно")
        .0
    });

    cx.update_window(handle.into(), |_, window, cx| {
        let image = VectorImage::new(
            "test",
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect width="24" height="24" fill="#ffffff"/></svg>"##,
            100.,
        );

        let ImageSource::Custom(load) = image.source() else {
            panic!("ожидался векторный источник изображения");
        };

        let rendered = load(window, cx)
            .expect("источник должен вернуть картинку")
            .expect("SVG должен растеризоваться");

        // Разрешение картинки совпадает с запрошенным размером в пикселях экрана.
        let device = window.scale_factor() as i32 * 100;
        assert_eq!(
            rendered.size(0),
            Size::new(DevicePixels(device), DevicePixels(device))
        );
    })
    .unwrap();
}

#[gpui_kit::test]
fn пустой_svg_не_ломает_рендер(cx: &mut TestAppContext) {
    cx.update(gpui_kit::init);

    let handle = cx.update(|cx| {
        gpui_kit::open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds::new(
                    Default::default(),
                    size(px(400.), px(400.)),
                ))),
                ..Default::default()
            },
            cx,
            |_window, cx| cx.new(|_| Empty),
        )
        .expect("не удалось открыть тестовое окно")
        .0
    });

    cx.update_window(handle.into(), |_, window, cx| {
        let image = VectorImage::new("broken", "<svg", 100.);
        let ImageSource::Custom(load) = image.source() else {
            panic!("ожидался векторный источник изображения");
        };

        // Битый SVG не должен падать: источник просто не отдаёт картинку.
        let rendered = load(window, cx);
        assert!(rendered.is_none() || rendered.is_some_and(|image| image.is_ok()));
    })
    .unwrap();
}
