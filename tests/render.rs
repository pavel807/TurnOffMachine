//! Служебный тест: раскладывает иконки приложения в PNG, чтобы их можно было
//! посмотреть глазами. Запускается вручную: `cargo test --test render -- --ignored`.

use resvg::tiny_skia::Pixmap;
use turnoff_gpui::assets::AppIcon;
use turnoff_gpui::palette::Palette;
use turnoff_gpui::scene::TimeOfDay;
use turnoff_gpui::vector;

/// Куда складывать результаты.
const OUT_DIR: &str = "/tmp/turnoff-icons";

#[test]
#[ignore = "рисует файлы для визуальной проверки"]
fn иконки_сохраняются_в_png() {
    std::fs::create_dir_all(OUT_DIR).expect("не удалось создать каталог");

    let palette = Palette::for_time_of_day(TimeOfDay::Night);

    for icon in AppIcon::ALL {
        let svg = String::from_utf8_lossy(&icon.svg()).to_string();
        let tinted = vector::tint(&svg, palette.accent, palette.accent_dark);
        let path = format!("{OUT_DIR}/{}.png", icon.asset_path().replace('/', "_"));
        render(&tinted, 192., &path);
    }
}

/// Растеризовать SVG в PNG.
fn render(svg: &str, size: f32, path: &str) {
    let options = resvg::usvg::Options::default();
    let tree = resvg::usvg::Tree::from_data(svg.as_bytes(), &options).expect("SVG не разобран");

    let mut pixmap = Pixmap::new(size as u32, size as u32).expect("не удалось создать буфер");
    let transform = resvg::tiny_skia::Transform::from_scale(
        size / tree.size().width(),
        size / tree.size().height(),
    );
    resvg::render(&tree, transform, &mut pixmap.as_mut());

    pixmap.save_png(path).expect("не удалось сохранить PNG");
}
