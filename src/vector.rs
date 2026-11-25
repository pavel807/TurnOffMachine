//! Векторный рендеринг SVG: точное соответствие растра размеру кадра.
//!
//! Размер растра считается ровно в device-пикселях того кадра, который элемент
//! занимает на экране. Никакой сетки округления: она давала расхождение сторон
//! до нескольких процентов, и картинка заметно тянулась. Пропорции внутри
//! растра держит сам SVG через `preserveAspectRatio`.
//!
//! Растеризация идёт в фоне. Пока новый кадр считается, на экран отдаётся
//! предыдущий готовый кадр той же картинки — ресайз и анимация не мигают.
//! Одновременно считается не больше одного кадра на картинку: запросы
//! склеиваются, поэтому перетаскивание окна не порождает лавину работы.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash as _, Hasher as _};
use std::sync::{Arc, Mutex, OnceLock};

use gpui_kit::{
    App, AppContext as _, DevicePixels, Hsla, ImageCacheError, ImageSource, ParsedSvg, RenderImage,
    SharedString, SvgSize, Window, size,
};

/// Сколько готовых растров держим.
const CACHE_CAPACITY: usize = 24;
/// Сколько разобранных SVG держим.
const PARSED_CAPACITY: usize = 192;
/// Предел стороны растра: дальше текстура не нужна и опасна для памяти.
const MAX_DEVICE_SIDE: u32 = 8192;

/// Ключ готового растра: картинка, её содержимое и точный кадр.
#[derive(Clone, PartialEq, Eq, Hash)]
struct ExactKey {
    logical: SharedString,
    content: u64,
    width: u32,
    height: u32,
}

struct RasterCache {
    order: VecDeque<ExactKey>,
    images: HashMap<ExactKey, Arc<RenderImage>>,
    /// Последний готовый кадр картинки — независимо от содержимого и размера.
    latest: HashMap<SharedString, ExactKey>,
    /// Что считается сейчас: не больше одной задачи на картинку.
    pending: HashSet<SharedString>,
    /// Содержимое, которое не удалось разобрать.
    failed: HashSet<u64>,
    parsed_order: VecDeque<u64>,
    parsed: HashMap<u64, Arc<ParsedSvg>>,
}

impl RasterCache {
    fn new() -> Self {
        Self {
            order: VecDeque::new(),
            images: HashMap::new(),
            latest: HashMap::new(),
            pending: HashSet::new(),
            failed: HashSet::new(),
            parsed_order: VecDeque::new(),
            parsed: HashMap::new(),
        }
    }

    fn get(&mut self, key: &ExactKey) -> Option<Arc<RenderImage>> {
        let image = self.images.get(key).cloned()?;
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.clone());
        Some(image)
    }

    fn stale(&self, logical: &SharedString) -> Option<Arc<RenderImage>> {
        let key = self.latest.get(logical)?;
        self.images.get(key).cloned()
    }

    fn insert(&mut self, key: ExactKey, image: Arc<RenderImage>) -> Vec<Arc<RenderImage>> {
        self.images.insert(key.clone(), image);
        self.order.push_back(key.clone());
        self.latest.insert(key.logical.clone(), key);

        let mut evicted = Vec::new();
        while self.order.len() > CACHE_CAPACITY {
            let Some(old) = self.order.pop_front() else {
                break;
            };
            if let Some(image) = self.images.remove(&old) {
                evicted.push(image);
            }
            if self.latest.get(&old.logical) == Some(&old) {
                self.latest.remove(&old.logical);
            }
        }
        evicted
    }

    fn parsed(&self, hash: u64) -> Option<Arc<ParsedSvg>> {
        self.parsed.get(&hash).cloned()
    }

    fn insert_parsed(&mut self, hash: u64, parsed: Arc<ParsedSvg>) {
        if self.parsed.insert(hash, parsed).is_none() {
            self.parsed_order.push_back(hash);
        }
        while self.parsed_order.len() > PARSED_CAPACITY {
            let Some(old) = self.parsed_order.pop_front() else {
                break;
            };
            self.parsed.remove(&old);
        }
    }
}

fn state() -> &'static Mutex<RasterCache> {
    static STATE: OnceLock<Mutex<RasterCache>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(RasterCache::new()))
}

/// Логический размер в device-пиксели — ровно, без сетки.
fn device(logical: f32, scale: f32) -> u32 {
    let value = (logical * scale).round();
    if !value.is_finite() {
        return 1;
    }
    value.clamp(1.0, MAX_DEVICE_SIDE as f32) as u32
}

fn hash_svg(svg: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    svg.hash(&mut hasher);
    hasher.finish()
}

/// Как SVG занимает кадр — уходит в `preserveAspectRatio`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fit {
    /// Растянуть, пропорции не сохраняются.
    Stretch,
    /// Вписать в квадрат, пропорции сохраняются.
    Square,
    /// Равномерно заполнить кадр с обрезкой, привязка к верху.
    Sky,
    /// Равномерно заполнить кадр с обрезкой, привязка к низу.
    Band,
}

impl Fit {
    fn aspect(self) -> &'static str {
        match self {
            Self::Stretch => "none",
            Self::Square => "xMidYMid meet",
            Self::Sky => "xMidYMin slice",
            Self::Band => "xMidYMax slice",
        }
    }
}

/// Векторная картинка под фактический размер кадра.
#[derive(Clone, Debug)]
pub struct VectorImage {
    logical: SharedString,
    svg: SharedString,
    content: u64,
    width: f32,
    height: f32,
}

impl VectorImage {
    fn build(
        id: impl Into<SharedString>,
        svg: impl Into<SharedString>,
        width: f32,
        height: f32,
        fit: Fit,
    ) -> Self {
        let svg: SharedString = preserve_aspect(&svg.into(), fit.aspect()).into();
        let content = hash_svg(&svg);
        Self {
            logical: id.into(),
            svg,
            content,
            width: width.max(1.0),
            height: height.max(1.0),
        }
    }

    /// Иконка или кольцо: квадратный кадр, пропорции сохраняются.
    pub fn square(id: impl Into<SharedString>, svg: impl Into<SharedString>, size: f32) -> Self {
        Self::build(id, svg, size, size, Fit::Square)
    }

    /// Небо: равномерный масштаб с обрезкой, привязка к верху.
    pub fn sky(id: impl Into<SharedString>, svg: impl Into<SharedString>, w: f32, h: f32) -> Self {
        Self::build(id, svg, w, h, Fit::Sky)
    }

    /// Полоса ландшафта: масштаб по ширине, привязка к низу.
    ///
    /// `aspect` — высота исходного SVG, делённая на его ширину. Кадр получает
    /// ровно эту пропорцию, поэтому горы не вытягиваются при любой форме окна.
    pub fn band(
        id: impl Into<SharedString>,
        svg: impl Into<SharedString>,
        width: f32,
        aspect: f32,
    ) -> Self {
        Self::build(id, svg, width, width * aspect, Fit::Band)
    }

    /// Растянуть без сохранения пропорций.
    pub fn stretch(
        id: impl Into<SharedString>,
        svg: impl Into<SharedString>,
        w: f32,
        h: f32,
    ) -> Self {
        Self::build(id, svg, w, h, Fit::Stretch)
    }

    /// Совместимость: квадратный кадр по ширине.
    pub fn new(id: impl Into<SharedString>, svg: impl Into<SharedString>, width: f32) -> Self {
        Self::square(id, svg, width)
    }

    /// Ширина кадра в логических пикселях — по ней ставится элемент.
    pub fn width(&self) -> f32 {
        self.width
    }

    /// Высота кадра в логических пикселях — по ней ставится элемент.
    pub fn height(&self) -> f32 {
        self.height
    }

    /// Источник для `img()`.
    pub fn source(&self) -> ImageSource {
        let logical = self.logical.clone();
        let svg = self.svg.clone();
        let content = self.content;
        let width = self.width;
        let height = self.height;

        ImageSource::Custom(Arc::new(move |window: &mut Window, app: &mut App| {
            let scale = window.scale_factor();
            let exact = ExactKey {
                logical: logical.clone(),
                content,
                width: device(width, scale),
                height: device(height, scale),
            };

            let Ok(mut cache) = state().lock() else {
                return None;
            };
            if let Some(image) = cache.get(&exact) {
                return Some(Ok(image));
            }

            let stale = cache.stale(&logical);
            if cache.pending.contains(&logical) || cache.failed.contains(&content) {
                return stale.map(Ok);
            }
            cache.pending.insert(logical.clone());
            let parsed = cache.parsed(content);
            drop(cache);

            let renderer = app.svg_renderer();
            let bytes = svg.clone();
            let target = SvgSize::ExactSize(size(
                DevicePixels(exact.width as i32),
                DevicePixels(exact.height as i32),
            ));
            let done_key = exact;
            let done_logical = logical.clone();

            app.spawn(async move |cx| {
                let outcome = cx
                    .background_spawn(async move {
                        let parsed = match parsed {
                            Some(parsed) => parsed,
                            None => Arc::new(
                                renderer
                                    .parse_svg(bytes.as_bytes())
                                    .map_err(|e| ImageCacheError::Usvg(Arc::new(e)))?,
                            ),
                        };
                        let image = renderer
                            .render_parsed(&parsed, target)
                            .map_err(|e| ImageCacheError::Usvg(Arc::new(e)))?;
                        Ok::<_, ImageCacheError>((parsed, image))
                    })
                    .await;

                cx.update(|app| {
                    let Ok(mut cache) = state().lock() else {
                        return;
                    };
                    cache.pending.remove(&done_logical);
                    match outcome {
                        Ok((parsed, image)) => {
                            cache.insert_parsed(content, parsed);
                            for evicted in cache.insert(done_key, image) {
                                app.drop_image(evicted, None);
                            }
                        }
                        Err(_) => {
                            cache.failed.insert(content);
                        }
                    }
                });
                cx.refresh();
            })
            .detach();

            stale.map(Ok)
        }))
    }
}

/// Задать `preserveAspectRatio`, заменив существующее значение.
pub fn preserve_aspect(svg: &str, value: &str) -> String {
    const NAME: &str = "preserveAspectRatio=\"";
    if let Some(start) = svg.find(NAME) {
        let value_start = start + NAME.len();
        if let Some(end) = svg[value_start..].find('"') {
            let mut out = String::with_capacity(svg.len() + value.len());
            out.push_str(&svg[..value_start]);
            out.push_str(value);
            out.push_str(&svg[value_start + end..]);
            return out;
        }
    }
    match svg.find("<svg") {
        Some(start) => {
            let at = start + "<svg".len();
            let mut out = String::with_capacity(svg.len() + 40);
            out.push_str(&svg[..at]);
            out.push_str(&format!(r#" preserveAspectRatio="{value}""#));
            out.push_str(&svg[at..]);
            out
        }
        None => svg.to_string(),
    }
}

/// Совместимость: растянуть на кадр произвольных пропорций.
pub fn stretch_to_frame(svg: &str) -> String {
    preserve_aspect(svg, "none")
}

/// Подставить цвета в SVG.
pub fn tint(svg: &str, main: Hsla, dark: Hsla) -> String {
    svg.replace("currentColorDark", &hex(dark))
        .replace("currentColor", &hex(main))
}

/// Цвет строкой `#rrggbb` либо `#rrggbbaa`.
pub fn hex(color: Hsla) -> String {
    let rgb = color.to_rgb();
    let channel = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    let (r, g, b) = (channel(rgb.r), channel(rgb.g), channel(rgb.b));
    if rgb.a >= 1.0 {
        format!("#{r:02x}{g:02x}{b:02x}")
    } else {
        format!("#{r:02x}{g:02x}{b:02x}{:02x}", channel(rgb.a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_size_is_exact_not_gridded() {
        assert_eq!(device(440.0, 2.0), 880);
        assert_eq!(device(441.0, 2.0), 882);
        assert_eq!(device(34.0, 1.0), 34);
        assert_eq!(device(0.0, 2.0), 1);
    }

    #[test]
    fn raster_keeps_the_element_aspect() {
        let scale = 2.0;
        let (w, h) = (440.0_f32, 680.0_f32);
        let rw = device(w, scale) as f32;
        let rh = device(h, scale) as f32;
        assert!((rw / rh - w / h).abs() < 0.002);
    }

    #[test]
    fn band_keeps_source_proportion() {
        let image = VectorImage::band("t", "<svg/>", 900., 1100. / 1200.);
        assert!((image.height() - 825.).abs() < 0.5);
        assert_eq!(image.width(), 900.);
    }

    #[test]
    fn replaces_existing_aspect_once() {
        let svg = r#"<svg preserveAspectRatio="none" viewBox="0 0 4 3"></svg>"#;
        let out = preserve_aspect(svg, "xMidYMax slice");
        assert!(out.contains("xMidYMax slice"));
        assert_eq!(out.matches("preserveAspectRatio").count(), 1);
    }

    #[test]
    fn stale_survives_content_and_size_change() {
        let mut cache = RasterCache::new();
        let first = ExactKey {
            logical: "ring".into(),
            content: 1,
            width: 64,
            height: 64,
        };
        cache.insert(first, Arc::new(RenderImage::new(Default::default())));
        let second = ExactKey {
            logical: "ring".into(),
            content: 2,
            width: 96,
            height: 96,
        };
        assert!(cache.get(&second).is_none());
        assert!(cache.stale(&"ring".into()).is_some());
    }

    #[test]
    fn one_pending_task_per_picture() {
        let mut cache = RasterCache::new();
        assert!(cache.pending.insert("sky".into()));
        assert!(!cache.pending.insert("sky".into()));
    }
}
