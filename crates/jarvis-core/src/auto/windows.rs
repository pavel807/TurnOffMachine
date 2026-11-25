//! Windows: UI Automation (UIA) через `xa11y-windows` — реализация контракта
//! auto-модуля без модели зрения (как macOS [`super::ax`], Linux
//! [`super::atspi_linux`]).
//!
//! Вместо скриншота и нейросети-зрения LLM читает дерево элементов активного
//! приложения текстом (`ui_read`) и действует по именам: клик по кнопке
//! «Сохранить», ввод в поле «Поиск». Каждый элемент несёт свою нативную
//! ссылку из capture — действие идёт точно в тот узел, который модель видела,
//! без повторного поиска по имени (узел stale → честная ошибка).
//!
//! Фолбэк: контролы без UIA-паттернов (кастомные отрисовки) кликаются по
//! координатам центра [`ElementData::bounds`] через [`crate::platform::desktop`]
//! (SendInput); координаты UIA — физические пиксели, как у SendInput, поэтому
//! конверсии не требуется.
//!
//! Отмена: голосовое «стоп» проверяется между шагами ([`super::is_cancelled`]).

use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use xa11y::capture::CaptureOptions;
use xa11y::{Element, ElementData, Role};

use super::elements::{self, Element as UiElement, ScreenSnapshot};

/// Роли, которые перечисляет `ui_read` на Windows: словарь ELEMENT_ROLES
/// плюс то, что UIA даёт сверх него (пункты меню, вкладки, элементы списков).
const WINDOWS_ROLES: &[&str] = &[
    "button",
    "checkbox",
    "radio button",
    "pop up button",
    "text field",
    "secure text field",
    "static text",
    "link",
    "slider",
    "menu bar item",
    "menu item",
    "tab",
    "list item",
    "tree item",
];

/// Бюджеты обхода: как в autocore::capture::default_options — достаточно,
/// чтобы увидеть экран, мало — чтобы не утонуть в плотном приложении.
fn capture_options() -> CaptureOptions {
    let mut o = CaptureOptions::default();
    o.max_depth = Some(12);
    o.max_nodes = 600;
    o.max_text_bytes = 4096;
    o.text = false;
    o.relations = false;
    o.formatting = false;
    o.timeout = Some(std::time::Duration::from_secs(2));
    o
}

/// Провайдер на каждый вызов: CoCreateInstance дёшев, а глобальный кэш
/// добавил бы COM-apartment вопросы между потоками движка.
fn provider() -> Result<Arc<dyn xa11y::Provider>> {
    Ok(Arc::new(xa11y_windows::WindowsProvider::new()?))
}

/// Роль xa11y → словарь Jarvis (как ax_role в macOS). Пустая строка — узел
/// пропускаем (контейнеры и мусор не перечисляем).
fn xa_role(role: Role, password: bool) -> &'static str {
    match role {
        Role::Button => "button",
        Role::CheckBox | Role::Switch => "checkbox",
        Role::RadioButton => "radio button",
        Role::ComboBox => "pop up button",
        Role::TextField | Role::TextArea | Role::SpinButton => {
            if password {
                "secure text field"
            } else {
                "text field"
            }
        }
        Role::StaticText | Role::Heading => "static text",
        Role::Link => "link",
        Role::Slider => "slider",
        Role::MenuBar => "menu bar item",
        Role::MenuItem => "menu item",
        Role::Tab => "tab",
        Role::ListItem => "list item",
        Role::TreeItem => "tree item",
        _ => "",
    }
}

fn ensure_not_cancelled() -> Result<()> {
    if crate::auto::is_cancelled() {
        return Err(anyhow!("автоматизация остановлена голосом"));
    }
    Ok(())
}

/// Сравнение имён приложений: нечёткое, в обе стороны («Блокнот» ↔ «notepad»).
fn app_matches(have: &str, want: &str) -> bool {
    let h = elements::normalize(have);
    let w = elements::normalize(want);
    !w.is_empty() && (h.contains(&w) || w.contains(&h))
}

/// Снимок экрана: нумерованные элементы + живые узлы UIA для действий.
struct Captured {
    snapshot: ScreenSnapshot,
    /// `live[i]` — нативный элемент UIA для элемента с `id == i`.
    live: Vec<Element>,
}

/// Выбрать приложение: указанное по имени, иначе активное (foreground),
/// иначе первое с окнами.
fn choose_app<'a>(
    p: &Arc<dyn xa11y::Provider>,
    apps: &'a [ElementData],
    app: Option<&str>,
) -> Result<&'a ElementData> {
    if let Some(want) = app {
        let want = want.trim();
        if let Some(found) = apps
            .iter()
            .find(|a| a.name.as_deref().is_some_and(|n| app_matches(n, want)))
        {
            return Ok(found);
        }
        let known: Vec<&str> = apps
            .iter()
            .filter_map(|a| a.name.as_deref())
            .take(10)
            .collect();
        bail!(
            "приложение «{want}» не найдено среди запущенных. Видны: {}",
            if known.is_empty() {
                "ничего с окнами".to_string()
            } else {
                known.join(", ")
            }
        );
    }
    // Активное: у UIA foreground-окно имеет active=true (docs xa11y: только
    // для окон). Иначе — первое приложение с окнами.
    if let Some(found) = apps.iter().find(|a| {
        p.app_windows(a)
            .map(|w| w.iter().any(|win| win.states.active))
            .unwrap_or(false)
    }) {
        return Ok(found);
    }
    apps.iter()
        .find(|a| p.app_windows(a).map(|w| !w.is_empty()).unwrap_or(false))
        .ok_or_else(|| anyhow!("не видно ни одного приложения с окнами"))
}

/// Собрать дерево элементов активного (или указанного) приложения.
fn collect(app: Option<&str>) -> Result<Captured> {
    let p = provider()?;
    let apps = p
        .list_apps()
        .map_err(|e| anyhow!("UIA: не перечислить приложения: {e:#}"))?;
    let chosen = choose_app(&p, &apps, app)?;
    let app_name = chosen.name.clone().unwrap_or_else(|| "неизвестно".into());
    let wins = p
        .app_windows(chosen)
        .map_err(|e| anyhow!("UIA: нет окон у «{app_name}»: {e:#}"))?;

    let opts = capture_options();
    let mut uels: Vec<UiElement> = Vec::new();
    let mut live: Vec<Element> = Vec::new();
    let mut window_title = String::new();

    for (widx, win_data) in wins.iter().take(3).enumerate() {
        let widx = widx + 1;
        if widx == 1 {
            window_title = win_data.name.clone().unwrap_or_default();
        }
        let root = Element::new(win_data.clone(), Arc::clone(&p));
        let capture = root
            .capture(&opts)
            .map_err(|e| anyhow!("UIA: не прочитать окно «{}»: {e:#}", window_title))?;
        for node in &capture.nodes {
            let d = node.element.data();
            let role = xa_role(d.role, d.states.password);
            if role.is_empty() {
                continue;
            }
            uels.push(UiElement {
                id: uels.len(),
                role: role.to_string(),
                raw_role: d.role.to_snake_case().to_string(),
                name: d.name.clone().unwrap_or_default(),
                description: d.description.clone().unwrap_or_default(),
                value: d.value.clone().unwrap_or_default(),
                depth: node.depth as u32 + 1,
                window: widx,
                app: app_name.clone(),
            });
            live.push(node.element.clone());
        }
    }

    Ok(Captured {
        snapshot: ScreenSnapshot::new(app_name, window_title, uels),
        live,
    })
}

/// Выбрать элемент из снимка: сначала точное имя, потом нечёткое (как pick
/// в macOS ax.rs). Возвращает индекс элемента в снимке.
fn pick(captured: &Captured, role: &str, name: &str, window: Option<&str>) -> Result<usize> {
    let in_window = |el: &UiElement| match window {
        Some(w) if !w.trim().is_empty() => el.window.to_string() == w.trim(),
        _ => true,
    };
    let mut candidates: Vec<&UiElement> = captured
        .snapshot
        .elements
        .iter()
        .filter(|el| el.role == role)
        .filter(|el| in_window(el))
        .collect();
    // Роль может прийти словом на русском — тогда роль элемента не совпадёт.
    if candidates.is_empty() {
        candidates = captured
            .snapshot
            .elements
            .iter()
            .filter(|el| in_window(el))
            .collect();
    }
    let want = elements::strip_role_words(name);
    let want = if want.trim().is_empty() {
        name
    } else {
        want.as_str()
    };
    let mut scored: Vec<(u32, usize)> = candidates
        .iter()
        .map(|el| (elements::score_element(el, want), el.id))
        .filter(|(score, _)| *score > 0)
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    match scored.first() {
        Some((score, id)) if *score >= 50 => Ok(*id),
        Some(_) | None => {
            // Не нашли — показываем, что есть, чтобы модель выбрала другое имя.
            let names: Vec<String> = candidates
                .iter()
                .take(8)
                .map(|el| format!("{} «{}»", el.role, el.label()))
                .collect();
            Err(anyhow!(
                "элемент «{name}» не найден. Похожее на экране: {}",
                if names.is_empty() {
                    "ничего похожего нет".to_string()
                } else {
                    names.join(", ")
                }
            ))
        }
    }
}

pub struct ScreenDump {
    /// Имя активного процесса (может быть пустым).
    pub app: String,
    /// Текстовый отчёт, собранный обходом дерева (уже человекочитаемый).
    pub report: String,
}

impl ScreenDump {
    /// Текст для пользователя и для модели.
    pub fn to_text(&self) -> String {
        if self.report.trim().is_empty() {
            format!(
                "Активное приложение: {}. Элементов не видно (окно может быть пустым или защищённым).",
                self.app
            )
        } else {
            self.report.trim().to_string()
        }
    }
}

/// Снимок экрана: активное приложение и элементы управления.
pub fn read_screen() -> Result<ScreenDump> {
    let captured = collect(None)?;
    Ok(ScreenDump {
        app: captured.snapshot.app.clone(),
        report: captured.snapshot.render(),
    })
}

/// Найти элемент по роли и имени (нечётко) и нажать его.
pub fn click_element(
    app: Option<&str>,
    role: &str,
    name: &str,
    window: Option<&str>,
) -> Result<String> {
    ensure_not_cancelled()?;
    let role = super::normalize_role(role);
    if !WINDOWS_ROLES.contains(&role.as_str()) {
        return Err(anyhow!(
            "роль «{role}» не поддерживается: используй одну из {}",
            WINDOWS_ROLES.join(", ")
        ));
    }
    let captured = collect(app)?;
    let id = pick(&captured, &role, name, window)?;
    let what = captured.snapshot.elements[id].clone();
    ensure_not_cancelled()?;
    let target = &captured.live[id];
    match target.press() {
        Ok(()) => Ok(format!("нажато: {} «{}»", what.role, what.label())),
        Err(e) => {
            // Кастомные контролы без InvokePattern — клик по координатам
            // центра: bounds UIA в физических пикселях, SendInput тоже.
            let center = target
                .data()
                .bounds
                .map(|r| (r.x + r.width as i32 / 2, r.y + r.height as i32 / 2));
            if let Some((x, y)) = center {
                let action = crate::platform::desktop::DesktopAction::Click {
                    x,
                    y,
                    button: crate::platform::desktop::MouseButton::Left,
                    count: 1,
                };
                if crate::platform::desktop::execute(&action).is_ok() {
                    return Ok(format!(
                        "нажато (клик по координатам {x}×{y}): {} «{}»",
                        what.role,
                        what.label()
                    ));
                }
            }
            Err(anyhow!("клик не удался: {e:#}"))
        }
    }
}

/// Вписать значение в текстовое поле (ValuePattern, при отказе — ввод
/// с клавиатуры после фокуса). Фактическое значение перечитает верификация
/// на уровне инструмента (tools.rs) свежим снимком экрана.
pub fn set_field_value(
    app: Option<&str>,
    role: &str,
    name: &str,
    text: &str,
    submit: bool,
    window: Option<&str>,
) -> Result<String> {
    ensure_not_cancelled()?;
    let role = super::normalize_role(role);
    if role != "text field" && role != "secure text field" {
        return Err(anyhow!(
            "вписывать текст можно только в поле, получена роль «{role}»"
        ));
    }
    let captured = collect(app)?;
    let id = pick(&captured, &role, name, window)?;
    let what = captured.snapshot.elements[id].clone();
    ensure_not_cancelled()?;
    let target = &captured.live[id];
    let _ = target.focus();
    match target.set_value(text) {
        Ok(()) => {}
        // Поле без ValuePattern (редакторы кода, web-поля): фокус + набор.
        Err(_) => {
            target
                .type_text(text)
                .map_err(|e| anyhow!("ввод в поле не удался: {e:#}"))?;
        }
    }
    if submit {
        let _ =
            crate::platform::desktop::execute(&crate::platform::desktop::DesktopAction::PressKey {
                key: "enter".into(),
            });
    }
    Ok(format!(
        "вписано в {} «{}»: «{text}»",
        what.role,
        what.label()
    ))
}

/// Выбрать пункт меню приложения: путь из 1–3 имён
/// (например `["Файл", "Экспортировать", "PDF"]`). Меню динамическое —
/// после каждого клика пересобираем снимок.
pub fn menu_path(app: Option<&str>, path: &[&str]) -> Result<String> {
    ensure_not_cancelled()?;
    if path.is_empty() || path.len() > 3 {
        bail!("путь меню — из 1–3 пунктов, получено {}", path.len());
    }
    for (i, seg) in path.iter().enumerate() {
        ensure_not_cancelled()?;
        let captured = collect(app)?;
        // Первый сегмент — верхнее меню, остальные — пункты/подменю; роли
        // пробуем обе: в UIA верхние пункты бывают и MenuItem, и MenuBarItem.
        let roles: &[&str] = if i == 0 {
            &["menu bar item", "menu item"]
        } else {
            &["menu item", "menu bar item"]
        };
        let mut chosen: Option<usize> = None;
        for role in roles {
            if let Ok(id) = pick(&captured, role, seg, None) {
                chosen = Some(id);
                break;
            }
        }
        let Some(id) = chosen else {
            let names: Vec<String> = captured
                .snapshot
                .elements
                .iter()
                .filter(|el| el.role.contains("menu"))
                .take(8)
                .map(|el| format!("{} «{}»", el.role, el.label()))
                .collect();
            bail!(
                "пункт меню «{seg}» не найден. В меню видно: {}",
                if names.is_empty() {
                    "ничего (возможно, меню приложения не в UIA — попробуй ui_click по экрану)"
                        .to_string()
                } else {
                    names.join(", ")
                }
            );
        };
        let what = captured.snapshot.elements[id].clone();
        let target = &captured.live[id];
        if let Err(e) = target.press() {
            bail!("пункт «{}» не нажался: {e:#}", what.label());
        }
        // Подменю открывается не мгновенно.
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    Ok(format!("меню пройдено: {}", path.join(" → ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xa_role_maps_dictionary_and_skips_containers() {
        assert_eq!(xa_role(Role::Button, false), "button");
        assert_eq!(xa_role(Role::ComboBox, false), "pop up button");
        assert_eq!(xa_role(Role::TextField, false), "text field");
        assert_eq!(xa_role(Role::TextField, true), "secure text field");
        assert_eq!(xa_role(Role::Switch, false), "checkbox");
        assert_eq!(xa_role(Role::MenuItem, false), "menu item");
        // Контейнеры не перечисляем.
        assert_eq!(xa_role(Role::Group, false), "");
        assert_eq!(xa_role(Role::Window, false), "");
        assert_eq!(xa_role(Role::Unknown, false), "");
    }

    #[test]
    fn app_matches_is_fuzzy_bidirectional() {
        assert!(app_matches("Блокнот", "блокнот"));
        assert!(app_matches("notepad - Заметки", "Notepad"));
        assert!(!app_matches("Калькулятор", "notepad"));
        assert!(!app_matches("Калькулятор", ""));
    }
}
