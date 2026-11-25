//! Корневой экран: круг запуска, раскрывающийся в интерфейс, и управление питанием.
//!
//! Запуск — одна непрерывная сцена: точка вырастает в круг с кольцом прогресса,
//! круг раскрывается на всё окно, и из него проявляется интерфейс.
//!
//! Компоновка знает четыре ступени плотности. При уменьшении окна уменьшаются
//! часы, кнопка и отступы, а лишние подписи убираются. Секции и их обёртки не
//! сжимаются, поэтому нижняя секция никогда не выдавливается за границу окна:
//! при нехватке высоты колонка растёт и прокручивается.
//!
//! Методы отрисовки берут `&self`: они только читают состояние, а меняют его
//! обработчики событий через `cx.listener`.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use gpui_kit::actions;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenuItem};
use gpui_kit::component::{
    ActiveTheme as _, Icon, IconName, Side, Sizable as _, Theme, ThemeColor, TitleBar,
    WindowExt as _, h_flex, v_flex,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    AnyElement, App, AppContext as _, BoxShadow, Context, FocusHandle, FontWeight,
    InteractiveElement as _, IntoElement, ObjectFit, ParentElement as _, Render, Role,
    SharedString, StatefulInteractiveElement as _, Styled as _, StyledImage as _, Task,
    TestSupportExt as _, Window, div, img, linear_color_stop, linear_gradient, point, px, relative,
    rems,
};

use crate::assets::AppIcon;
use crate::battery::PowerSource;
use crate::i18n::{self, LocalNow};
use crate::palette::{CLOCK_FONT_SIZE, POWER_BUTTON_SIZE, Palette, RING_SIZE};
use crate::power::{self, PowerAction};
use crate::scene::{SceneLayer, TimeOfDay};
use crate::vector::{self, VectorImage};

actions!(
    turnoff,
    [
        ToggleScene,
        CancelScheduledShutdown,
        NextAction,
        PreviousAction,
        CommitAction,
        Quit
    ]
);

/// Минимальный размер окна.
pub const MIN_WINDOW_WIDTH: f32 = 320.;
pub const MIN_WINDOW_HEIGHT: f32 = 420.;

/// Ширина выпадающего списка: определённая, иначе строки растут по содержимому.
const MENU_WIDTH: f32 = 284.;
/// Пропорция исходных SVG сцены: высота / ширина.
const SCENE_ASPECT: f32 = 1100. / 1200.;
/// Шаг кадровой анимации.
const FRAME: Duration = Duration::from_millis(8);

/// Плотность компоновки по высоте окна.
#[derive(Clone, Copy, Debug)]
struct Layout {
    /// Кегль крупных часов.
    clock: f32,
    /// Диаметр кнопки питания.
    power: f32,
    /// Размер иконки на кнопке.
    icon: f32,
    /// Отступ между секциями, в rems.
    gap: f32,
    /// Верхний отступ колонки, в rems.
    pad_top: f32,
    /// Нижний отступ колонки, в rems.
    pad_bottom: f32,
    /// Показывать пояснение под кнопкой.
    show_hint: bool,
    /// Показывать приветствие под часами.
    show_greeting: bool,
}

impl Layout {
    fn for_height(height: f32) -> Self {
        if height >= 660.0 {
            Self {
                clock: CLOCK_FONT_SIZE.as_f32(),
                power: POWER_BUTTON_SIZE.as_f32(),
                icon: 34.0,
                gap: 1.5,
                pad_top: 2.5,
                pad_bottom: 1.5,
                show_hint: true,
                show_greeting: true,
            }
        } else if height >= 560.0 {
            Self {
                clock: 54.0,
                power: 76.0,
                icon: 30.0,
                gap: 1.0,
                pad_top: 2.0,
                pad_bottom: 1.25,
                show_hint: true,
                show_greeting: true,
            }
        } else if height >= 480.0 {
            Self {
                clock: 44.0,
                power: 64.0,
                icon: 26.0,
                gap: 0.75,
                pad_top: 1.75,
                pad_bottom: 1.0,
                show_hint: false,
                show_greeting: true,
            }
        } else {
            Self {
                clock: 36.0,
                power: 56.0,
                icon: 22.0,
                gap: 0.5,
                pad_top: 1.5,
                pad_bottom: 0.875,
                show_hint: false,
                show_greeting: false,
            }
        }
    }
}

/// Стадия запуска.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    /// Круг вырастает из точки.
    Seed,
    /// Кольцо набирает прогресс.
    Charge,
    /// Круг раскрывается на окно, интерфейс проявляется.
    Bloom,
    /// Готово.
    Ready,
}

impl Stage {
    /// Показывать ли интерфейс под кругом.
    fn shows_app(self) -> bool {
        matches!(self, Self::Bloom | Self::Ready)
    }
}

/// Корневой экран: фоновая сцена и управление питанием.
pub struct RootView {
    /// Владеет фокусом окна и обрабатывает клавиатурные команды.
    focus_handle: FocusHandle,
    /// Фокус кнопки питания — Enter работает по ней.
    power_focus: FocusHandle,
    /// Текущее местное время.
    now: LocalNow,
    /// Период суток, задающий сцену и палитру.
    scene: TimeOfDay,
    /// Ручной выбор периода; `None` — сцена следует за временем.
    scene_override: Option<TimeOfDay>,
    /// Выбранное действие питания.
    action: PowerAction,
    /// Источник питания.
    power_source: PowerSource,
    /// Ошибка последнего действия; успех не показывается.
    error: Option<String>,
    /// Идёт ли выполнение действия.
    busy: bool,

    /// Стадия запуска.
    stage: Stage,
    /// Рост круга из точки.
    seed: f32,
    /// Прогресс загрузки в процентах.
    progress: u8,
    /// Раскрытие круга на окно.
    bloom: f32,
    /// Проявление интерфейса.
    reveal: f32,

    /// Наведение на кнопку питания.
    hover: f32,
    hover_target: f32,
    /// Пульс при смене действия.
    pulse: f32,
    /// Въезд сообщения об ошибке.
    error_in: f32,
    /// Растворение при смене палитры.
    veil: f32,
    veil_on: bool,
    micro_running: bool,

    _boot: Task<()>,
    _tick: Task<()>,
    _micro: Task<()>,
}

impl RootView {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let now = LocalNow::read();
        let scene = TimeOfDay::from_hour(now.hour);
        apply_scene_theme(scene, cx);

        let focus_handle = cx.focus_handle();
        let power_focus = cx.focus_handle().tab_stop(true);
        focus_handle.focus(window, cx);

        // Батарея читается сразу и затем раз в 30 секунд; часы — каждую секунду.
        let tick = cx.spawn(async move |this, cx| {
            loop {
                let source = cx
                    .background_spawn(async move { PowerSource::read() })
                    .await;
                let alive = this
                    .update(cx, |state, cx| {
                        state.now = LocalNow::read();
                        state.power_source = source;
                        state.sync_scene(cx);
                        if state.stage.shows_app() {
                            cx.notify();
                        }
                    })
                    .is_ok();
                if !alive {
                    return;
                }
                for _ in 0..30 {
                    cx.background_executor().timer(Duration::from_secs(1)).await;
                    let alive = this
                        .update(cx, |state, cx| {
                            state.now = LocalNow::read();
                            state.sync_scene(cx);
                            if state.stage.shows_app() {
                                cx.notify();
                            }
                        })
                        .is_ok();
                    if !alive {
                        return;
                    }
                }
            }
        });

        let boot = cx.spawn(async move |this, cx| {
            // 1. Круг вырастает из точки с лёгким перехлёстом.
            let mut t = 0.0f32;
            loop {
                cx.background_executor().timer(FRAME).await;
                t = (t + 0.008 / 0.44).min(1.0);
                let done = this
                    .update(cx, |state, cx| {
                        state.seed = ease_out_back(t);
                        cx.notify();
                        t >= 1.0
                    })
                    .unwrap_or(true);
                if done {
                    break;
                }
            }
            let _ = this.update(cx, |state, cx| {
                state.seed = 1.0;
                state.stage = Stage::Charge;
                cx.notify();
            });

            // 2. Кольцо набирает прогресс неравномерно.
            let mut seed: u32 = 0x9e37_79b9;
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(42))
                    .await;
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let step = 2 + ((seed >> 16) % 5) as u8;
                let done = this
                    .update(cx, |state, cx| {
                        state.progress = state.progress.saturating_add(step).min(100);
                        cx.notify();
                        state.progress >= 100
                    })
                    .unwrap_or(true);
                if done {
                    break;
                }
            }

            // Пауза на заполненном кольце.
            cx.background_executor()
                .timer(Duration::from_millis(280))
                .await;
            let _ = this.update(cx, |state, cx| {
                state.stage = Stage::Bloom;
                cx.notify();
            });

            // 3. Круг раскрывается на окно; интерфейс проявляется из него.
            let mut b = 0.0f32;
            loop {
                cx.background_executor().timer(FRAME).await;
                b = (b + 0.008 / 0.72).min(1.0);
                let done = this
                    .update(cx, |state, cx| {
                        state.bloom = ease_in_out_cubic(b);
                        state.reveal = ease_out_cubic(((b - 0.42) / 0.58).clamp(0.0, 1.0));
                        cx.notify();
                        b >= 1.0
                    })
                    .unwrap_or(true);
                if done {
                    break;
                }
            }
            let _ = this.update(cx, |state, cx| {
                state.stage = Stage::Ready;
                state.bloom = 1.0;
                state.reveal = 1.0;
                cx.notify();
            });
        });

        Self {
            focus_handle,
            power_focus,
            now,
            scene,
            scene_override: None,
            action: PowerAction::Shutdown,
            power_source: PowerSource::Unknown,
            error: None,
            busy: false,
            stage: Stage::Seed,
            seed: 0.0,
            progress: 0,
            bloom: 0.0,
            reveal: 0.0,
            hover: 0.0,
            hover_target: 0.0,
            pulse: 1.0,
            error_in: 0.0,
            veil: 0.0,
            veil_on: false,
            micro_running: false,
            _boot: boot,
            _tick: tick,
            _micro: Task::ready(()),
        }
    }

    /// Кадровый цикл микровзаимодействий; сам останавливается в покое.
    fn kick_micro(&mut self, cx: &mut Context<Self>) {
        if self.micro_running {
            return;
        }
        self.micro_running = true;
        self._micro = cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(FRAME).await;
                let active = this
                    .update(cx, |state, cx| {
                        let mut active = false;

                        let delta = state.hover_target - state.hover;
                        if delta.abs() > 0.002 {
                            state.hover += delta * 0.20;
                            active = true;
                        } else {
                            state.hover = state.hover_target;
                        }

                        if state.pulse < 1.0 {
                            state.pulse = (state.pulse + 0.075).min(1.0);
                            active = true;
                        }

                        let target = if state.error.is_some() { 1.0 } else { 0.0 };
                        let delta = target - state.error_in;
                        if delta.abs() > 0.002 {
                            state.error_in += delta * 0.18;
                            active = true;
                        } else {
                            state.error_in = target;
                        }

                        if state.veil_on {
                            state.veil += 0.055;
                            if state.veil >= 1.0 {
                                state.veil = 0.0;
                                state.veil_on = false;
                            }
                            active = true;
                        }

                        cx.notify();
                        active
                    })
                    .unwrap_or(false);
                if !active {
                    break;
                }
            }
            let _ = this.update(cx, |state, _| state.micro_running = false);
        });
    }

    /// Подстроить сцену под местное время, если период не выбран вручную.
    fn sync_scene(&mut self, cx: &mut Context<Self>) {
        if self.scene_override.is_some() {
            return;
        }
        let period = TimeOfDay::from_hour(self.now.hour);
        if period != self.scene {
            self.scene = period;
            apply_scene_theme(period, cx);
            self.start_veil(cx);
        }
    }

    /// Мягкое растворение при смене палитры.
    fn start_veil(&mut self, cx: &mut Context<Self>) {
        self.veil = 0.0;
        self.veil_on = true;
        self.kick_micro(cx);
    }

    /// Выбранное в данный момент действие питания.
    pub fn selected_action(&self) -> PowerAction {
        self.action
    }

    /// Пропустить загрузку — для тестов.
    ///
    /// Анимацию отменяем: иначе она продолжит писать `reveal` и вернёт экран
    /// к прозрачному состоянию после того, как тест его показал.
    pub fn skip_loading(&mut self, cx: &mut Context<Self>) {
        self._boot = Task::ready(());
        self.stage = Stage::Ready;
        self.seed = 1.0;
        self.progress = 100;
        self.bloom = 1.0;
        self.reveal = 1.0;
        cx.notify();
    }

    /// Переключить период суток вручную.
    fn cycle_scene(&mut self, cx: &mut Context<Self>) {
        let next = self.scene.next();
        self.scene = next;
        self.scene_override = Some(next);
        apply_scene_theme(next, cx);
        self.start_veil(cx);
        cx.notify();
    }

    /// Выбрать действие питания.
    fn select_action(&mut self, action: PowerAction, cx: &mut Context<Self>) {
        if self.action != action {
            self.pulse = 0.0;
            self.kick_micro(cx);
        }
        self.action = action;
        self.set_error(None, cx);
        cx.notify();
    }

    /// Шаг по кругу действий, пропуская недоступные на этой платформе.
    fn step_action(&mut self, step: isize, cx: &mut Context<Self>) {
        let all = PowerAction::ALL;
        let len = all.len() as isize;
        let mut index = all.iter().position(|a| *a == self.action).unwrap_or(0) as isize;
        for _ in 0..len {
            index = (index + step).rem_euclid(len);
            if all[index as usize].is_supported() {
                break;
            }
        }
        self.select_action(all[index as usize], cx);
    }

    /// Показать или убрать сообщение об ошибке.
    fn set_error(&mut self, message: Option<String>, cx: &mut Context<Self>) {
        if self.error != message {
            self.error = message;
            self.kick_micro(cx);
        }
    }

    /// Выполнить действие питания.
    fn run_action(&mut self, action: PowerAction, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        self.busy = true;
        self.set_error(None, cx);
        cx.notify();

        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move { power::execute(action) })
                .await;
            let _ = this.update(cx, |state, cx| {
                state.busy = false;
                // Успех виден по самой системе — показываем только ошибку.
                state.set_error(result.err(), cx);
                cx.notify();
            });
        })
        .detach();
    }

    /// Запросить действие, при необходимости показав подтверждение.
    fn request_action(&mut self, action: PowerAction, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(reason) = action.unsupported_reason() {
            self.set_error(Some(reason.to_string()), cx);
            cx.notify();
            return;
        }
        if self.busy {
            return;
        }
        if !action.needs_confirmation() {
            self.run_action(action, cx);
            return;
        }

        let this = cx.weak_entity();
        let restore = self.power_focus.clone();
        window.open_alert_dialog(cx, move |alert, _, _| {
            let this = this.clone();
            let ok_focus = restore.clone();
            let cancel_focus = restore.clone();
            alert
                .title(action.confirm_title())
                .description(action.confirm_description())
                .ok_text(action.label())
                .cancel_text("Отмена")
                .show_cancel(true)
                .centered()
                .on_ok(move |_, window, cx| {
                    let _ = this.update(cx, |state, cx| state.run_action(action, cx));
                    ok_focus.focus(window, cx);
                    true
                })
                .on_cancel(move |_, window, cx| {
                    cancel_focus.focus(window, cx);
                    true
                })
        });
    }

    /// Отменить отложенное системой выключение.
    fn cancel_scheduled(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move { power::cancel_pending_shutdown() })
                .await;
            let _ = this.update(cx, |state, cx| {
                state.set_error(result.err(), cx);
                cx.notify();
            });
        })
        .detach();
    }

    /// Корневая обёртка: команды, фокус и скругление окна.
    fn root(&self, window: &mut Window, cx: &mut Context<Self>, content: AnyElement) -> AnyElement {
        let palette = Palette::for_time_of_day(self.scene);
        let radii = gpui_kit::component::window_corner_radii(window, cx);

        div()
            .id("turnoff-root")
            .key_context("TurnOff")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|this, _: &ToggleScene, _, cx| this.cycle_scene(cx)))
            .on_action(
                cx.listener(|this, _: &CancelScheduledShutdown, _, cx| this.cancel_scheduled(cx)),
            )
            .on_action(cx.listener(|this, _: &NextAction, _, cx| this.step_action(1, cx)))
            .on_action(cx.listener(|this, _: &PreviousAction, _, cx| this.step_action(-1, cx)))
            .on_action(cx.listener(|this, _: &CommitAction, window, cx| {
                let action = this.action;
                this.request_action(action, window, cx);
            }))
            .on_action(|_: &Quit, _, cx| cx.quit())
            .size_full()
            .relative()
            .overflow_hidden()
            .rounded_tl(radii.top_left)
            .rounded_tr(radii.top_right)
            .rounded_bl(radii.bottom_left)
            .rounded_br(radii.bottom_right)
            .bg(palette.background)
            .child(content)
            .into_any_element()
    }

    // ── Сцена ────────────────────────────────────────────────────────────────

    /// Фоновая сцена: небо заполняет окно, ландшафт прижат к низу.
    fn render_scene(&self, window: &mut Window) -> AnyElement {
        let palette = Palette::for_time_of_day(self.scene);
        let viewport = window.viewport_size();
        let width = viewport.width.as_f32().max(1.);
        let height = viewport.height.as_f32().max(1.);
        let fade = (height * 0.26).clamp(80., 220.);

        div()
            .absolute()
            .inset_0()
            .overflow_hidden()
            .opacity(self.reveal)
            .child(
                div().absolute().inset_0().children(
                    self.scene
                        .layers()
                        .into_iter()
                        .map(|(layer, path)| self.render_scene_layer(layer, path, width, height)),
                ),
            )
            .child(
                div()
                    .absolute()
                    .top_0()
                    .left_0()
                    .right_0()
                    .h(px(132.))
                    .bg(linear_gradient(
                        180.,
                        linear_color_stop(palette.scrim, 0.).opacity(0.55),
                        linear_color_stop(palette.scrim, 1.).opacity(0.),
                    )),
            )
            .child(
                div()
                    .absolute()
                    .bottom_0()
                    .left_0()
                    .right_0()
                    .h(px(fade))
                    .bg(linear_gradient(
                        180.,
                        linear_color_stop(palette.scrim, 0.).opacity(0.),
                        linear_color_stop(palette.scrim, 1.).opacity(0.92),
                    )),
            )
            .into_any_element()
    }

    /// Слой сцены: кадр элемента и растр считаются из одних и тех же чисел,
    /// поэтому картинка не тянется и не меняет размер при смене масштаба.
    fn render_scene_layer(
        &self,
        layer: SceneLayer,
        path: &'static str,
        width: f32,
        height: f32,
    ) -> AnyElement {
        let Some(source) = scene_source(path) else {
            return div().absolute().inset_0().into_any_element();
        };
        let id = format!("scene:{layer:?}");

        if layer == SceneLayer::Sky {
            let image = VectorImage::sky(id, source, width, height);
            return img(image.source())
                .absolute()
                .top_0()
                .left_0()
                .w(px(image.width()))
                .h(px(image.height()))
                .object_fit(ObjectFit::Fill)
                .into_any_element();
        }

        let image = VectorImage::band(id, source, width, SCENE_ASPECT);
        img(image.source())
            .absolute()
            .bottom_0()
            .left_0()
            .w(px(image.width()))
            .h(px(image.height()))
            .object_fit(ObjectFit::Fill)
            .into_any_element()
    }

    // ── Круг запуска ─────────────────────────────────────────────────────────

    /// Круг, который раскрывается в окно.
    ///
    /// Оболочка растёт от точки до диагонали окна, скругление уходит от круга
    /// к прямоугольнику, стекло гаснет — и сквозь него проступает интерфейс.
    fn render_seed(&self, window: &mut Window) -> AnyElement {
        let palette = Palette::for_time_of_day(self.scene);
        let viewport = window.viewport_size();
        let w = viewport.width.as_f32().max(1.);
        let h = viewport.height.as_f32().max(1.);
        // Круг всегда помещается в окно, каким бы узким оно ни стало.
        let ring = RING_SIZE.as_f32().min((w.min(h) - 48.0).max(120.0));
        let bloom = self.bloom;

        let grown = ring * self.seed.max(0.0);
        let size = if self.stage == Stage::Bloom {
            let diagonal = (w * w + h * h).sqrt() * 1.2;
            grown + (diagonal - grown) * bloom
        } else {
            grown
        };
        let radius = size * 0.5 * (1.0 - bloom);
        let shell_opacity = 1.0 - smoothstep(0.34, 0.94, bloom);
        let inner = if self.stage == Stage::Bloom {
            ring
        } else {
            grown
        };
        let inner_opacity = smoothstep(0.55, 1.0, self.seed) * (1.0 - smoothstep(0.0, 0.30, bloom));
        let glow = 0.10 + 0.26 * (self.progress as f32 / 100.0);
        let compact = ring < 210.0;
        let power_size = if compact { 26.0 } else { 34.0 };

        let ring_image = VectorImage::square(
            format!("boot-ring:{}", vector::hex(palette.accent)),
            loading_ring(self.progress, &palette),
            ring,
        );
        let power = icon_image(AppIcon::Power, power_size, &palette);

        div()
            .absolute()
            .inset_0()
            .overflow_hidden()
            .child(
                div()
                    .absolute()
                    .left(px((w - size * 1.22) * 0.5))
                    .top(px((h - size * 1.22) * 0.5))
                    .w(px(size * 1.22))
                    .h(px(size * 1.22))
                    .rounded(px(size * 0.61))
                    .opacity(shell_opacity * (1.0 - bloom))
                    .bg(linear_gradient(
                        180.,
                        linear_color_stop(palette.accent, 0.).opacity(glow),
                        linear_color_stop(palette.accent, 1.).opacity(0.),
                    )),
            )
            .child(
                div()
                    .absolute()
                    .left(px((w - size) * 0.5))
                    .top(px((h - size) * 0.5))
                    .w(px(size))
                    .h(px(size))
                    .rounded(px(radius))
                    .bg(palette.glass)
                    .opacity(shell_opacity)
                    .border_1()
                    .border_color(palette.hairline)
                    .shadow(vec![
                        BoxShadow {
                            color: gpui_kit::black().opacity(0.55),
                            offset: point(px(0.), px(20.)),
                            blur_radius: px(56.),
                            spread_radius: px(-12.),
                            inset: false,
                        },
                        BoxShadow {
                            color: palette.on_scene.opacity(0.10),
                            offset: point(px(0.), px(1.)),
                            blur_radius: px(0.),
                            spread_radius: px(0.),
                            inset: true,
                        },
                    ]),
            )
            .child(
                div()
                    .absolute()
                    .left(px((w - inner) * 0.5))
                    .top(px((h - inner) * 0.5))
                    .w(px(inner))
                    .h(px(inner))
                    .opacity(inner_opacity)
                    .child(
                        div().absolute().inset_0().child(
                            img(ring_image.source())
                                .size_full()
                                .object_fit(ObjectFit::Fill),
                        ),
                    )
                    .child(
                        v_flex()
                            .absolute()
                            .inset_0()
                            .items_center()
                            .justify_center()
                            .gap(rems(if compact { 0.25 } else { 0.5 }))
                            .child(img(power.source()).size(px(power_size)))
                            .child(
                                div()
                                    .min_w(rems(4.5))
                                    .text_center()
                                    .text_size(rems(if compact { 2.0 } else { 2.75 }))
                                    .line_height(relative(1.))
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(palette.on_scene)
                                    .child(format!("{}%", self.progress.min(100))),
                            )
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(palette.on_scene)
                                    .child(i18n::greeting(self.scene)),
                            )
                            .when(!compact, |this| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .text_color(palette.on_scene_muted)
                                        .child("загрузка…"),
                                )
                            }),
                    ),
            )
            .into_any_element()
    }

    // ── Интерфейс ────────────────────────────────────────────────────────────

    fn render_title_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let palette = Palette::for_time_of_day(self.scene);
        let brand = icon_image(AppIcon::Power, 15., &palette);
        let scene_icon = icon_image(self.scene.icon(), 14., &palette);

        TitleBar::new()
            .bg(palette.background.opacity(0.))
            .border_color(palette.background.opacity(0.))
            .child(
                h_flex()
                    .items_center()
                    .gap(rems(0.5))
                    .min_w_0()
                    .child(img(brand.source()).size(rems(1.)).flex_shrink_0())
                    .child(
                        div()
                            .text_xs()
                            .font_weight(FontWeight::BOLD)
                            .text_color(palette.on_scene)
                            .truncate()
                            .child("TURN OFF MACHINE"),
                    ),
            )
            .child(
                div()
                    .id("toggle-scene")
                    .test_support()
                    .flex_shrink_0()
                    .child(
                        Button::new("toggle-scene-trigger")
                            .ghost()
                            .small()
                            .child(img(scene_icon.source()).size(rems(0.875)))
                            .accessibility_label("Сменить время суток")
                            .tooltip("Сменить время суток")
                            .on_click(cx.listener(|this, _, _, cx| this.cycle_scene(cx))),
                    ),
            )
    }

    /// Прокручиваемая колонка секций.
    ///
    /// Колонка и обёртки секций не сжимаются, поэтому при нехватке высоты
    /// колонка растёт и прокручивается, а низ не уходит за границу окна.
    fn render_content(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let layout = Layout::for_height(window.viewport_size().height.as_f32());

        div()
            .id("content-scroll")
            .relative()
            .size_full()
            .min_h_0()
            .overflow_y_scroll()
            .child(
                v_flex()
                    .w_full()
                    .min_h_full()
                    .flex_shrink_0()
                    .items_center()
                    .justify_between()
                    .gap(rems(layout.gap))
                    .px(rems(1.25))
                    .pt(rems(layout.pad_top))
                    .pb(rems(layout.pad_bottom))
                    .child(self.staggered(0, self.render_clock(layout)))
                    .child(self.staggered(1, self.render_power(layout, cx)))
                    .child(self.staggered(2, self.render_footer(layout, cx))),
            )
    }

    /// Секция входит сверху вниз: смещение только вверх, поэтому за нижнюю
    /// границу ничего не выходит даже в середине анимации.
    fn staggered(&self, index: usize, content: impl IntoElement) -> AnyElement {
        let start = index as f32 * 0.16;
        let local = ((self.reveal - start) / (1.0 - start)).clamp(0.0, 1.0);
        let p = ease_out_cubic(local);
        div()
            .w_full()
            .flex_shrink_0()
            .relative()
            .top(px((p - 1.0) * 14.0))
            .opacity(p)
            .child(content)
            .into_any_element()
    }

    fn render_clock(&self, layout: Layout) -> impl IntoElement {
        let palette = Palette::for_time_of_day(self.scene);

        v_flex()
            .items_center()
            .gap(rems(0.375))
            .w_full()
            .min_w_0()
            .flex_shrink_0()
            .child(
                div()
                    .max_w_full()
                    .text_xs()
                    .text_center()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(palette.on_scene_muted)
                    .child(i18n::long_date(&self.now)),
            )
            .child(
                h_flex()
                    .items_baseline()
                    .justify_center()
                    .gap(rems(0.375))
                    .max_w_full()
                    .min_w_0()
                    .child(
                        div()
                            .flex_shrink_0()
                            .text_size(px(layout.clock))
                            .line_height(relative(1.))
                            .font_weight(FontWeight::LIGHT)
                            .text_color(palette.on_scene)
                            .child(self.now.clock()),
                    )
                    .child(
                        div()
                            .flex_shrink_0()
                            .min_w(rems(1.75))
                            .text_size(rems(1.25))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(palette.on_scene_muted)
                            .child(self.now.seconds()),
                    ),
            )
            .when(layout.show_greeting, |this| {
                this.child(
                    div()
                        .max_w_full()
                        .text_xs()
                        .text_center()
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(palette.on_scene_muted)
                        .child(i18n::greeting(self.scene)),
                )
            })
    }

    /// Круглая кнопка питания и выбор действия под ней.
    fn render_power(&self, layout: Layout, cx: &mut Context<Self>) -> impl IntoElement {
        let palette = Palette::for_time_of_day(self.scene);
        let hover = self.hover;
        let pulse = ease_out_cubic(self.pulse);
        let busy = self.busy;

        let body = layout.power * (1.0 + 0.045 * hover);
        let radius = body * 0.5;
        let halo = body * 1.22 + 10.0 * hover;
        let area = layout.power + 40.0;
        let icon_size = layout.icon * (0.88 + 0.12 * pulse) * (1.0 + 0.05 * hover);
        let icon = icon_image(self.action.icon(), icon_size, &palette);

        v_flex()
            .items_center()
            .gap(rems(0.875))
            .w_full()
            .min_w_0()
            .flex_shrink_0()
            .child(
                div().id("power-commit").test_support().child(
                    div()
                        .id("power-button-area")
                        .on_hover(cx.listener(|this, hovered, _, cx| {
                            let target = if *hovered { 1.0 } else { 0.0 };
                            if this.hover_target != target {
                                this.hover_target = target;
                                this.kick_micro(cx);
                            }
                        }))
                        .relative()
                        .flex()
                        .items_center()
                        .justify_center()
                        .w(px(area))
                        .h(px(area))
                        .child(
                            div()
                                .absolute()
                                .w(px(halo))
                                .h(px(halo))
                                .rounded(px(halo * 0.5))
                                .bg(linear_gradient(
                                    180.,
                                    linear_color_stop(palette.accent, 0.)
                                        .opacity(0.14 + 0.18 * hover),
                                    linear_color_stop(palette.accent, 1.).opacity(0.),
                                ))
                                .shadow(vec![BoxShadow {
                                    color: palette.accent.opacity(0.22 + 0.30 * hover),
                                    offset: point(px(0.), px(0.)),
                                    blur_radius: px(34. + 14. * hover),
                                    spread_radius: px(3.),
                                    inset: false,
                                }]),
                        )
                        .child(
                            div()
                                .id("power-button")
                                .track_focus(&self.power_focus)
                                .focus_visible(|style| {
                                    style.border_color(palette.accent).border_2()
                                })
                                .relative()
                                .w(px(body))
                                .h(px(body))
                                .rounded(px(radius))
                                .bg(linear_gradient(
                                    165.,
                                    linear_color_stop(palette.accent, 0.)
                                        .opacity(0.22 + 0.14 * hover),
                                    linear_color_stop(palette.background, 1.).opacity(0.94),
                                ))
                                .border_1()
                                .border_color(palette.accent.opacity(0.40 + 0.36 * hover))
                                .shadow(vec![
                                    BoxShadow {
                                        color: gpui_kit::black().opacity(0.42),
                                        offset: point(px(0.), px(7. + 3. * hover)),
                                        blur_radius: px(20. + 8. * hover),
                                        spread_radius: px(0.),
                                        inset: false,
                                    },
                                    BoxShadow {
                                        color: palette.on_scene.opacity(0.18 + 0.10 * hover),
                                        offset: point(px(0.), px(-1.)),
                                        blur_radius: px(2.),
                                        spread_radius: px(0.),
                                        inset: true,
                                    },
                                ])
                                .flex()
                                .items_center()
                                .justify_center()
                                .role(Role::Button)
                                .aria_label(format!(
                                    "{} — {}",
                                    self.action.label(),
                                    self.action.hint()
                                ))
                                .opacity(if busy { 0.55 } else { 1.0 })
                                .cursor_pointer()
                                .when(busy, |button| button.cursor_not_allowed())
                                .on_click(cx.listener(|this, _, window, cx| {
                                    let action = this.action;
                                    this.request_action(action, window, cx);
                                }))
                                .child(
                                    img(icon.source())
                                        .size(px(icon_size))
                                        .opacity(0.4 + 0.6 * pulse),
                                ),
                        ),
                ),
            )
            .when(layout.show_hint, |this| {
                this.child(
                    div()
                        .max_w_full()
                        .px(rems(0.5))
                        .text_xs()
                        .text_center()
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(palette.on_scene_muted)
                        .child(if busy {
                            "Выполняется…".to_string()
                        } else {
                            format!("{} — {}", self.action.label(), self.action.hint())
                        }),
                )
            })
            .child(self.hairline(&palette))
            .child(self.render_action_select(cx))
    }

    /// Волосяная линия, растворяющаяся к краям.
    fn hairline(&self, palette: &Palette) -> AnyElement {
        h_flex()
            .w_full()
            .max_w(rems(19.))
            .h(px(1.))
            .child(div().flex_1().h_full().bg(linear_gradient(
                90.,
                linear_color_stop(palette.on_scene, 0.).opacity(0.),
                linear_color_stop(palette.on_scene, 1.).opacity(0.18),
            )))
            .child(div().flex_1().h_full().bg(linear_gradient(
                90.,
                linear_color_stop(palette.on_scene, 0.).opacity(0.18),
                linear_color_stop(palette.on_scene, 1.).opacity(0.),
            )))
            .into_any_element()
    }

    /// Кнопка выбора действия: стекло, подкрашенное акцентом сцены.
    fn render_action_select(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let palette = Palette::for_time_of_day(self.scene);
        let selected = self.action;
        let this = cx.weak_entity();
        let trigger_icon = icon_image(self.action.icon(), 16., &palette);
        let hover_border = palette.accent.opacity(0.55);

        div().id("action-select").test_support().child(
            Button::new("action-select-trigger")
                .h(rems(2.375))
                .px(rems(1.))
                .rounded(px(19.))
                .bg(palette.tint)
                .border_1()
                .border_color(palette.accent.opacity(0.30))
                .hover(move |style| style.border_color(hover_border))
                .child(
                    h_flex()
                        .items_center()
                        .gap(rems(0.5))
                        .min_w_0()
                        .child(img(trigger_icon.source()).size(rems(1.)).flex_shrink_0())
                        .child(
                            div()
                                .text_xs()
                                .font_weight(FontWeight::BOLD)
                                .text_color(palette.on_scene)
                                .truncate()
                                .child(self.action.label()),
                        ),
                )
                .accessibility_label(format!("Действие: {}", self.action.label()))
                .dropdown_caret(true)
                .dropdown_menu(move |popup, _window, _cx| {
                    let mut popup = popup
                        .min_w(px(MENU_WIDTH))
                        .max_w(px(MENU_WIDTH))
                        .check_side(Side::Right)
                        .label("Действие с компьютером");

                    for action in PowerAction::ALL {
                        let this = this.clone();
                        let is_selected = action == selected;
                        let supported = action.is_supported();
                        let caption = action
                            .unsupported_reason()
                            .unwrap_or_else(|| action.caption());
                        let row_palette = palette;
                        let row_icon = icon_image(action.icon(), 18., &row_palette);

                        popup = popup.item(
                            PopupMenuItem::element(move |_window, _cx| {
                                h_flex()
                                    .w_full()
                                    .min_w_0()
                                    .overflow_hidden()
                                    .items_center()
                                    .gap(rems(0.625))
                                    .opacity(if supported { 1.0 } else { 0.45 })
                                    .child(
                                        div()
                                            .flex()
                                            .items_center()
                                            .justify_center()
                                            .size(rems(1.875))
                                            .flex_shrink_0()
                                            .rounded(px(9.))
                                            .bg(row_palette.accent.opacity(if is_selected {
                                                0.26
                                            } else {
                                                0.12
                                            }))
                                            .child(img(row_icon.source()).size(rems(1.125))),
                                    )
                                    .child(
                                        v_flex()
                                            .flex_1()
                                            .min_w_0()
                                            .overflow_hidden()
                                            .gap(rems(0.0625))
                                            .child(
                                                div()
                                                    .text_sm()
                                                    .font_weight(if is_selected {
                                                        FontWeight::BOLD
                                                    } else {
                                                        FontWeight::MEDIUM
                                                    })
                                                    .text_color(row_palette.on_scene)
                                                    .truncate()
                                                    .child(action.label()),
                                            )
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .text_color(row_palette.on_scene_muted)
                                                    .truncate()
                                                    .child(caption),
                                            ),
                                    )
                            })
                            .checked(is_selected)
                            .disabled(!supported)
                            .on_click(move |_, _, cx| {
                                let _ =
                                    this.update(cx, |state, cx| state.select_action(action, cx));
                            }),
                        );
                    }
                    popup
                }),
        )
    }

    /// Нижняя часть: питание, ошибка и отмена отложенного выключения.
    fn render_footer(&self, layout: Layout, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .items_center()
            .gap(rems(layout.gap.min(0.75)))
            .w_full()
            .min_w_0()
            .flex_shrink_0()
            .child(self.render_power_chip(cx))
            .child(self.render_error(cx))
            .child(
                div().id("cancel-scheduled").test_support().child(
                    Button::new("cancel-scheduled-trigger")
                        .ghost()
                        .small()
                        .icon(IconName::CircleX)
                        .label("Отменить выключение")
                        .tooltip("Снять отложенное системой выключение")
                        .on_click(cx.listener(|this, _, _, cx| this.cancel_scheduled(cx))),
                ),
            )
    }

    /// Чип питания: батарея, сеть или честное «неизвестно».
    fn render_power_chip(&self, cx: &mut Context<Self>) -> AnyElement {
        let palette = Palette::for_time_of_day(self.scene);
        let radius = cx.theme().radius;

        let shell = div()
            .flex_col()
            .items_center()
            .gap(rems(0.25))
            .min_w(rems(6.5))
            .max_w_full()
            .px(rems(1.))
            .py(rems(0.5))
            .rounded(radius)
            .bg(palette.glass)
            .border_1()
            .border_color(palette.hairline)
            .shadow(vec![BoxShadow {
                color: palette.on_scene.opacity(0.08),
                offset: point(px(0.), px(-1.)),
                blur_radius: px(0.),
                spread_radius: px(0.),
                inset: true,
            }]);

        match self.power_source {
            PowerSource::Battery(info) => shell
                .child(
                    h_flex()
                        .items_center()
                        .gap(rems(0.375))
                        .child(
                            Icon::new(info.icon())
                                .with_size(px(13.))
                                .text_color(palette.accent),
                        )
                        .child(
                            div()
                                .text_xs()
                                .font_weight(FontWeight::BOLD)
                                .text_color(palette.on_scene)
                                .child(format!("{}%", info.percent)),
                        ),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(palette.on_scene_muted)
                        .child(info.status()),
                )
                .into_any_element(),
            PowerSource::Mains => shell
                .child(
                    h_flex()
                        .items_center()
                        .gap(rems(0.375))
                        .child(
                            Icon::new(IconName::BatteryCharging)
                                .with_size(px(13.))
                                .text_color(palette.accent),
                        )
                        .child(
                            div()
                                .text_xs()
                                .font_weight(FontWeight::BOLD)
                                .text_color(palette.on_scene)
                                .child("Сеть"),
                        ),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(palette.on_scene_muted)
                        .child("питание от сети"),
                )
                .into_any_element(),
            PowerSource::Unknown => shell
                .child(
                    div()
                        .text_xs()
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(palette.on_scene_muted)
                        .child("Питание"),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(palette.on_scene_muted)
                        .child("данные недоступны"),
                )
                .into_any_element(),
        }
    }

    /// Ошибка последнего действия — с въездом и возможностью закрыть.
    fn render_error(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let radius = cx.theme().radius;
        let danger = cx.theme().danger;
        let shown = self.error_in;

        div().when_some(self.error.clone(), move |this, message| {
            this.child(
                h_flex()
                    .id("action-status")
                    .test_support()
                    .relative()
                    .top(px((shown - 1.0) * 10.0))
                    .opacity(shown)
                    .items_start()
                    .gap(rems(0.5))
                    .max_w_full()
                    .px(rems(0.875))
                    .py(rems(0.5))
                    .rounded(radius)
                    .bg(danger.opacity(0.14))
                    .border_1()
                    .border_color(danger.opacity(0.38))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .text_xs()
                            .text_color(danger)
                            .child(message),
                    )
                    .child(
                        Button::new("dismiss-error")
                            .ghost()
                            .xsmall()
                            .icon(IconName::CircleX)
                            .accessibility_label("Скрыть сообщение")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.set_error(None, cx);
                                cx.notify();
                            })),
                    ),
            )
        })
    }
}

impl Render for RootView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let palette = Palette::for_time_of_day(self.scene);
        let mut page = div()
            .relative()
            .size_full()
            .overflow_hidden()
            .bg(palette.background);

        if self.stage.shows_app() {
            let title_bar = self.render_title_bar(cx).into_any_element();
            let content = self.render_content(window, cx).into_any_element();
            page = page.child(self.render_scene(window)).child(
                div()
                    .absolute()
                    .inset_0()
                    .flex()
                    .flex_col()
                    .opacity(self.reveal)
                    .child(title_bar)
                    .child(div().flex_1().min_h_0().child(content)),
            );
        }

        if self.stage != Stage::Ready {
            page = page.child(self.render_seed(window));
        }

        if self.veil_on {
            let curve = (self.veil * std::f32::consts::PI).sin();
            page = page.child(
                div()
                    .absolute()
                    .inset_0()
                    .bg(palette.scrim)
                    .opacity(curve * 0.55),
            );
        }

        self.root(window, cx, page.into_any_element())
    }
}

// ── Кривые анимации ──────────────────────────────────────────────────────────

fn ease_out_cubic(t: f32) -> f32 {
    let u = 1.0 - t.clamp(0.0, 1.0);
    1.0 - u * u * u
}

fn ease_in_out_cubic(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        let u = -2.0 * t + 2.0;
        1.0 - u * u * u / 2.0
    }
}

/// Пружинное появление с лёгким перехлёстом.
fn ease_out_back(t: f32) -> f32 {
    const C1: f32 = 1.70158;
    const C3: f32 = C1 + 1.0;
    let u = t.clamp(0.0, 1.0) - 1.0;
    1.0 + C3 * u * u * u + C1 * u * u
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ── Тема и ресурсы ───────────────────────────────────────────────────────────

/// Палитра сцены как тема приложения: меню, диалог и тултипы наследуют стиль.
///
/// Вызывать только при смене периода: `Theme::update` перерисовывает все окна.
fn apply_scene_theme(period: TimeOfDay, cx: &mut App) {
    let palette = Palette::for_time_of_day(period);
    let mut colors = *ThemeColor::dark();

    colors.background = palette.background;
    colors.foreground = palette.on_scene;
    colors.border = palette.hairline;
    colors.input = palette.on_scene.opacity(0.18);

    colors.popover = palette.glass;
    colors.popover_foreground = palette.on_scene;
    colors.overlay = gpui_kit::black().opacity(0.48);

    colors.list = palette.glass;
    colors.list_hover = palette.accent.opacity(0.16);
    colors.list_active = palette.accent.opacity(0.26);
    colors.list_active_border = palette.accent.opacity(0.48);
    colors.accent = palette.tint;
    colors.accent_foreground = palette.on_scene;

    colors.muted = palette.glass;
    colors.muted_foreground = palette.on_scene_muted;
    colors.secondary = palette.tint;
    colors.secondary_foreground = palette.on_scene;
    colors.secondary_hover = palette.accent.opacity(0.18);
    colors.secondary_active = palette.accent.opacity(0.26);

    colors.primary = palette.accent;
    colors.primary_foreground = palette.on_accent;
    colors.primary_hover = palette.accent_light;
    colors.primary_active = palette.accent_dark;
    colors.button_primary = palette.accent;
    colors.button_primary_foreground = palette.on_accent;
    colors.button_primary_hover = palette.accent_light;
    colors.button_primary_active = palette.accent_dark;

    colors.button = palette.tint;
    colors.button_foreground = palette.on_scene;
    colors.button_hover = palette.accent.opacity(0.20);
    colors.button_active = palette.accent.opacity(0.28);

    colors.title_bar = palette.background.opacity(0.);
    colors.title_bar_border = palette.background.opacity(0.);
    colors.ring = palette.accent;
    colors.selection = palette.accent.opacity(0.30);
    colors.scrollbar = palette.glass;
    colors.scrollbar_thumb = palette.accent.opacity(0.34);
    colors.scrollbar_thumb_hover = palette.accent.opacity(0.52);

    Theme::update(cx, |theme| {
        theme.colors = colors;
        theme.radius = px(11.);
        theme.radius_lg = px(15.);
    });
}

/// Исходник слоя сцены: читается один раз на путь.
fn scene_source(path: &'static str) -> Option<SharedString> {
    static SOURCES: OnceLock<Mutex<HashMap<&'static str, SharedString>>> = OnceLock::new();
    let map = SOURCES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().ok()?;
    if let Some(entry) = map.get(path) {
        return Some(entry.clone());
    }
    let svg = crate::assets::svg(path)?;
    let raw: SharedString = String::from_utf8_lossy(&svg).into_owned().into();
    map.insert(path, raw.clone());
    Some(raw)
}

/// Иконка; раскраска кэшируется по акценту, а не считается каждый кадр.
fn icon_image(icon: AppIcon, size: f32, palette: &Palette) -> VectorImage {
    static ART: OnceLock<Mutex<HashMap<(&'static str, u32), (SharedString, SharedString)>>> =
        OnceLock::new();
    let key = (icon.asset_path(), u32::from(palette.accent.to_rgb()));
    let map = ART.get_or_init(|| Mutex::new(HashMap::new()));

    if let Ok(mut map) = map.lock() {
        if let Some((id, svg)) = map.get(&key) {
            return VectorImage::square(id.clone(), svg.clone(), size);
        }
        let tinted = vector::tint(
            &String::from_utf8_lossy(&icon.svg()),
            palette.accent,
            palette.accent_dark,
        );
        let entry: (SharedString, SharedString) = (
            format!("icon:{}:{:08x}", icon.asset_path(), key.1).into(),
            tinted.into(),
        );
        map.insert(key, entry.clone());
        return VectorImage::square(entry.0, entry.1, size);
    }

    // Замок отравлен — рисуем без кэша, интерфейс не должен падать.
    let tinted = vector::tint(
        &String::from_utf8_lossy(&icon.svg()),
        palette.accent,
        palette.accent_dark,
    );
    VectorImage::square(
        format!("icon:{}:{:08x}", icon.asset_path(), key.1),
        tinted,
        size,
    )
}

/// SVG кольца загрузки: дуга, соответствующая прогрессу.
fn loading_ring(progress: u8, palette: &Palette) -> String {
    const RADIUS: f32 = 120.;
    const CIRCUMFERENCE: f32 = 2. * std::f32::consts::PI * RADIUS;
    let offset = CIRCUMFERENCE * (1. - progress.min(100) as f32 / 100.);

    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">
<defs><linearGradient id="ring" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="{light}"/><stop offset="1" stop-color="{accent}"/>
</linearGradient></defs>
<circle cx="128" cy="128" r="{radius}" fill="none" stroke="{track}" stroke-width="7"/>
<circle cx="128" cy="128" r="{radius}" fill="none" stroke="url(#ring)" stroke-width="7"
 stroke-linecap="round" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
 transform="rotate(-90 128 128)"/>
</svg>"##,
        light = vector::hex(palette.accent_light),
        accent = vector::hex(palette.accent),
        track = vector::hex(palette.on_scene.opacity(0.09)),
        radius = RADIUS,
        circumference = CIRCUMFERENCE,
        offset = offset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn easings_hit_their_endpoints() {
        for f in [ease_out_cubic, ease_in_out_cubic, ease_out_back] {
            assert!(f(0.0).abs() < 1e-4);
            assert!((f(1.0) - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn smoothstep_is_monotonic_and_bounded() {
        assert_eq!(smoothstep(0.3, 0.9, 0.0), 0.0);
        assert_eq!(smoothstep(0.3, 0.9, 1.0), 1.0);
        assert!(smoothstep(0.3, 0.9, 0.5) < smoothstep(0.3, 0.9, 0.7));
    }

    #[test]
    fn only_bloom_and_ready_show_the_app() {
        assert!(!Stage::Seed.shows_app());
        assert!(!Stage::Charge.shows_app());
        assert!(Stage::Bloom.shows_app());
        assert!(Stage::Ready.shows_app());
    }

    #[test]
    fn layout_shrinks_monotonically_with_height() {
        let heights = [800.0, 640.0, 560.0, 520.0, 470.0, 420.0];
        let layouts: Vec<Layout> = heights.iter().map(|h| Layout::for_height(*h)).collect();
        for pair in layouts.windows(2) {
            assert!(pair[1].clock <= pair[0].clock);
            assert!(pair[1].power <= pair[0].power);
            assert!(pair[1].gap <= pair[0].gap);
            assert!(pair[1].pad_bottom <= pair[0].pad_bottom);
        }
    }

    #[test]
    fn smallest_layout_drops_optional_text() {
        let layout = Layout::for_height(MIN_WINDOW_HEIGHT);
        assert!(!layout.show_hint);
        assert!(!layout.show_greeting);
    }

    #[test]
    fn scene_aspect_matches_source_viewbox() {
        assert!((SCENE_ASPECT - 1100.0 / 1200.0).abs() < 1e-6);
    }
}
