//! Действия питания: macOS, Linux и Windows.
//!
//! Если платформа не умеет действие, оно помечается недоступным и объясняет
//! причину — подменять его другим нельзя: пользователь получит не тот результат.

use std::process::Command;

use crate::assets::AppIcon;

/// Действие питания.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PowerAction {
    Shutdown,
    Restart,
    Sleep,
    Hibernate,
}

struct Attempt {
    program: &'static str,
    args: Vec<String>,
}

fn attempt(program: &'static str, args: &[&str]) -> Attempt {
    Attempt {
        program,
        args: args.iter().map(|a| (*a).to_string()).collect(),
    }
}

enum Failure {
    /// Программы нет — есть смысл попробовать другой путь.
    Missing,
    /// Программа отказала: текст нужен пользователю.
    Refused(String),
}

impl PowerAction {
    pub const ALL: [Self; 4] = [Self::Shutdown, Self::Restart, Self::Sleep, Self::Hibernate];

    pub fn label(self) -> &'static str {
        match self {
            Self::Shutdown => "Выключить",
            Self::Restart => "Перезагрузить",
            Self::Sleep => "Сон",
            Self::Hibernate => "Гибернация",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Self::Shutdown => "закрыть приложения и выключить компьютер",
            Self::Restart => "перезагрузить компьютер",
            Self::Sleep => "перевести компьютер в сон",
            Self::Hibernate => "сохранить состояние на диск и уснуть",
        }
    }

    /// Короткая подпись для строки меню.
    pub fn caption(self) -> &'static str {
        match self {
            Self::Shutdown => "выключить компьютер",
            Self::Restart => "перезагрузить компьютер",
            Self::Sleep => "перевести в сон",
            Self::Hibernate => "сохранить на диск и уснуть",
        }
    }

    pub fn icon(self) -> AppIcon {
        match self {
            Self::Shutdown => AppIcon::Power,
            Self::Restart => AppIcon::Restart,
            Self::Sleep => AppIcon::Sleep,
            Self::Hibernate => AppIcon::Hibernate,
        }
    }

    /// Умеет ли платформа это действие.
    pub fn is_supported(self) -> bool {
        if cfg!(target_os = "macos") {
            return !matches!(self, Self::Hibernate);
        }
        if cfg!(any(target_os = "linux", target_os = "windows")) {
            return true;
        }
        matches!(self, Self::Shutdown | Self::Restart)
    }

    pub fn unsupported_reason(self) -> Option<&'static str> {
        if self.is_supported() {
            return None;
        }
        if cfg!(target_os = "macos") && matches!(self, Self::Hibernate) {
            return Some(
                "macOS не выполняет гибернацию по команде: режим сна задаётся в настройках питания",
            );
        }
        Some("действие недоступно на этой платформе")
    }

    pub fn needs_confirmation(self) -> bool {
        matches!(self, Self::Shutdown | Self::Restart)
    }

    pub fn confirm_title(self) -> &'static str {
        let mac = cfg!(target_os = "macos");
        match (self, mac) {
            (Self::Shutdown, true) => "Выключить Mac?",
            (Self::Shutdown, false) => "Выключить компьютер?",
            (Self::Restart, true) => "Перезагрузить Mac?",
            (Self::Restart, false) => "Перезагрузить компьютер?",
            (Self::Sleep, _) => "Перевести компьютер в сон?",
            (Self::Hibernate, _) => "Перевести компьютер в гибернацию?",
        }
    }

    pub fn confirm_description(self) -> &'static str {
        match self {
            Self::Shutdown | Self::Restart => "Несохранённые данные будут потеряны.",
            Self::Sleep => "Работа продолжится после пробуждения.",
            Self::Hibernate => "Работа продолжится после выхода из гибернации.",
        }
    }

    #[cfg(target_os = "macos")]
    fn attempts(self) -> Vec<Attempt> {
        match self {
            Self::Shutdown => vec![attempt(
                "osascript",
                &["-e", "tell application \"System Events\" to shut down"],
            )],
            Self::Restart => vec![attempt(
                "osascript",
                &["-e", "tell application \"System Events\" to restart"],
            )],
            Self::Sleep => vec![attempt("pmset", &["sleepnow"])],
            Self::Hibernate => Vec::new(),
        }
    }

    #[cfg(target_os = "linux")]
    fn attempts(self) -> Vec<Attempt> {
        let verb = match self {
            Self::Shutdown => "poweroff",
            Self::Restart => "reboot",
            Self::Sleep => "suspend",
            Self::Hibernate => "hibernate",
        };
        let mut list = vec![attempt("systemctl", &[verb]), attempt("loginctl", &[verb])];
        match self {
            Self::Shutdown => list.push(attempt("shutdown", &["-h", "now"])),
            Self::Restart => list.push(attempt("shutdown", &["-r", "now"])),
            Self::Sleep | Self::Hibernate => {}
        }
        list
    }

    #[cfg(target_os = "windows")]
    fn attempts(self) -> Vec<Attempt> {
        match self {
            Self::Shutdown => vec![attempt("shutdown", &["/s", "/f", "/t", "0"])],
            Self::Restart => vec![attempt("shutdown", &["/r", "/f", "/t", "0"])],
            Self::Sleep => vec![attempt(
                "rundll32.exe",
                &["powrprof.dll,SetSuspendState", "0,1,0"],
            )],
            Self::Hibernate => vec![attempt("shutdown", &["/h"])],
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    fn attempts(self) -> Vec<Attempt> {
        match self {
            Self::Shutdown => vec![attempt("shutdown", &["-h", "now"])],
            Self::Restart => vec![attempt("shutdown", &["-r", "now"])],
            Self::Sleep | Self::Hibernate => Vec::new(),
        }
    }
}

/// Выполнить действие питания.
pub fn execute(action: PowerAction) -> Result<(), String> {
    if let Some(reason) = action.unsupported_reason() {
        return Err(reason.to_string());
    }
    run_attempts(action.attempts())
        .map_err(|message| format!("{}: {message}", action.label().to_lowercase()))
}

/// Отменить отложенное системой выключение.
pub fn cancel_pending_shutdown() -> Result<(), String> {
    let candidates = if cfg!(target_os = "windows") {
        vec![attempt("shutdown", &["/a"])]
    } else if cfg!(target_os = "macos") {
        vec![attempt("killall", &["shutdown"])]
    } else {
        vec![attempt("shutdown", &["-c"])]
    };

    run_attempts(candidates).map_err(|message| {
        if cfg!(target_os = "macos") {
            format!(
                "Отложенное выключение не найдено. Если его запустил другой пользователь, \
                 отмена требует прав администратора. ({message})"
            )
        } else {
            message
        }
    })
}

fn run_attempts(candidates: Vec<Attempt>) -> Result<(), String> {
    let mut first_refusal: Option<String> = None;
    for candidate in candidates {
        match run(&candidate) {
            Ok(()) => return Ok(()),
            Err(Failure::Missing) => continue,
            Err(Failure::Refused(message)) => {
                if first_refusal.is_none() {
                    first_refusal = Some(message);
                }
            }
        }
    }
    Err(first_refusal
        .unwrap_or_else(|| "не найден системный способ выполнить действие".to_string()))
}

fn run(candidate: &Attempt) -> Result<(), Failure> {
    let mut command = Command::new(candidate.program);
    command.args(&candidate.args);
    hide_console(&mut command);

    let output = match command.output() {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Err(Failure::Missing),
        Err(error) => {
            return Err(Failure::Refused(format!(
                "не удалось запустить {}: {error}",
                candidate.program
            )));
        }
    };

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        format!(
            "{} завершился с кодом {}",
            candidate.program,
            output.status.code().unwrap_or(-1)
        )
    };

    Err(Failure::Refused(match advice(&detail) {
        Some(advice) => format!("{detail}. {advice}"),
        None => detail,
    }))
}

fn advice(message: &str) -> Option<&'static str> {
    let text = message.to_lowercase();
    if text.contains("not authorized") || text.contains("-1743") || text.contains("apple events") {
        return Some(
            "Разрешите приложению управлять System Events: \
             Системные настройки → Конфиденциальность и безопасность → Автоматизация",
        );
    }
    if text.contains("authentication")
        || text.contains("access denied")
        || text.contains("permission denied")
        || text.contains("not permitted")
        || text.contains("must be root")
    {
        return Some("Нужны права: выполните действие из активного графического сеанса");
    }
    if text.contains("hibern") && cfg!(target_os = "windows") {
        return Some("Включите гибернацию командой powercfg /h on от администратора");
    }
    if text.contains("hibern") || text.contains("swap") {
        return Some("Гибернация требует настроенного swap не меньше объёма памяти");
    }
    None
}

#[cfg(target_os = "windows")]
fn hide_console(command: &mut Command) {
    use std::os::windows::process::CommandExt as _;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(target_os = "windows"))]
fn hide_console(_command: &mut Command) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_action_has_a_command() {
        for action in PowerAction::ALL {
            assert_eq!(
                action.is_supported(),
                !action.attempts().is_empty(),
                "{action:?}"
            );
        }
    }

    #[test]
    fn unsupported_action_explains_itself() {
        for action in PowerAction::ALL {
            assert_eq!(action.is_supported(), action.unsupported_reason().is_none());
        }
    }

    #[test]
    fn only_destructive_actions_need_confirmation() {
        assert!(PowerAction::Shutdown.needs_confirmation());
        assert!(!PowerAction::Sleep.needs_confirmation());
    }
}
