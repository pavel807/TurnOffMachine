//! Локальное время и тексты интерфейса.
//!
//! Единственный модуль, знающий о библиотеке времени: наружу отдаётся
//! `LocalNow` — снимок простыми числами.

use chrono::{Datelike as _, Local, Timelike as _};

use crate::scene::TimeOfDay;

/// Снимок местного времени.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalNow {
    pub year: i32,
    /// Месяц, 1..=12.
    pub month: usize,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    /// День недели, 0 — понедельник.
    pub weekday: usize,
}

impl LocalNow {
    pub fn read() -> Self {
        let now = Local::now();
        Self {
            year: now.year(),
            month: now.month() as usize,
            day: now.day(),
            hour: now.hour(),
            minute: now.minute(),
            second: now.second(),
            weekday: now.weekday().num_days_from_monday() as usize,
        }
    }

    /// Часы и минуты, 24 часа.
    pub fn clock(&self) -> String {
        format!("{:02}:{:02}", self.hour, self.minute)
    }

    /// Секунды двумя цифрами.
    pub fn seconds(&self) -> String {
        format!("{:02}", self.second)
    }
}

const WEEKDAYS: [&str; 7] = [
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
];

const MONTHS: [&str; 12] = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
];

/// Дата в виде «вторник, 25 ноября 2026».
pub fn long_date(now: &LocalNow) -> String {
    let weekday = WEEKDAYS[now.weekday.min(6)];
    let month = MONTHS[now.month.saturating_sub(1).min(11)];
    format!("{weekday}, {} {month} {}", now.day, now.year)
}

/// Приветствие для периода суток.
pub fn greeting(time: TimeOfDay) -> &'static str {
    match time {
        TimeOfDay::Morning => "Доброе утро",
        TimeOfDay::Day => "Добрый день",
        TimeOfDay::Evening => "Добрый вечер",
        TimeOfDay::Night => "Доброй ночи",
    }
}
