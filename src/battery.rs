//! Источник питания: macOS, Linux, Windows.
//!
//! Отсутствие батареи и недоступность данных — разные состояния. Выдавать
//! неизвестность за «питание от сети» нельзя: это неверное сообщение.

use gpui_kit::component::IconName;

/// Заряд батареи.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatteryInfo {
    /// Остаток заряда в процентах.
    pub percent: u8,
    /// Идёт ли зарядка.
    pub charging: bool,
}

impl BatteryInfo {
    pub fn icon(self) -> IconName {
        if self.charging {
            IconName::BatteryCharging
        } else if self.percent >= 80 {
            IconName::BatteryFull
        } else if self.percent >= 40 {
            IconName::BatteryMedium
        } else {
            IconName::BatteryLow
        }
    }

    pub fn status(self) -> &'static str {
        if self.charging {
            "зарядка"
        } else {
            "от батареи"
        }
    }
}

/// Откуда питается компьютер.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerSource {
    /// Есть батарея, её состояние известно.
    Battery(BatteryInfo),
    /// Батареи нет — питание от сети.
    Mains,
    /// Прочитать состояние не удалось.
    Unknown,
}

impl PowerSource {
    /// Прочитать состояние питания.
    pub fn read() -> Self {
        platform::read()
    }
}

/// Разобрать вывод `pmset -g batt`.
pub fn parse(text: &str) -> Option<BatteryInfo> {
    for line in text.lines() {
        let Some((head, tail)) = line.split_once('%') else {
            continue;
        };
        let Some(digits) = head
            .rsplit(|ch: char| !ch.is_ascii_digit())
            .find(|part| !part.is_empty())
        else {
            continue;
        };
        let Ok(percent) = digits.parse::<u8>() else {
            continue;
        };
        let tail = tail.to_lowercase();
        let discharging = tail.contains("discharging");
        let charging = !discharging && (tail.contains("charging") || tail.contains("ac power"));
        return Some(BatteryInfo {
            percent: percent.min(100),
            charging,
        });
    }
    None
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{BatteryInfo, PowerSource, parse};
    use std::process::Command;

    pub fn read() -> PowerSource {
        let Ok(output) = Command::new("pmset").args(["-g", "batt"]).output() else {
            return PowerSource::Unknown;
        };
        let text = String::from_utf8_lossy(&output.stdout);
        if let Some(info) = parse(&text) {
            return PowerSource::Battery(info);
        }
        if text.contains("AC Power") {
            return PowerSource::Mains;
        }
        PowerSource::Unknown
    }

    #[allow(dead_code)]
    fn _assert_types(_: BatteryInfo) {}
}

#[cfg(target_os = "linux")]
mod platform {
    use super::{BatteryInfo, PowerSource};
    use std::fs;
    use std::path::Path;

    pub fn read() -> PowerSource {
        let root = Path::new("/sys/class/power_supply");
        let Ok(entries) = fs::read_dir(root) else {
            return PowerSource::Unknown;
        };

        let mut saw_battery = false;
        for entry in entries.flatten() {
            let path = entry.path();
            let kind = fs::read_to_string(path.join("type")).unwrap_or_default();
            if kind.trim() != "Battery" {
                continue;
            }
            saw_battery = true;

            let capacity = fs::read_to_string(path.join("capacity")).unwrap_or_default();
            let Ok(percent) = capacity.trim().parse::<u8>() else {
                continue;
            };
            let status = fs::read_to_string(path.join("status")).unwrap_or_default();
            let status = status.trim();
            let charging = status == "Charging" || status == "Full";
            return PowerSource::Battery(BatteryInfo {
                percent: percent.min(100),
                charging,
            });
        }

        if saw_battery {
            PowerSource::Unknown
        } else {
            PowerSource::Mains
        }
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use super::{BatteryInfo, PowerSource};

    #[repr(C)]
    struct SystemPowerStatus {
        ac_line_status: u8,
        battery_flag: u8,
        battery_life_percent: u8,
        reserved1: u8,
        battery_life_time: u32,
        battery_full_life_time: u32,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetSystemPowerStatus(status: *mut SystemPowerStatus) -> i32;
    }

    pub fn read() -> PowerSource {
        let mut status = SystemPowerStatus {
            ac_line_status: 255,
            battery_flag: 255,
            battery_life_percent: 255,
            reserved1: 0,
            battery_life_time: u32::MAX,
            battery_full_life_time: u32::MAX,
        };
        let ok = unsafe { GetSystemPowerStatus(&mut status) };
        if ok == 0 {
            return PowerSource::Unknown;
        }
        // 128 — батареи нет, 255 — состояние неизвестно.
        if status.battery_flag & 128 != 0 {
            return PowerSource::Mains;
        }
        if status.battery_life_percent == 255 {
            return PowerSource::Unknown;
        }
        PowerSource::Battery(BatteryInfo {
            percent: status.battery_life_percent.min(100),
            charging: status.ac_line_status == 1 && status.battery_life_percent < 100,
        })
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
mod platform {
    use super::PowerSource;
    pub fn read() -> PowerSource {
        PowerSource::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_discharging() {
        let text = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t87%; discharging; 3:20 remaining present: true";
        let info = parse(text).unwrap();
        assert_eq!(info.percent, 87);
        assert!(!info.charging);
    }

    #[test]
    fn parses_charging() {
        let text = " -InternalBattery-0\t42%; charging; 1:10 remaining present: true";
        let info = parse(text).unwrap();
        assert_eq!(info.percent, 42);
        assert!(info.charging);
    }

    #[test]
    fn no_battery_line_is_not_a_battery() {
        assert!(parse("Now drawing from 'AC Power'").is_none());
    }
}
