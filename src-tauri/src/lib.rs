use std::process::Command as StdCommand;
use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_shell::ShellExt;

use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

struct SystemState(Mutex<System>);

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct SystemStats {
    cpu: f32,
    mem_used: u64,
    mem_total: u64,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct BatteryInfo {
    percent: Option<u8>,
    charging: Option<bool>,
}

#[tauri::command]
fn get_system_stats(state: tauri::State<'_, SystemState>) -> SystemStats {
    let mut sys = state.0.lock().unwrap();
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    SystemStats {
        cpu: sys.global_cpu_usage(),
        mem_used: sys.used_memory(),
        mem_total: sys.total_memory(),
    }
}

#[tauri::command]
async fn get_battery_info() -> BatteryInfo {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = StdCommand::new("pmset").args(["-g", "batt"]).output() {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines() {
                for token in line.split_whitespace() {
                    if let Some(pct) = token.strip_suffix("%;") {
                        if let Ok(p) = pct.parse::<u8>() {
                            let charging = !line.to_lowercase().contains("discharging");
                            return BatteryInfo {
                                percent: Some(p.min(100)),
                                charging: Some(charging),
                            };
                        }
                    }
                }
            }
        }
        BatteryInfo {
            percent: None,
            charging: None,
        }
    }
    #[cfg(target_os = "windows")]
    {
        let script = "$b = Get-CimInstance Win32_Battery | Select-Object -First 1; if ($b) { \"$($b.EstimatedChargeRemaining);$($b.BatteryStatus)\" }";
        if let Ok(out) = StdCommand::new("powershell")
            .args(["-NoProfile", "-Command", script])
            .output()
        {
            let text = String::from_utf8_lossy(&out.stdout);
            let s = text.trim();
            if !s.is_empty() {
                let mut parts = s.splitn(2, ';');
                if let (Some(p), Some(st)) = (parts.next(), parts.next()) {
                    if let Ok(p) = p.trim().parse::<u8>() {
                        let charging = st.trim() != "1";
                        return BatteryInfo {
                            percent: Some(p.min(100)),
                            charging: Some(charging),
                        };
                    }
                }
            }
        }
        BatteryInfo {
            percent: None,
            charging: None,
        }
    }
    #[cfg(target_os = "linux")]
    {
        for name in ["BAT0", "BAT1", "BAT2"] {
            let base = format!("/sys/class/power_supply/{name}");
            let Ok(cap) = std::fs::read_to_string(format!("{base}/capacity")) else {
                continue;
            };
            let Ok(p) = cap.trim().parse::<u8>() else {
                continue;
            };
            let status = std::fs::read_to_string(format!("{base}/status")).unwrap_or_default();
            return BatteryInfo {
                percent: Some(p.min(100)),
                charging: Some(status.trim() == "Charging"),
            };
        }
        BatteryInfo {
            percent: None,
            charging: None,
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        BatteryInfo {
            percent: None,
            charging: None,
        }
    }
}

#[tauri::command]
async fn execute_power_action(
    app: tauri::AppHandle,
    action: String,
    _delay_seconds: u64,
) -> Result<String, String> {
    let shell = app.shell();

    let (program, args): (String, Vec<String>) = match std::env::consts::OS {
        "macos" => match action.as_str() {
            "Shutdown" => (
                "osascript".to_string(),
                vec![
                    "-e".to_string(),
                    "tell application \"System Events\" to shut down".to_string(),
                ],
            ),
            "Restart" => (
                "osascript".to_string(),
                vec![
                    "-e".to_string(),
                    "tell application \"System Events\" to restart".to_string(),
                ],
            ),
            "Sleep" => (
                "osascript".to_string(),
                vec![
                    "-e".to_string(),
                    "tell application \"System Events\" to sleep".to_string(),
                ],
            ),
            "Hibernate" => ("pmset".to_string(), vec!["sleepnow".to_string()]),
            _ => return Err("Unknown action".to_string()),
        },
        "linux" => match action.as_str() {
            "Shutdown" => ("systemctl".to_string(), vec!["poweroff".to_string()]),
            "Restart" => ("systemctl".to_string(), vec!["reboot".to_string()]),
            "Sleep" => ("systemctl".to_string(), vec!["suspend".to_string()]),
            "Hibernate" => ("systemctl".to_string(), vec!["hibernate".to_string()]),
            _ => return Err("Unknown action".to_string()),
        },
        "windows" => match action.as_str() {
            "Shutdown" => (
                "shutdown".to_string(),
                vec!["/s".to_string(), "/t".to_string(), "0".to_string()],
            ),
            "Restart" => (
                "shutdown".to_string(),
                vec!["/r".to_string(), "/t".to_string(), "0".to_string()],
            ),
            "Sleep" => (
                "rundll32.exe".to_string(),
                vec![
                    "powrprof.dll,SetSuspendState".to_string(),
                    "0".to_string(),
                    "1".to_string(),
                    "0".to_string(),
                ],
            ),
            "Hibernate" => ("shutdown".to_string(), vec!["/h".to_string()]),
            _ => return Err("Unknown action".to_string()),
        },
        _ => return Err("Unsupported OS".to_string()),
    };

    let output = shell
        .command(program)
        .args(args)
        .output()
        .await
        .map_err(|e| format!("Failed to execute command: {}", e))?;

    if output.status.success() {
        Ok("Action executed successfully".to_string())
    } else {
        let status_code = output.status.code().unwrap_or(-1);
        Err(format!(
            "Command failed with status code: {}. Stderr: {}",
            status_code,
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

#[tauri::command]
async fn cancel_power_action(app: tauri::AppHandle) -> Result<String, String> {
    let shell = app.shell();

    match std::env::consts::OS {
        "macos" | "linux" => {
            let _ = shell
                .command("shutdown")
                .args(["-k"])
                .output()
                .await;
            Ok("Timer cancelled".to_string())
        }
        "windows" => {
            let _ = shell
                .command("shutdown")
                .args(["/a"])
                .output()
                .await;
            Ok("Timer cancelled or no active timer".to_string())
        }
        _ => Ok("Nothing to cancel".to_string()),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            let mut sys = System::new_with_specifics(
                RefreshKind::nothing()
                    .with_cpu(CpuRefreshKind::everything())
                    .with_memory(MemoryRefreshKind::everything()),
            );
            sys.refresh_cpu_usage();
            std::thread::sleep(std::time::Duration::from_millis(200));
            sys.refresh_cpu_usage();
            app.manage(SystemState(Mutex::new(sys)));

            #[cfg(target_os = "macos")]
            {
                use objc2_app_kit::{NSColor, NSWindow};

                let window = app.get_webview_window("main").unwrap();
                let ns_window_ptr = window.ns_window().unwrap() as *mut NSWindow;
                let ns_window: &NSWindow = unsafe { &*ns_window_ptr };

                let clear_color = NSColor::clearColor();
                ns_window.setBackgroundColor(Some(&clear_color));
                ns_window.setOpaque(false);
                ns_window.setHasShadow(false);
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            execute_power_action,
            cancel_power_action,
            get_system_stats,
            get_battery_info
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}