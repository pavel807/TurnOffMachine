use tauri::Manager;
use tauri_plugin_shell::ShellExt;

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
        .setup(|app| {
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
            cancel_power_action
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}