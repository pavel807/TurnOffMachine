//! OCR чтения экрана: скриншот → текст с координатами. Без модели зрения и
//! без VL — системные распознаватели каждой ОС, без новых Rust-зависимостей:
//! - Windows: `Windows.Media.Ocr` (WinRT) через PowerShell;
//! - macOS: **Vision** (`VNRecognizeTextRequest`) через osascript/JXA —
//!   домашний стиль auto/ax.rs, osascript есть в любой macOS;
//! - Linux: `tesseract` CLI с TSV-выдачей (опционально, честная подсказка
//!   «установите tesseract-ocr»).
//!
//! Зачем координаты: модель получает строки с прямоугольниками и может
//! кликнуть по любой надписи через `desktop_input click` — это второй путь
//! Computer Use там, где дерево доступности недоступно (игры, canvas,
//! удалённые сессии).
//!
//! Единицы координат — те же, что ждёт `desktop_input` на этой ОС:
//! физические пиксели на Windows, логические точки на macOS (пересчёт
//! Retina-масштаба внутри скрипта), пиксели скриншота на Linux.
//!
//! Скрипт Windows честно сообщает масштаб: если скриншот больше
//! `OcrEngine.MaxImageDimension` (у движка лимит 1000 px), картинка
//! уменьшается, и парсер возвращает координаты к пикселям экрана.

/// Одна строка распознанного текста с прямоугольником в координатах ввода
/// этой ОС (см. модульную документацию).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrLine {
    pub text: String,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

/// Считать текст с текущего экрана: скриншот + OCR.
/// Возвращает строки в порядке чтения (сверху вниз) и путь к скриншоту.
pub fn read_screen_text() -> Result<(Vec<OcrLine>, String), String> {
    let path =
        crate::platform::screenshot().ok_or_else(|| "Не удалось сделать скриншот".to_string())?;
    #[cfg(target_os = "windows")]
    {
        let (scale, lines) = run_ocr(&path)?;
        let inv = 1.0 / scale;
        let mapped: Vec<OcrLine> = lines
            .into_iter()
            .map(|l| OcrLine {
                text: l.text,
                x: ((l.x as f32) * inv).round() as i32,
                y: ((l.y as f32) * inv).round() as i32,
                w: ((l.w as f32) * inv).round() as i32,
                h: ((l.h as f32) * inv).round() as i32,
            })
            .collect();
        Ok((mapped, path))
    }
    #[cfg(target_os = "macos")]
    {
        let lines = vision_ocr(&path)?;
        Ok((lines, path))
    }
    #[cfg(target_os = "linux")]
    {
        let lines = tesseract_ocr(&path)?;
        Ok((lines, path))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        Err("OCR экрана поддерживается на Windows, macOS и Linux".to_string())
    }
}

/// Человекочитаемый отчёт для модели: строки с координатами, чтобы можно
/// было кликнуть по надписи через desktop_input click.
pub fn format_lines(lines: &[OcrLine]) -> String {
    let mut out = String::new();
    for (i, l) in lines.iter().enumerate() {
        out.push_str(&format!(
            "[{}] «{}» @ ({}, {}) {}×{}\n",
            i + 1,
            l.text,
            l.x,
            l.y,
            l.w,
            l.h
        ));
    }
    out
}

// =========================================================================
// Общий разбор: `OCR-OK\t<масштаб>` + строки `x\ty\tw\th\tтекст`
// =========================================================================

/// Разбор вывода OCR-скрипта Windows (PowerShell) и macOS (JXA). Координаты
/// приходят в масштабе, сообщённом скриптом, — здесь возвращаем их к
/// координатам скриншота.
#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), allow(dead_code))]
fn parse_ocr_output(out: &str) -> Option<(f32, Vec<OcrLine>)> {
    let mut lines = Vec::new();
    let mut scale: f32 = 1.0;
    for raw in out.lines() {
        let line = raw.trim_end();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("OCR-OK") {
            let value = rest.trim();
            scale = value
                .parse::<f32>()
                .ok()
                .filter(|s| *s > 0.0)
                .unwrap_or(1.0);
            continue;
        }
        let mut parts = line.splitn(5, '\t');
        let x = parts.next()?.trim().parse::<i32>().ok()?;
        let y = parts.next()?.trim().parse::<i32>().ok()?;
        let w = parts.next()?.trim().parse::<i32>().ok()?;
        let h = parts.next()?.trim().parse::<i32>().ok()?;
        let text = parts.next().unwrap_or("").trim().to_string();
        if text.is_empty() {
            continue;
        }
        lines.push(OcrLine { text, x, y, w, h });
    }
    if out.contains("OCR-OK") {
        Some((scale, lines))
    } else {
        None
    }
}

// =========================================================================
// Windows: Windows.Media.Ocr (WinRT) через PowerShell
// =========================================================================

#[cfg(target_os = "windows")]
fn run_ocr(path: &str) -> Result<(f32, Vec<OcrLine>), String> {
    let script = ocr_script(path);
    let out = super::windows::ps_run(&script)?;
    parse_ocr_output(&out).ok_or_else(|| {
        format!(
            "OCR вернул непонятный ответ: {}",
            out.chars().take(200).collect::<String>()
        )
    })
}

/// PowerShell: скриншот → Windows.Media.Ocr → строки с прямоугольниками.
/// Все подстановки — base64, как в остальном платформенном слое.
#[cfg(target_os = "windows")]
fn ocr_script(path: &str) -> String {
    format!(
        r#"
$p=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{}'));
Add-Type -AssemblyName System.Runtime.WindowsRuntime,System.Drawing | Out-Null;
$null=[Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime];
$null=[Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime];
$null=[Windows.Graphics.Imaging.BitmapDecoder,Windows.Graphics.Imaging,ContentType=WindowsRuntime];
$null=[Windows.Globalization.Language,Windows.Globalization,ContentType=WindowsRuntime];
$AsTask=([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {{ $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }}) | Select-Object -First 1;
function Await($t,$r) {{ $n=$AsTask.MakeGenericMethod($r).Invoke($null,@($t)); $n.Wait(-1) | Out-Null; $n.Result }};
$ocr=[Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages();
if (-not $ocr) {{ $ocr=[Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage((New-Object Windows.Globalization.Language('ru'))) }};
if (-not $ocr) {{ $ocr=[Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage((New-Object Windows.Globalization.Language('en-US'))) }};
if (-not $ocr) {{ throw 'OCR-язык не установлен: Параметры → Время и язык → Языки → добавьте русский или английский' }};
$img=[System.Drawing.Image]::FromFile($p);
$scale=1.0;
try {{
  $md=[Windows.Media.Ocr.OcrEngine]::MaxImageDimension;
  if ($img.Width -gt $md -or $img.Height -gt $md) {{
    $scale=[Math]::Round($md/[double][Math]::Max($img.Width,$img.Height),4);
    $nw=[int][Math]::Round($img.Width*$scale); $nh=[int][Math]::Round($img.Height*$scale);
    $tmp=[IO.Path]::ChangeExtension($p,'.ocr.png');
    $small=New-Object System.Drawing.Bitmap $nw,$nh;
    $g=[System.Drawing.Graphics]::FromImage($small);
    $g.InterpolationMode=[System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic;
    $g.DrawImage($img,0,0,$nw,$nh); $g.Dispose();
    $small.Save($tmp,[System.Drawing.Imaging.ImageFormat]::Png); $small.Dispose();
    $p=$tmp;
  }}
}} finally {{ $img.Dispose() }};
$file=Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($p)) ([Windows.Storage.StorageFile]);
$stream=Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream]);
$decoder=Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder]);
$bmp=Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap]);
$res=Await ($ocr.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult]);
('OCR-OK' + [char]9 + $scale.ToString([Globalization.CultureInfo]::InvariantCulture));
foreach ($l in $res.Lines) {{
  $minX=[double]::MaxValue; $minY=[double]::MaxValue; $maxX=[double]::MinValue; $maxY=[double]::MinValue;
  foreach ($w in $l.Words) {{ $r=$w.BoundingRect;
    if ($r.X -lt $minX) {{ $minX=$r.X }};
    if ($r.Y -lt $minY) {{ $minY=$r.Y }};
    if ($r.X+$r.Width -gt $maxX) {{ $maxX=$r.X+$r.Width }};
    if ($r.Y+$r.Height -gt $maxY) {{ $maxY=$r.Y+$r.Height }};
  }};
  if ($maxX -lt $minX) {{ continue }};
  ('{{0}}`t{{1}}`t{{2}}`t{{3}}`t{{4}}' -f [int]$minX,[int]$minY,[int]($maxX-$minX),[int]($maxY-$minY),$l.Text);
}}
"#,
        super::windows::b64(path)
    )
}

// =========================================================================
// macOS: Vision (VNRecognizeTextRequest) через osascript/JXA
// =========================================================================

/// JXA-скрипт OCR: параметры приходят через переменные окружения (путь и
/// размер скриншота в пикселях) — подстановок в сам скрипт нет. Координаты
/// сразу пересчитываются в логические точки (у Vision нормализованный
/// прямоугольник с началом в ЛЕВОМ НИЖНЕМ углу), потому что cliclick и
/// остальной desktop_input на macOS ждут именно их.
#[cfg(target_os = "macos")]
const VISION_SCRIPT: &str = r#"ObjC.import('Vision');
ObjC.import('AppKit');
ObjC.import('Foundation');
function run() {
  const env = $.NSProcessInfo.processInfo.environment;
  const path = env.objectForKey('JARVIS_OCR_PATH').js;
  const W = parseInt(env.objectForKey('JARVIS_OCR_W').js);
  const H = parseInt(env.objectForKey('JARVIS_OCR_H').js);
  const screen = $.NSScreen.screens.objectAtIndex(0);
  const logicalW = screen.frame.size.width;
  const scale = logicalW > 0 ? W / logicalW : 1.0;
  const lw = W / scale, lh = H / scale;
  const url = $.NSURL.fileURLWithPath(path);
  const handler = $.VNImageRequestHandler.alloc.initWithURLOptions(url, $.NSDictionary.dictionary);
  const request = $.VNRecognizeTextRequest.alloc.init;
  request.recognitionLevel = $.VNRequestTextRecognitionLevelAccurate;
  request.usesLanguageCorrection = false;
  request.recognitionLanguages = $(['ru-RU', 'en-US']);
  const errRef = Ref();
  const ok = handler.performRequestsError($([request]), errRef);
  if (!ok) {
    const msg = errRef[0] ? errRef[0].localizedDescription.js : 'неизвестная ошибка';
    return 'OCR-ERR' + '\t' + msg;
  }
  let out = 'OCR-OK' + '\t' + '1.0';
  const results = request.results;
  for (let i = 0; i < results.count; i++) {
    const obs = results.objectAtIndex(i);
    const cand = obs.topCandidates(1).objectAtIndex(0);
    const bb = obs.boundingBox;
    const x = Math.round(bb.origin.x * lw);
    const y = Math.round((1 - bb.origin.y - bb.size.height) * lh);
    const w = Math.round(bb.size.width * lw);
    const h = Math.round(bb.size.height * lh);
    out += '\n' + x + '\t' + y + '\t' + w + '\t' + h + '\t' + cand.string.js;
  }
  return out;
}
"#;

#[cfg(target_os = "macos")]
fn vision_ocr(path: &str) -> Result<Vec<OcrLine>, String> {
    let (w, h) = png_dimensions(path).ok_or_else(|| "Не удалось прочитать скриншот".to_string())?;
    let out = std::process::Command::new("osascript")
        .args(["-l", "JavaScript", "-"])
        .env("JARVIS_OCR_PATH", path)
        .env("JARVIS_OCR_W", w.to_string())
        .env("JARVIS_OCR_H", h.to_string())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .as_mut()
                .expect("stdin открыт")
                .write_all(VISION_SCRIPT.as_bytes())?;
            let out = child.wait_with_output()?;
            if out.status.success() {
                Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
            } else {
                let err = String::from_utf8_lossy(&out.stderr);
                Err(std::io::Error::other(
                    err.lines()
                        .find(|l| !l.trim().is_empty())
                        .unwrap_or("osascript завершился с ошибкой"),
                ))
            }
        })
        .map_err(|e| format!("osascript OCR: {e}"))?;
    if let Some(rest) = out.strip_prefix("OCR-ERR") {
        return Err(format!("Vision OCR не удался: {}", rest.trim()));
    }
    parse_ocr_output(&out)
        .map(|(_, lines)| lines)
        .ok_or_else(|| {
            format!(
                "OCR вернул непонятный ответ: {}",
                out.chars().take(200).collect::<String>()
            )
        })
}

// =========================================================================
// Linux: tesseract CLI с TSV-выдачей (опциональная зависимость)
// =========================================================================

#[cfg(target_os = "linux")]
fn tesseract_ocr(path: &str) -> Result<Vec<OcrLine>, String> {
    // Русский и английский сразу; если языкового пакета нет — повторяем с
    // языком по умолчанию, чтобы не читать «установите rus» как тупик.
    let run = |langs: Option<&str>| -> Result<String, String> {
        let mut cmd = std::process::Command::new("tesseract");
        cmd.arg(path).arg("stdout").arg("tsv");
        if let Some(l) = langs {
            cmd.arg("-l").arg(l);
        }
        let out = cmd.output().map_err(|_| {
            "tesseract не установлен: sudo apt install tesseract-ocr tesseract-ocr-rus".to_string()
        })?;
        if out.status.success() {
            Ok(String::from_utf8_lossy(&out.stdout).to_string())
        } else {
            Err(String::from_utf8_lossy(&out.stderr)
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("tesseract завершился с ошибкой")
                .trim()
                .to_string())
        }
    };
    let tsv = match run(Some("rus+eng")) {
        Ok(t) => t,
        Err(e) if e.contains("Failed to load language") || e.contains("rus") => run(None)?,
        Err(e) => return Err(e),
    };
    let lines = parse_tesseract_tsv(&tsv);
    if lines.is_empty() {
        return Err("tesseract не нашёл текста на экране (пустой или графический экран)".into());
    }
    Ok(lines)
}

/// Разбор TSV tesseract: слова уровня 5 группируются в строки по
/// (block, par, line) — прямоугольник строки это объединение прямоугольников
/// слов, текст — слова через пробел. Слова с низкой уверенностью
/// (< 30%) выбрасываются: это обычно мусорная разметка.
pub fn parse_tesseract_tsv(tsv: &str) -> Vec<OcrLine> {
    use std::collections::BTreeMap;
    /// Слово из TSV: left, top, width, height, текст.
    type TsvWord = (i32, i32, i32, i32, String);
    /// Ключ строки: (block, par, line) — BTreeMap сохраняет порядок чтения.
    type TsvRows = BTreeMap<(u32, u32, u32), Vec<TsvWord>>;
    let mut rows: TsvRows = BTreeMap::new();
    for (i, raw) in tsv.lines().enumerate() {
        if i == 0 {
            continue; // заголовок
        }
        let cols: Vec<&str> = raw.split('\t').collect();
        // level page block par line word left top width height conf text
        if cols.len() < 12 {
            continue;
        }
        let level = cols[0].trim().parse::<u32>().unwrap_or(0);
        if level != 5 {
            continue; // слова: уровень 5 (4 — строка без координат по словам)
        }
        let conf = cols[10].trim().parse::<f32>().unwrap_or(0.0);
        let text = cols[11].trim().to_string();
        if conf < 30.0 || text.is_empty() {
            continue;
        }
        let key = (
            cols[2].trim().parse().unwrap_or(0),
            cols[3].trim().parse().unwrap_or(0),
            cols[4].trim().parse().unwrap_or(0),
        );
        let word = (
            cols[6].trim().parse().unwrap_or(0),
            cols[7].trim().parse().unwrap_or(0),
            cols[8].trim().parse().unwrap_or(0),
            cols[9].trim().parse().unwrap_or(0),
            text,
        );
        rows.entry(key).or_default().push(word);
    }
    rows.into_values()
        .filter(|words| !words.is_empty())
        .map(|words| {
            let (x0, y0) = (words[0].0, words[0].1);
            let x1 = words.iter().map(|w| w.0 + w.2).max().unwrap_or(x0);
            let y1 = words.iter().map(|w| w.1 + w.3).max().unwrap_or(y0);
            let (x0, y0) = words
                .iter()
                .fold((x0, y0), |(ax, ay), w| (ax.min(w.0), ay.min(w.1)));
            OcrLine {
                text: words
                    .iter()
                    .map(|w| w.4.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
                x: x0,
                y: y0,
                w: x1 - x0,
                h: y1 - y0,
            }
        })
        .collect()
}

// =========================================================================
// Общие утилиты
// =========================================================================

/// Размер PNG в пикселях из заголовка IHDR: подпись 8 байт, длина 4,
/// "IHDR" 4, ширина 4 (big-endian, смещение 16), высота 4 (смещение 20).
/// macOS нужен размер для пересчёта Retina-масштаба; чтение без крейта
/// image — 24 байта заголовка вместо полной декодировки PNG.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn png_dimensions(path: &str) -> Option<(u32, u32)> {
    use std::io::Read;
    let mut head = [0u8; 24];
    let mut f = std::fs::File::open(path).ok()?;
    f.read_exact(&mut head).ok()?;
    const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    if head[..8] != SIGNATURE || &head[12..16] != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes([head[16], head[17], head[18], head[19]]);
    let h = u32::from_be_bytes([head[20], head[21], head[22], head[23]]);
    (w > 0 && h > 0).then_some((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scale_and_lines() {
        let out = "OCR-OK\t0.5\n10\t20\t100\t30\tПривет мир\n0\t60\t80\t20\tВторая строка\n";
        let (scale, lines) = parse_ocr_output(out).expect("разбор");
        assert!((scale - 0.5).abs() < 1e-6);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "Привет мир");
        assert_eq!((lines[0].x, lines[0].y), (10, 20));
    }

    #[test]
    fn maps_scaled_coords_back_to_physical() {
        // Скриншот 2000 px ужат до 1000 (scale 0.5): OCR дал x=500 → 1000 px.
        let lines = vec![OcrLine {
            text: "кнопка".into(),
            x: 500,
            y: 250,
            w: 100,
            h: 20,
        }];
        let inv = 1.0 / 0.5;
        let mapped: Vec<OcrLine> = lines
            .into_iter()
            .map(|l| OcrLine {
                text: l.text,
                x: ((l.x as f32) * inv).round() as i32,
                y: ((l.y as f32) * inv).round() as i32,
                w: ((l.w as f32) * inv).round() as i32,
                h: ((l.h as f32) * inv).round() as i32,
            })
            .collect();
        assert_eq!((mapped[0].x, mapped[0].y), (1000, 500));
        assert_eq!((mapped[0].w, mapped[0].h), (200, 40));
    }

    #[test]
    fn rejects_output_without_ok_marker() {
        assert!(parse_ocr_output("мусор без маркера").is_none());
    }

    #[test]
    fn formats_numbered_lines_with_coords() {
        let lines = vec![OcrLine {
            text: "Сохранить".into(),
            x: 12,
            y: 34,
            w: 80,
            h: 16,
        }];
        let text = format_lines(&lines);
        assert!(text.contains("[1] «Сохранить» @ (12, 34) 80×16"));
    }

    #[test]
    fn tesseract_tsv_groups_words_into_lines() {
        // Заголовок + слова одной строки (block 1) и мусор с низкой
        // уверенностью (выбрасывается).
        let tsv = "level\tpage\tblock\tpar\tline\tword\tleft\ttop\twidth\theight\tconf\ttext\n\
                   5\t1\t1\t1\t1\t0\t10\t20\t40\t14\t90\tСохранить\n\
                   5\t1\t1\t1\t1\t1\t60\t20\t30\t14\t90\tфайл\n\
                   5\t1\t1\t2\t1\t0\t10\t60\t50\t14\t-1\tмусор\n";
        let lines = parse_tesseract_tsv(tsv);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Сохранить файл");
        assert_eq!(
            (lines[0].x, lines[0].y, lines[0].w, lines[0].h),
            (10, 20, 80, 14)
        );
    }

    #[test]
    fn tesseract_tsv_keeps_reading_order_across_blocks() {
        let tsv = "level\tpage\tblock\tpar\tline\tword\tleft\ttop\twidth\theight\tconf\ttext\n\
                   5\t1\t2\t1\t1\t0\t10\t80\t30\t14\t95\tВнизу\n\
                   5\t1\t1\t1\t1\t0\t10\t20\t30\t14\t95\tСверху\n";
        let lines = parse_tesseract_tsv(tsv);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "Сверху");
        assert_eq!(lines[1].text, "Внизу");
    }

    #[test]
    fn png_dimensions_reads_ihdr() {
        let mut tmp = std::env::temp_dir();
        tmp.push("jarvis_ocr_test.png");
        std::fs::write(
            &tmp,
            [
                0x89u8, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, // подпись
                0, 0, 0, 13, b'I', b'H', b'D', b'R', // длина + тип
                0, 0, 0x10, 0x00, // ширина 4096
                0, 0, 0x08, 0x80, // высота 2176
            ],
        )
        .expect("запись тестового png");
        assert_eq!(png_dimensions(tmp.to_str().unwrap()), Some((4096, 2176)));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn png_dimensions_rejects_non_png() {
        assert_eq!(png_dimensions("/dev/null"), None);
    }
}
