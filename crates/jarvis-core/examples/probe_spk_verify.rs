//! Проверка спикер-энкодера: наш эмбеддинг jett.wav vs эталонный .npy Zyphra.
use jarvis_core::tts_zonos::speaker::SpeakerEncoder;

fn read_wav(path: &str) -> anyhow::Result<(u32, Vec<f32>)> {
    let data = std::fs::read(path)?;
    anyhow::ensure!(&data[0..4] == b"RIFF", "не wav");
    let mut pos = 12usize;
    let (mut rate, mut samples) = (0u32, Vec::new());
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes([data[pos+4], data[pos+5], data[pos+6], data[pos+7]]) as usize;
        let body = &data[pos+8..pos+8+size];
        if id == b"fmt " { rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]); }
        else if id == b"data" {
            for c in body.chunks_exact(2) { samples.push(i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0); }
        }
        pos += 8 + size + (size & 1);
    }
    Ok((rate, samples))
}

fn read_npy(path: &str) -> anyhow::Result<Vec<f32>> {
    let data = std::fs::read(path)?;
    let hlen = u16::from_le_bytes([data[8], data[9]]) as usize;
    let off = 10 + hlen;
    Ok(data[off..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn main() -> anyhow::Result<()> {
    let wav_path = std::env::args().nth(1).unwrap_or("/tmp/zonos_ref/jett.wav".into());
    let npy_path = std::env::args().nth(2).unwrap_or("/tmp/zonos_ref/jett.embedding.npy".into());
    let (rate, wav) = read_wav(&wav_path)?;
    println!("jett.wav: {} Гц, {} сэмплов", rate, wav.len());
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let enc = SpeakerEncoder::open(&dir.join("spk-encoder.gguf"))?;
    let mine = enc.embedding(&wav, rate)?;
    let theirs = read_npy(&npy_path)?;
    println!("наш: {} изм., эталон: {} изм.", mine.len(), theirs.len());
    let dot: f32 = mine.iter().zip(theirs.iter()).map(|(a, b)| a * b).sum();
    let nm = (mine.iter().map(|x| x * x).sum::<f32>()).sqrt();
    let nt = (theirs.iter().map(|x| x * x).sum::<f32>()).sqrt();
    println!("косинус: {:.4}  |наш|={nm:.4} |этал|={nt:.4}", dot / (nm * nt));
    // Режим --cross: сравнить эмбеддинги двух клипов между собой.
    if std::env::args().nth(3).is_some_and(|a| a == "--cross") {
        let (r2, w2) = read_wav(std::env::args().nth(4).unwrap().as_str())?;
        let emb2 = enc.embedding(&w2, r2)?;
        let dot2: f32 = mine.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        let n2 = (emb2.iter().map(|x| x * x).sum::<f32>()).sqrt();
        println!("перекрёстный косинус {wav_path} ↔ {} = {:.4}", std::env::args().nth(4).unwrap(), dot2 / (nm * n2));
    }
    // Режим --from-clip: посчитать эмбеддинг ДРУГОГО клипа и сравнить с эталоном.
    if let Some(clip) = std::env::args().nth(3).filter(|a| a == "--from-clip").and_then(|_| std::env::args().nth(4)) {
        let (crate_rate, clip_wav) = read_wav(&clip)?;
        let clip_emb = enc.embedding(&clip_wav, crate_rate)?;
        let dot2: f32 = clip_emb.iter().zip(theirs.iter()).map(|(a, b)| a * b).sum();
        let n2 = (clip_emb.iter().map(|x| x * x).sum::<f32>()).sqrt();
        println!("клип {clip}: косинус к эталонному npy = {:.4} (|вектор|={n2:.4})", dot2 / (n2 * nt));
        let dot3: f32 = clip_emb.iter().zip(mine.iter()).map(|(a, b)| a * b).sum();
        println!("клип {clip}: косинус к нашей загрузке первого клипа = {:.4}", dot3 / (n2 * nm));
    }
    let max_diff = mine.iter().zip(theirs.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("max|Δ| = {max_diff:.5}");
    // Первые 8 значений.
    println!("наш    первые 8: {:?}", mine[..8].iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    println!("эталон первые 8: {:?}", theirs[..8].iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    Ok(())
}
