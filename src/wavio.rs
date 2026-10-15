//! WAV read / write. Samples are normalised to ±1, where 1.0 is one count
//! past the largest positive integer (the usual 2^(bits−1) divisor). 8, 16,
//! 24 and 32-bit integer PCM and 32-bit float are accepted. Output is 16 or
//! 24-bit integer PCM only.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use rayon::prelude::*;

use crate::design::{design_prototype, output_frames, DesignInfo, Phase, QualitySpec, Window};
use crate::dither::{DitherMode, Quantizer};
use crate::resample::{render_channel, Polyphase};

#[derive(Clone, Debug)]
pub struct Settings {
    pub rate: u32,
    pub bits: u16,
    pub window: Window,
    pub phase: Phase,
    pub quality: QualitySpec,
    pub dither: DitherMode,
    /// Applied as a linear gain before quantising. 1.0 dB ≈ ×0.891.
    /// Useful when a long filter's ringing pushes intersample peaks over 0 dBFS.
    pub headroom_db: f64,
}

pub struct Report {
    pub spec_in: WavSpec,
    pub frames_in: u64,
    pub frames_out: u64,
    pub design: Option<DesignInfo>,
    pub peak: f64,
    pub samples_over_full_scale: u64,
    pub clipped: u64,
    pub shaper: String,
}

pub fn read_wav(path: &Path) -> Result<(WavSpec, Vec<Vec<f64>>)> {
    let mut reader =
        WavReader::open(path).with_context(|| format!("could not open {}", path.display()))?;
    let spec = reader.spec();
    let ch = spec.channels as usize;
    if ch == 0 {
        bail!("WAV has zero channels");
    }
    let mut planar = vec![Vec::new(); ch];

    match spec.sample_format {
        SampleFormat::Float => {
            if spec.bits_per_sample != 32 {
                bail!(
                    "only 32-bit float WAV is supported (file is {}-bit float)",
                    spec.bits_per_sample
                );
            }
            for (i, sample) in reader.samples::<f32>().enumerate() {
                let s = sample.context("truncated float WAV")? as f64;
                planar[i % ch].push(s);
            }
        }
        SampleFormat::Int => match spec.bits_per_sample {
            8 => {
                for (i, sample) in reader.samples::<i8>().enumerate() {
                    let s = sample.context("truncated WAV")? as f64 / 128.0;
                    planar[i % ch].push(s);
                }
            }
            16 => {
                for (i, sample) in reader.samples::<i16>().enumerate() {
                    let s = sample.context("truncated WAV")? as f64 / 32768.0;
                    planar[i % ch].push(s);
                }
            }
            24 | 32 => {
                let denom = if spec.bits_per_sample == 24 {
                    8_388_608.0
                } else {
                    2_147_483_648.0
                };
                for (i, sample) in reader.samples::<i32>().enumerate() {
                    let s = sample.context("truncated WAV")? as f64 / denom;
                    planar[i % ch].push(s);
                }
            }
            bits => bail!("unsupported integer bit depth {bits}"),
        },
    }

    let frames = planar.iter().map(|c| c.len()).min().unwrap_or(0);
    for c in &mut planar {
        c.truncate(frames);
    }
    if frames == 0 {
        bail!("WAV contains no sample frames");
    }
    Ok((spec, planar))
}

pub fn convert_file(input: &Path, output: &Path, settings: &Settings) -> Result<Report> {
    if settings.bits != 16 && settings.bits != 24 {
        bail!("output bit depth must be 16 or 24");
    }
    if !matches!(settings.rate, 44_100 | 48_000 | 96_000 | 192_000) {
        bail!("output rate must be 44100, 48000, 96000 or 192000");
    }
    if !settings.headroom_db.is_finite() || !(-12.0..=48.0).contains(&settings.headroom_db) {
        bail!("headroom must be between -12 and 48 dB");
    }

    let (spec_in, channels) = read_wav(input)?;
    let frames_in = channels[0].len();
    let fs_in = spec_in.sample_rate;
    let frames_out = output_frames(frames_in, fs_in, settings.rate);

    let bytes = frames_out as u64 * channels.len() as u64 * (settings.bits as u64 / 8) + 128;
    if bytes >= 4_000_000_000 {
        bail!(
            "output would be {:.2} GB, past the classic 4 GB WAV limit. \
             Split the source or pick a lower rate / bit depth.",
            bytes as f64 / 1e9
        );
    }

    let (plan, design) = if fs_in == settings.rate {
        (None, None)
    } else {
        let proto = design_prototype(
            fs_in,
            settings.rate,
            settings.window,
            settings.phase,
            settings.quality,
        )?;
        let info = proto.info.clone();
        (Some(Polyphase::build(&proto)), Some(info))
    };

    let spec_out = WavSpec {
        channels: channels.len() as u16,
        sample_rate: settings.rate,
        bits_per_sample: settings.bits,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::new(BufWriter::new(File::create(output)?), spec_out)
        .with_context(|| format!("could not create {}", output.display()))?;

    let gain = 10f64.powf(-settings.headroom_db / 20.0);
    let mut quantizers: Vec<Quantizer> = (0..channels.len())
        .map(|ch| Quantizer::new(settings.dither, settings.bits, ch, settings.rate))
        .collect();
    let shaper = Quantizer::shaper_name(settings.dither, settings.rate);

    let mut peak = 0.0f64;
    let mut overs = 0u64;
    let block = 65_536usize;

    for start in (0..frames_out).step_by(block) {
        let end = (start + block).min(frames_out);
        let rendered: Vec<Vec<f64>> = match &plan {
            None => channels.iter().map(|ch| ch[start..end].to_vec()).collect(),
            Some(poly) => channels
                .par_iter()
                .map(|ch| render_channel(ch, poly, start, end))
                .collect(),
        };
        let n = end - start;
        for i in 0..n {
            for (q, ch) in quantizers.iter_mut().zip(rendered.iter()) {
                let s = ch[i] * gain;
                let a = s.abs();
                if a > peak {
                    peak = a;
                }
                if a > 1.0 {
                    overs += 1;
                }
                let code = q.quantize(s);
                if settings.bits == 16 {
                    writer.write_sample(code as i16)?;
                } else {
                    writer.write_sample(code)?;
                }
            }
        }
    }

    let clipped: u64 = quantizers.iter().map(|q| q.clips).sum();
    writer.finalize()?;

    Ok(Report {
        spec_in,
        frames_in: frames_in as u64,
        frames_out: frames_out as u64,
        design,
        peak,
        samples_over_full_scale: overs,
        clipped,
        shaper,
    })
}
