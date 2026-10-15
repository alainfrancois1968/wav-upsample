//! `wav-upsample` — offline polyphase sinc resampler.

#![deny(unsafe_code)]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
use hound::SampleFormat;
use wav_upsample::design::{self, Phase, QualitySpec, Window};
use wav_upsample::dither::DitherMode;
use wav_upsample::wavio::{self, Settings};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FilterOpt {
    /// Steep Kaiser-windowed sinc. Closest thing here to a long poly-sinc-ext.
    Kaiser,
    /// Gaussian-windowed sinc. Short ringing, gentler rolloff, à la poly-sinc-gauss.
    Gauss,
    /// 4-term Blackman-Harris. Classic window, stopband floor near −92 dB.
    BlackmanHarris,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PhaseOpt {
    /// Symmetric impulse. No phase distortion, equal pre- and post-ring.
    Linear,
    /// Cepstrum conversion of the same magnitude. No pre-ring.
    Minimum,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum QualityOpt {
    /// ~90% of Nyquist, ~100 dB design. Fast, less ringing.
    Short,
    /// ~95.5% of Nyquist, ~140 dB design. The default. About 24-bit clean.
    High,
    /// ~98.5% of Nyquist, ~170 dB design. Long. Linear phase rings for a while.
    Extreme,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DitherOpt {
    /// Round to nearest. Fine for a 24-bit archival intermediate, poor for 16-bit.
    None,
    /// Triangular dither, ±1 LSB. Correct at every sample rate.
    Tpdf,
    /// Noise-shaped TPDF. Wannamaker F-weight at 44.1 kHz, first-order elsewhere.
    Shaped,
}

#[derive(Parser, Debug)]
#[command(
    name = "wav-upsample",
    version,
    about = "High-quality offline WAV resampler. Polyphase windowed-sinc, Rayon-parallel."
)]
struct Args {
    /// Input WAV (16/24/32-bit PCM or 32-bit float).
    #[arg(short, long)]
    input: PathBuf,

    /// Output WAV. Ignored with --info.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Output sample rate: 44100, 48000, 96000, 192000 (also 44.1, 48, 96, 192).
    #[arg(short, long, value_parser = parse_rate)]
    rate: u32,

    /// Output bit depth, 16 or 24.
    #[arg(short, long, value_parser = parse_bits)]
    bits: u16,

    /// Steep Kaiser-windowed sinc. Closest thing here to a long poly-sinc-ext.
    #[arg(long, value_enum, default_value = "kaiser")]
    filter: FilterOpt,

    /// linear keeps pre-ring; minimum removes it.
    #[arg(long, value_enum, default_value = "linear")]
    phase: PhaseOpt,

    /// short, high (default), or extreme.
    #[arg(long, value_enum, default_value = "high")]
    quality: QualityOpt,

    /// none, tpdf (default), or shaped.
    #[arg(long, value_enum, default_value = "tpdf")]
    dither: DitherOpt,

    /// Passband edge as a fraction of the lower Nyquist (0.50–0.995).
    /// Overrides the quality preset's bandwidth but not its attenuation.
    #[arg(long)]
    bandwidth: Option<f64>,

    /// Kaiser design stopband in dB (40–200). Sets the filter length for every window.
    #[arg(long)]
    atten: Option<f64>,

    /// Gain cut applied before quantising, in dB. Try 1 to 3 if `clipped` is non-zero.
    #[arg(long, default_value_t = 0.0)]
    headroom: f64,

    /// Rayon worker threads. 0 uses every core.
    #[arg(short = 'j', long, default_value_t = 0)]
    threads: usize,

    /// Print the filter and exit. Does not write audio.
    #[arg(long)]
    info: bool,
}

fn parse_rate(s: &str) -> Result<u32, String> {
    let rate = match s {
        "44.1" | "44.1k" | "44100" => 44_100,
        "48" | "48k" | "48000" => 48_000,
        "96" | "96k" | "96000" => 96_000,
        "192" | "192k" | "192000" => 192_000,
        other => other
            .parse::<u32>()
            .map_err(|_| format!("unknown rate `{s}`"))?,
    };
    if matches!(rate, 44_100 | 48_000 | 96_000 | 192_000) {
        Ok(rate)
    } else {
        Err("output rate must be 44100, 48000, 96000 or 192000".into())
    }
}

fn parse_bits(s: &str) -> Result<u16, String> {
    match s {
        "16" => Ok(16),
        "24" => Ok(24),
        _ => Err("bit depth must be 16 or 24".into()),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()?;
    }

    let mut quality = match args.quality {
        QualityOpt::Short => QualitySpec::short(),
        QualityOpt::High => QualitySpec::high(),
        QualityOpt::Extreme => QualitySpec::extreme(),
    };
    if let Some(bw) = args.bandwidth {
        quality.bandwidth = bw;
    }
    if let Some(atten) = args.atten {
        quality.atten_db = atten;
    }

    let window = match args.filter {
        FilterOpt::Kaiser => Window::Kaiser,
        FilterOpt::Gauss => Window::Gauss,
        FilterOpt::BlackmanHarris => Window::BlackmanHarris,
    };
    let phase = match args.phase {
        PhaseOpt::Linear => Phase::Linear,
        PhaseOpt::Minimum => Phase::Minimum,
    };
    let dither = match args.dither {
        DitherOpt::None => DitherMode::None,
        DitherOpt::Tpdf => DitherMode::Tpdf,
        DitherOpt::Shaped => DitherMode::Shaped,
    };

    if args.info {
        let (spec, channels) = wavio::read_wav(&args.input)?;
        let frames = channels[0].len();
        print_input(&spec, frames);
        if spec.sample_rate == args.rate {
            println!("ratio:     1/1 (bit depth / dither only, no resampling filter)");
        } else {
            let proto =
                design::design_prototype(spec.sample_rate, args.rate, window, phase, quality)?;
            print_design(&proto.info);
            let macs = design::output_frames(frames, spec.sample_rate, args.rate) as f64
                * channels.len() as f64
                * proto.info.taps_per_phase as f64;
            println!("work:      {:.3e} multiply-adds", macs);
        }
        println!(
            "dither:    {}",
            wav_upsample::dither::Quantizer::shaper_name(dither, args.rate)
        );
        return Ok(());
    }

    let output = match args.output {
        Some(p) => p,
        None => bail!("--output is required unless --info is set"),
    };

    let settings = Settings {
        rate: args.rate,
        bits: args.bits,
        window,
        phase,
        quality,
        dither,
        headroom_db: args.headroom,
    };

    let started = Instant::now();
    let report = wavio::convert_file(&args.input, &output, &settings)?;
    let elapsed = started.elapsed();

    print_input(&report.spec_in, report.frames_in as usize);
    let direction = match report.spec_in.sample_rate.cmp(&args.rate) {
        std::cmp::Ordering::Less => "upsample",
        std::cmp::Ordering::Equal => "bit-depth only",
        std::cmp::Ordering::Greater => "downsample",
    };
    println!(
        "output:    {} Hz, {}-bit, {} frames ({:.3} s)  [{}]",
        args.rate,
        args.bits,
        report.frames_out,
        report.frames_out as f64 / args.rate as f64,
        direction
    );
    if let Some(info) = &report.design {
        print_design(info);
    } else {
        println!("ratio:     1/1");
    }
    println!("dither:    {}", report.shaper);
    if args.headroom != 0.0 {
        println!("headroom:  {:.2} dB", args.headroom);
    }
    println!("peak:      {}", fmt_db(report.peak));
    println!(
        "over 0 dB: {} samples before quantising",
        report.samples_over_full_scale
    );
    println!("clipped:   {} integer samples", report.clipped);
    let audio_secs = report.frames_out as f64 / args.rate as f64;
    let rt = if elapsed.as_secs_f64() > 0.0 {
        audio_secs / elapsed.as_secs_f64()
    } else {
        0.0
    };
    println!(
        "time:      {:.2} s, {:.1}× realtime",
        elapsed.as_secs_f64(),
        rt
    );
    println!("wrote:     {}", output.display());
    if report.clipped > 0 {
        println!("note:      clipping is not hidden. Re-run with --headroom 1 (or more).");
    }
    Ok(())
}

fn print_input(spec: &hound::WavSpec, frames: usize) {
    let kind = match spec.sample_format {
        SampleFormat::Float => "float",
        SampleFormat::Int => "int",
    };
    let secs = if spec.sample_rate == 0 {
        0.0
    } else {
        frames as f64 / spec.sample_rate as f64
    };
    println!(
        "input:     {} Hz, {} ch, {}-bit {kind}, {frames} frames ({secs:.3} s)",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );
}

fn print_design(info: &design::DesignInfo) {
    let window = match info.window {
        Window::Kaiser => "kaiser",
        Window::Gauss => "gauss",
        Window::BlackmanHarris => "blackman-harris",
    };
    let phase = match info.phase {
        Phase::Linear => "linear",
        Phase::Minimum => "minimum",
    };
    println!(
        "ratio:     {}/{}   filter: {window}, {phase} phase",
        info.l, info.m
    );
    println!(
        "passband:  {:.1} Hz   stopband starts {:.1} Hz   transition {:.1} Hz",
        info.passband_hz, info.stopband_hz, info.transition_hz
    );
    println!(
        "length:    {} taps/output sample, prototype {}   design atten {:.0} dB",
        info.taps_per_phase, info.prototype_len, info.atten_db
    );
    println!(
        "aligned:   {:.3} input samples of filter delay removed",
        info.delay_hr as f64 / info.l as f64
    );
}

fn fmt_db(peak: f64) -> String {
    if peak <= 0.0 || !peak.is_finite() {
        "-inf dBFS".to_string()
    } else {
        format!("{:.2} dBFS", 20.0 * peak.log10())
    }
}
