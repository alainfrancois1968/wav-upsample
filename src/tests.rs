use std::f64::consts::PI;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use hound::{SampleFormat, WavSpec, WavWriter};

use crate::design::{
    design_prototype, output_frames, rational_ratio, response_db, Phase, QualitySpec, Window,
};
use crate::dither::{DitherMode, Quantizer};
use crate::resample::{render_channel, Polyphase};
use crate::wavio::{convert_file, Settings};

fn spec(window: Window, phase: Phase, quality: QualitySpec) -> Settings {
    Settings {
        rate: 96_000,
        bits: 24,
        window,
        phase,
        quality,
        dither: DitherMode::Tpdf,
        headroom_db: 0.0,
    }
}

fn render_all(
    input: &[f64],
    fs_in: u32,
    fs_out: u32,
    quality: QualitySpec,
    phase: Phase,
) -> (Vec<f64>, crate::design::DesignInfo) {
    let proto = design_prototype(fs_in, fs_out, Window::Kaiser, phase, quality).unwrap();
    let info = proto.info.clone();
    let poly = Polyphase::build(&proto);
    let n = output_frames(input.len(), fs_in, fs_out);
    let y = render_channel(input, &poly, 0, n);
    (y, info)
}

fn rms(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}

fn sine(fs: u32, freq: f64, n: usize, amp: f64) -> Vec<f64> {
    (0..n)
        .map(|i| amp * (2.0 * PI * freq * i as f64 / fs as f64).sin())
        .collect()
}

fn bin_amplitude(data: &[f64], k: usize) -> f64 {
    let n = data.len();
    let w = 2.0 * PI * (k as f64) / n as f64;
    let mut re = 0.0;
    let mut im = 0.0;
    for (i, &x) in data.iter().enumerate() {
        let a = w * i as f64;
        re += x * a.cos();
        im -= x * a.sin();
    }
    2.0 * (re * re + im * im).sqrt() / n as f64
}

#[test]
fn ratios_are_reduced() {
    assert_eq!(rational_ratio(44_100, 48_000), (160, 147));
    assert_eq!(rational_ratio(48_000, 44_100), (147, 160));
    assert_eq!(rational_ratio(44_100, 96_000), (320, 147));
    assert_eq!(rational_ratio(44_100, 192_000), (640, 147));
    assert_eq!(rational_ratio(48_000, 96_000), (2, 1));
    assert_eq!(rational_ratio(48_000, 192_000), (4, 1));
    assert_eq!(rational_ratio(96_000, 192_000), (2, 1));
    assert_eq!(rational_ratio(96_000, 48_000), (1, 2));
}

#[test]
fn kaiser_prototype_rejects_the_stopband() {
    let proto = design_prototype(
        48_000,
        96_000,
        Window::Kaiser,
        Phase::Linear,
        QualitySpec::high(),
    )
    .unwrap();
    let dc = response_db(&proto.h, 0.0);
    assert!(dc.abs() < 0.01, "DC response {dc} dB");
    // High rate is L·fs_in = 96 kHz. 30 kHz is above the 24 kHz input Nyquist.
    let stop = response_db(&proto.h, 30_000.0 / 96_000.0);
    assert!(
        stop < -100.0,
        "stopband at 30 kHz is only {stop:.1} dB; wanted < -100"
    );
    let pass = response_db(&proto.h, 1_000.0 / 96_000.0);
    assert!(pass.abs() < 0.05, "1 kHz response {pass} dB");
}

#[test]
fn dc_gain_is_unity_on_the_awkward_ratio() {
    let n = 8_000;
    let input = vec![0.25; n];
    let (y, info) = render_all(&input, 44_100, 48_000, QualitySpec::short(), Phase::Linear);
    let guard = (info.taps_per_phase * 48_000 / 44_100) + 32;
    let mid = &y[guard..y.len() - guard];
    let mean = mid.iter().sum::<f64>() / mid.len() as f64;
    assert!(
        (mean - 0.25).abs() < 1e-6,
        "DC mean {mean} (taps {})",
        info.taps_per_phase
    );
}

#[test]
fn two_x_impulse_lands_on_time() {
    let mut input = vec![0.0; 4_000];
    input[2_000] = 1.0;
    let (y, _) = render_all(&input, 48_000, 96_000, QualitySpec::high(), Phase::Linear);
    let (peak_i, peak_v) = y
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert!(
        (peak_i as isize - 4_000).abs() <= 1,
        "peak at {peak_i} value {peak_v}, expected ~4000"
    );
    assert!(*peak_v > 0.5, "peak {peak_v} is unexpectedly small");
}

#[test]
fn fractional_ratio_sine_gain() {
    let fs_in = 44_100;
    let fs_out = 48_000u32;
    let n = (fs_in as f64 * 0.30) as usize;
    let input = sine(fs_in, 1_000.0, n, 0.5);
    let (y, info) = render_all(&input, fs_in, fs_out, QualitySpec::high(), Phase::Linear);
    let guard = info.taps_per_phase * 3;
    let in_rms = rms(&input[guard..input.len() - guard]);
    let out_guard = guard * fs_out as usize / fs_in as usize + 64;
    let out_rms = rms(&y[out_guard..y.len() - out_guard]);
    let db = 20.0 * (out_rms / in_rms).log10();
    assert!(
        db.abs() < 0.02,
        "1 kHz gain is {db:.4} dB (taps {})",
        info.taps_per_phase
    );
}

#[test]
fn images_are_rejected_on_integer_upsample() {
    let fs_in = 48_000u32;
    let fs_out = 96_000u32;
    let fft_n = 8_192usize;
    let freq = 12_000.0;
    let guard_out = 6_000usize;
    let out_len = guard_out + fft_n + guard_out;
    let in_len = out_len / 2 + 8;
    let input = sine(fs_in, freq, in_len, 0.5);
    let (y, _) = render_all(&input, fs_in, fs_out, QualitySpec::high(), Phase::Linear);
    assert!(y.len() > guard_out + fft_n);
    let slice = &y[guard_out..guard_out + fft_n];
    let k_fund = (freq * fft_n as f64 / fs_out as f64).round() as usize;
    let k_image = ((fs_in as f64 - freq) * fft_n as f64 / fs_out as f64).round() as usize;
    let fund = bin_amplitude(slice, k_fund);
    let image = bin_amplitude(slice, k_image);
    let reject = 20.0 * (fund / image.max(1e-30)).log10();
    assert!(
        (fund - 0.5).abs() < 0.01,
        "fundamental amplitude {fund}, expected 0.5"
    );
    assert!(
        reject > 110.0,
        "image rejection only {reject:.1} dB (image amp {image:.3e})"
    );
}

#[test]
fn downsampling_kills_energy_above_the_new_nyquist() {
    let fs_in = 96_000u32;
    let n = 16_000;
    let input = sine(fs_in, 30_000.0, n, 0.5);
    let (y, info) = render_all(&input, fs_in, 48_000, QualitySpec::high(), Phase::Linear);
    let guard = info.taps_per_phase * 2;
    let mid = &y[guard..y.len() - guard];
    let db = 20.0 * (rms(mid) / (0.5 / 2.0_f64.sqrt())).log10();
    assert!(
        db < -100.0,
        "30 kHz alias/leakage at 48 kHz output is {db:.1} dB"
    );
}

#[test]
fn minimum_phase_has_the_same_magnitude_and_no_prering_dominance() {
    let linear = design_prototype(
        48_000,
        96_000,
        Window::Kaiser,
        Phase::Linear,
        QualitySpec::short(),
    )
    .unwrap();
    let mini = design_prototype(
        48_000,
        96_000,
        Window::Kaiser,
        Phase::Minimum,
        QualitySpec::short(),
    )
    .unwrap();
    for hz in [0.0, 100.0, 1_000.0, 8_000.0] {
        let a = response_db(&linear.h, hz / 96_000.0);
        let b = response_db(&mini.h, hz / 96_000.0);
        assert!(
            (a - b).abs() < 0.2,
            "{hz} Hz linear {a:.2} dB vs minimum {b:.2} dB"
        );
    }
    let peak = mini.info.delay_hr;
    assert!(
        peak < mini.h.len() / 5,
        "minimum-phase peak at {peak} / {} is not near the start",
        mini.h.len()
    );
    let before: f64 = mini.h[..peak].iter().map(|v| v * v).sum();
    let after: f64 = mini.h[peak + 1..].iter().map(|v| v * v).sum();
    assert!(
        before < after,
        "minimum-phase energy before the peak ({before}) exceeds the tail ({after})"
    );
}

#[test]
fn minimum_phase_resample_is_causal() {
    let mut input = vec![0.0; 4_000];
    input[2_000] = 1.0;
    let (y, info) = render_all(&input, 48_000, 96_000, QualitySpec::short(), Phase::Minimum);
    let (peak_i, _) = y
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let expected = 4_000.0; // 2× the impulse index
    assert!(
        (peak_i as f64 - expected).abs() < 8.0,
        "min-phase peak at {peak_i}, expected near {expected} (delay_hr {})",
        info.delay_hr
    );
    let pre: f64 = y[..peak_i.saturating_sub(2)].iter().map(|v| v.abs()).sum();
    let post: f64 = y[peak_i + 2..].iter().map(|v| v.abs()).sum();
    assert!(
        pre < post,
        "min-phase output has more absolute mass before the peak ({pre}) than after ({post})"
    );
}

#[test]
fn gauss_passband_is_flat_and_finite() {
    let proto = design_prototype(
        44_100,
        96_000,
        Window::Gauss,
        Phase::Linear,
        QualitySpec::high(),
    )
    .unwrap();
    assert!(proto.h.iter().all(|c| c.is_finite()));
    let dc = response_db(&proto.h, 0.0);
    assert!(dc.abs() < 0.05, "gauss DC {dc}");
}

#[test]
fn quantizer_exact_codes_and_tpdf_on_silence() {
    let mut q = Quantizer::new(DitherMode::None, 16, 0, 44_100);
    assert_eq!(q.quantize(0.5), 16_384);
    assert_eq!(q.quantize(0.0), 0);
    assert_eq!(q.clips, 0);
    // +1.0 sits one LSB past the largest positive code.
    assert_eq!(q.quantize(1.0), 32_767);
    assert!(q.clips >= 1);

    let mut d = Quantizer::new(DitherMode::Tpdf, 16, 1, 96_000);
    let mut nonzero = 0;
    let mut sum = 0i64;
    for _ in 0..4_000 {
        let s = d.quantize(0.0) as i64;
        assert!((-2..=2).contains(&s), "undithered-looking code {s}");
        if s != 0 {
            nonzero += 1;
        }
        sum += s;
    }
    assert!(nonzero > 100, "TPDF of silence produced almost no toggles");
    assert!(sum.abs() < 400, "TPDF of silence is biased: sum {sum}");
    assert_eq!(d.clips, 0);
}

#[test]
fn shaped_dither_stays_finite_on_a_sine() {
    let mut q = Quantizer::new(DitherMode::Shaped, 16, 0, 44_100);
    for i in 0..8_000 {
        let x = 0.4 * (2.0 * PI * 440.0 * i as f64 / 44_100.0).sin();
        let s = q.quantize(x);
        assert!((-32_768..=32_767).contains(&s));
    }
    assert_eq!(q.clips, 0);
}

fn temp_wav(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("wav-upsample-{name}-{nanos}.wav"))
}

fn write_i16(path: &std::path::Path, rate: u32, chans: Vec<Vec<f64>>) {
    let spec = WavSpec {
        channels: chans.len() as u16,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec).unwrap();
    let n = chans[0].len();
    for i in 0..n {
        for ch in &chans {
            let code = (ch[i] * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
            w.write_sample(code).unwrap();
        }
    }
    w.finalize().unwrap();
}

#[test]
fn wav_roundtrip_fractional_upsample() {
    let input_path = temp_wav("in");
    let output_path = temp_wav("out");
    let n = 12_000;
    let left = sine(44_100, 1_000.0, n, 0.4);
    let right = sine(44_100, 1_000.0, n, 0.2);
    write_i16(&input_path, 44_100, vec![left, right]);

    let mut settings = spec(Window::Kaiser, Phase::Linear, QualitySpec::high());
    settings.rate = 96_000;
    settings.bits = 24;
    settings.dither = DitherMode::Tpdf;
    let report = convert_file(&input_path, &output_path, &settings).unwrap();
    assert_eq!(report.frames_in, n as u64);
    assert_eq!(report.spec_in.channels, 2);
    assert_eq!(report.clipped, 0, "unexpected clips, peak {}", report.peak);

    let mut reader = hound::WavReader::open(&output_path).unwrap();
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 96_000);
    assert_eq!(spec.bits_per_sample, 24);
    assert_eq!(spec.channels, 2);
    let samples: Vec<i32> = reader.samples::<i32>().map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), report.frames_out as usize * 2);
    let mut acc = 0.0;
    let mut count = 0.0;
    let guard = 8_000usize;
    let total_frames = report.frames_out as usize;
    let end = total_frames.saturating_sub(guard);
    for frame in samples.chunks(2).take(end).skip(guard) {
        let l = frame[0] as f64 / 8_388_608.0;
        acc += l * l;
        count += 1.0;
    }
    let out_rms = (acc / count).sqrt();
    let expect = 0.4 / 2.0_f64.sqrt();
    let db = 20.0 * (out_rms / expect).log10();
    assert!(db.abs() < 0.05, "file RMS error {db:.3} dB (rms {out_rms})");
    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn extreme_192k_sine_does_not_diverge() {
    let n = 8_000;
    let input = sine(44_100, 1_000.0, n, 0.25);
    let (y, info) = render_all(
        &input,
        44_100,
        192_000,
        QualitySpec::extreme(),
        Phase::Linear,
    );
    assert!(
        info.taps_per_phase > 800,
        "extreme filter is suspiciously short"
    );
    assert!(y.iter().all(|v| v.is_finite()));
    let guard = info.taps_per_phase * 192_000 / 44_100;
    assert!(y.len() > guard * 2 + 100);
    let db = 20.0 * (rms(&y[guard..y.len() - guard]) / rms(&input)).log10();
    assert!(db.abs() < 0.05, "extreme 192 kHz gain {db:.3} dB");
}
