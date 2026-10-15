//! Windowed-sinc prototype design.
//!
//! The resampler is a single rational polyphase FIR: upsample by L, lowpass,
//! decimate by M, with L/M = fs_out/fs_in in lowest terms. One prototype
//! carries the whole transition, which is the same idea as HQPlayer's
//! poly-sinc family (not a cascade of half-band stages plus a leftover ratio).

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::minphase;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    Kaiser,
    Gauss,
    BlackmanHarris,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Linear,
    Minimum,
}

#[derive(Clone, Copy, Debug)]
pub struct QualitySpec {
    /// Passband edge as a fraction of the limiting Nyquist (the lower of the
    /// two rates). The stopband edge sits on that Nyquist.
    pub bandwidth: f64,
    /// Kaiser design attenuation. Gaussian / Blackman-Harris windows do not
    /// hit this number; it still sets the transition length.
    pub atten_db: f64,
    /// Gaussian window alpha (ignored by the other windows).
    pub gauss_alpha: f64,
}

impl QualitySpec {
    pub fn short() -> Self {
        Self {
            bandwidth: 0.90,
            atten_db: 100.0,
            gauss_alpha: 3.0,
        }
    }

    pub fn high() -> Self {
        Self {
            bandwidth: 0.955,
            atten_db: 140.0,
            gauss_alpha: 3.8,
        }
    }

    pub fn extreme() -> Self {
        Self {
            bandwidth: 0.985,
            atten_db: 170.0,
            gauss_alpha: 4.6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DesignInfo {
    pub l: usize,
    pub m: usize,
    pub taps_per_phase: usize,
    pub prototype_len: usize,
    pub bandwidth: f64,
    pub atten_db: f64,
    pub passband_hz: f64,
    pub stopband_hz: f64,
    pub transition_hz: f64,
    /// High-rate samples from the start of a causal convolution to the
    /// linear-phase centre, or to the minimum-phase peak. Dividing by `l`
    /// converts to input samples.
    pub delay_hr: usize,
    pub phase: Phase,
    pub window: Window,
}

#[derive(Clone, Debug)]
pub struct Prototype {
    pub h: Vec<f64>,
    pub info: DesignInfo,
}

pub fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

/// `(L, M)` such that `fs_out / fs_in = L / M` in lowest terms.
pub fn rational_ratio(fs_in: u32, fs_out: u32) -> (usize, usize) {
    let g = gcd_u32(fs_in, fs_out);
    ((fs_out / g) as usize, (fs_in / g) as usize)
}

pub fn output_frames(input_frames: usize, fs_in: u32, fs_out: u32) -> usize {
    let num = input_frames as u128 * fs_out as u128;
    let den = fs_in as u128;
    ((num + den / 2) / den) as usize
}

pub fn design_prototype(
    fs_in: u32,
    fs_out: u32,
    window: Window,
    phase: Phase,
    mut spec: QualitySpec,
) -> Result<Prototype> {
    if fs_in < 1000 || fs_out < 1000 {
        bail!("sample rate {fs_in} -> {fs_out} is not a usable PCM rate");
    }
    if !(0.50..=0.995).contains(&spec.bandwidth) {
        bail!("bandwidth must be between 0.50 and 0.995");
    }
    if !(40.0..=200.0).contains(&spec.atten_db) {
        bail!("attenuation must be between 40 and 200 dB");
    }
    spec.gauss_alpha = spec.gauss_alpha.clamp(1.5, 8.0);

    let (l, m) = rational_ratio(fs_in, fs_out);
    if l > 8192 || m > 8192 {
        bail!("ratio {l}/{m} is too large; this tool targets the 44.1/48 family");
    }

    let limit_hz = (fs_in.min(fs_out) as f64) * 0.5;
    let stopband_hz = limit_hz;
    let passband_hz = limit_hz * spec.bandwidth;
    let transition_hz = (stopband_hz - passband_hz).max(1.0);

    // Length of one polyphase branch, in input samples. Oppenheim/Schafer
    // Kaiser estimate: N ≈ (A − 8) / (2.285 · Δω), Δω in radians/sample
    // at the input rate. The same N is the number of MACs per output sample.
    let delta_cycles = transition_hz / fs_in as f64;
    let delta_omega = 2.0 * std::f64::consts::PI * delta_cycles;
    let n_est = (spec.atten_db - 8.0) / (2.285 * delta_omega);
    let mut taps = n_est.ceil().max(15.0) as usize;
    if taps % 2 == 0 {
        taps += 1;
    }
    if taps > 4095 {
        bail!(
            "this setting wants {taps} taps per output sample \
             (passband {:.1} Hz, {transition_hz:.0} Hz transition, {:.0} dB). \
             Lower --atten or widen the transition with --bandwidth.",
            passband_hz,
            spec.atten_db
        );
    }

    let n = taps
        .checked_mul(l)
        .ok_or_else(|| anyhow::anyhow!("filter length overflow"))?;
    if n > 3_000_000 {
        bail!("prototype length {n} is too large; use a shorter quality preset");
    }

    let fc_in = 0.5 * (passband_hz + stopband_hz) / fs_in as f64;
    // Prototype lives at fs_high = L · fs_in, so the cutoff scales by 1/L.
    let fc_high = fc_in / l as f64;
    let beta = kaiser_beta(spec.atten_db);
    let alpha = spec.gauss_alpha;

    let mut h = vec![0.0; n];
    h.par_iter_mut().enumerate().for_each(|(i, slot)| {
        let mid = (n as f64 - 1.0) / 2.0;
        let x = i as f64 - mid;
        let w = match window {
            Window::Kaiser => kaiser_window(i, n, beta),
            Window::Gauss => gauss_window(i, n, alpha),
            Window::BlackmanHarris => blackman_harris(i, n),
        };
        *slot = sinc_lowpass(x, fc_high) * w;
    });

    normalize_sum(&mut h);

    let delay_hr = if phase == Phase::Minimum {
        eprintln!("minimum-phase conversion of a {n}-tap prototype (this can take a moment)…");
        h = minphase::to_minimum_phase(&h)?;
        normalize_sum(&mut h);
        peak_index(&h)
    } else {
        // Integer sample closest to the linear-phase centre. For even lengths
        // the true centre is halfway between taps; the residual is ≤ 0.5
        // high-rate samples (≤ 0.5/L input samples) and is inaudible.
        ((n as f64 - 1.0) / 2.0).round() as usize
    };

    Ok(Prototype {
        h,
        info: DesignInfo {
            l,
            m,
            taps_per_phase: taps,
            prototype_len: n,
            bandwidth: spec.bandwidth,
            atten_db: spec.atten_db,
            passband_hz,
            stopband_hz,
            transition_hz,
            delay_hr,
            phase,
            window,
        },
    })
}

fn normalize_sum(h: &mut [f64]) {
    let sum: f64 = h.iter().sum();
    if sum.abs() > 1e-18 {
        let inv = 1.0 / sum;
        for c in h.iter_mut() {
            *c *= inv;
        }
    }
}

fn peak_index(h: &[f64]) -> usize {
    h.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Ideal lowpass impulse response. `fc` is the cutoff in cycles/sample
/// (0.5 = Nyquist). `x` is the time index in the same sample units.
pub fn sinc_lowpass(x: f64, fc: f64) -> f64 {
    if x.abs() < 1e-12 {
        2.0 * fc
    } else {
        (2.0 * std::f64::consts::PI * fc * x).sin() / (std::f64::consts::PI * x)
    }
}

pub fn kaiser_beta(atten_db: f64) -> f64 {
    if atten_db > 50.0 {
        0.1102 * (atten_db - 8.7)
    } else if atten_db >= 21.0 {
        let d = atten_db - 21.0;
        0.5842 * d.powf(0.4) + 0.07886 * d
    } else {
        0.0
    }
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x2 = (x * x) * 0.25;
    for k in 1..80 {
        term *= x2 / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
    }
    sum
}

fn kaiser_window(i: usize, n: usize, beta: f64) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    let mid = (n as f64 - 1.0) / 2.0;
    let r = (i as f64 - mid) / mid;
    if r.abs() >= 1.0 {
        return 0.0;
    }
    bessel_i0(beta * (1.0 - r * r).sqrt()) / bessel_i0(beta)
}

fn gauss_window(i: usize, n: usize, alpha: f64) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    let mid = (n as f64 - 1.0) / 2.0;
    let x = (i as f64 - mid) / mid;
    (-0.5 * alpha * alpha * x * x).exp()
}

fn blackman_harris(i: usize, n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    const A0: f64 = 0.35875;
    const A1: f64 = 0.48829;
    const A2: f64 = 0.14128;
    const A3: f64 = 0.01168;
    let phi = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64 - 1.0);
    A0 - A1 * phi.cos() + A2 * (2.0 * phi).cos() - A3 * (3.0 * phi).cos()
}

/// Magnitude of `h` at `cycles` cycles/sample, in dB. Used by tests and not
/// on the audio path.
pub fn response_db(h: &[f64], cycles: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * cycles;
    let mut re = 0.0;
    let mut im = 0.0;
    for (n, &c) in h.iter().enumerate() {
        let a = w * n as f64;
        re += c * a.cos();
        im -= c * a.sin();
    }
    20.0 * (re * re + im * im).sqrt().max(1e-30).log10()
}
