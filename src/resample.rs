//! Polyphase applicator.
//!
//! Prototype `h[0..N)` is the lowpass at the high rate `fs_high = L · fs_in`,
//! normalised so `sum(h) = 1`. Output sample `n` is the high-rate convolution
//! sampled at high-rate index `n·M`, which decimates by M:
//!
//! ```text
//! y[n] = L · Σ_s  h[p + s·L] · x[k0 − s]
//! p    = (n·M + delay) mod L
//! ```
//!
//! `delay` recentres a linear-phase filter on the output timestamp (the
//! causal form would otherwise shift the file by half the kernel). For a
//! minimum-phase prototype `delay` is the peak index, so the transient lines
//! up and the pre-ring does not come back.
//!
//! Each phase is then scaled to sum to exactly 1. A pure windowed sinc is
//! already extremely close; the per-phase trim kills residual DC gain wobble
//! between fractional delays (audible as a slow level modulation on a
//! non-integer ratio such as 44.1 → 48).

use rayon::prelude::*;

use crate::design::Prototype;

#[derive(Clone)]
pub struct Polyphase {
    pub l: usize,
    pub m: usize,
    pub taps: usize,
    pub delay_hr: usize,
    /// Phase-major, reversed so index 0 multiplies the earliest input sample.
    coeffs: Vec<f64>,
}

impl Polyphase {
    pub fn build(proto: &Prototype) -> Self {
        let l = proto.info.l;
        let taps = proto.info.taps_per_phase;
        let mut coeffs = vec![0.0; l * taps];
        for p in 0..l {
            let mut phase = vec![0.0; taps];
            let mut sum = 0.0;
            for s in 0..taps {
                let c = proto.h[p + s * l] * l as f64;
                phase[s] = c;
                sum += c;
            }
            let inv = if sum.abs() > 1e-18 { 1.0 / sum } else { 0.0 };
            let base = p * taps;
            for j in 0..taps {
                // s = taps-1-j is the earliest input (see module comment).
                coeffs[base + j] = phase[taps - 1 - j] * inv;
            }
        }
        Self {
            l,
            m: proto.info.m,
            taps,
            delay_hr: proto.info.delay_hr,
            coeffs,
        }
    }
}

pub fn render_channel(input: &[f64], poly: &Polyphase, start: usize, end: usize) -> Vec<f64> {
    let len = end - start;
    if len == 0 {
        return Vec::new();
    }
    // Small blocks stay serial. Large ones are sliced so Rayon has something
    // to steal even when the file is mono (stereo alone would only fill two
    // cores). Nested `par_iter` uses the same global pool.
    const SERIAL_LIMIT: usize = 8_192;
    const SLICE: usize = 8_192;
    if len < SERIAL_LIMIT {
        return render_serial(input, poly, start, end);
    }
    let mut out = vec![0.0; len];
    let offsets: Vec<usize> = (0..len).step_by(SLICE).collect();
    let parts: Vec<(usize, Vec<f64>)> = offsets
        .par_iter()
        .map(|&off| {
            let s = start + off;
            let e = (s + SLICE).min(end);
            (off, render_serial(input, poly, s, e))
        })
        .collect();
    for (off, part) in parts {
        out[off..off + part.len()].copy_from_slice(&part);
    }
    out
}

fn render_serial(input: &[f64], poly: &Polyphase, start: usize, end: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(end - start);
    let l = poly.l as u64;
    let m = poly.m as u64;
    let delay = poly.delay_hr as u64;
    let taps = poly.taps;
    let mut nm = start as u64 * m + delay;
    for _n in start..end {
        let p = (nm % l) as usize;
        let k_at_s0 = (nm / l) as isize;
        let k_left = k_at_s0 - (taps as isize - 1);
        let coeff = &poly.coeffs[p * taps..(p + 1) * taps];
        out.push(dot(input, k_left, coeff));
        nm += m;
    }
    out
}

#[inline]
fn dot(input: &[f64], k_left: isize, coeff: &[f64]) -> f64 {
    let taps = coeff.len() as isize;
    let in_len = input.len() as isize;
    let end = k_left + taps;
    let vs = k_left.max(0);
    let ve = end.min(in_len);
    if vs >= ve {
        return 0.0;
    }
    let xs = &input[vs as usize..ve as usize];
    let cs = &coeff[(vs - k_left) as usize..(ve - k_left) as usize];
    let n = xs.len();
    let mut acc = 0.0;
    let mut i = 0;
    while i + 8 <= n {
        acc += xs[i] * cs[i]
            + xs[i + 1] * cs[i + 1]
            + xs[i + 2] * cs[i + 2]
            + xs[i + 3] * cs[i + 3]
            + xs[i + 4] * cs[i + 4]
            + xs[i + 5] * cs[i + 5]
            + xs[i + 6] * cs[i + 6]
            + xs[i + 7] * cs[i + 7];
        i += 8;
    }
    while i < n {
        acc += xs[i] * cs[i];
        i += 1;
    }
    acc
}
