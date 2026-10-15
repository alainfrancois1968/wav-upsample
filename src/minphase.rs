//! Linear-phase FIR → minimum-phase FIR via the real cepstrum.
//!
//! Minimum phase keeps the magnitude response and removes the pre-ring.
//! That is the audible difference HQPlayer users chase with the `-mp`
//! variants: the transient is not preceded by a time-reversed copy of the
//! ringing. Stopband depth of a very long Kaiser filter makes the log-magnitude
//! cepstrum delicate, so the FFT is zero-padded and spectral zeros are floored.

use anyhow::{bail, Result};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

pub fn to_minimum_phase(h: &[f64]) -> Result<Vec<f64>> {
    let n = h.len();
    if n < 8 {
        bail!("filter is too short for a minimum-phase conversion");
    }
    let fft_len = choose_fft_len(n)?;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    let mut buf = vec![Complex::new(0.0, 0.0); fft_len];
    for (i, &c) in h.iter().enumerate() {
        buf[i] = Complex::new(c, 0.0);
    }
    fft.process(&mut buf);

    // rustfft's forward transform is unnormalized: DC = sum(h) ≈ 1.
    let floor = 1e-18;
    for c in &mut buf {
        let mag = c.norm().max(floor);
        *c = Complex::new(mag.ln(), 0.0);
    }

    ifft.process(&mut buf);
    let inv_len = 1.0 / fft_len as f64;
    for c in &mut buf {
        // Numerical imag residue is time-alias junk; the cepstrum of a
        // real sequence is real.
        *c = Complex::new(c.re * inv_len, 0.0);
    }

    let half = fft_len / 2;
    for i in 1..half {
        buf[i].re *= 2.0;
    }
    for i in (half + 1)..fft_len {
        buf[i] = Complex::new(0.0, 0.0);
    }

    fft.process(&mut buf);
    for c in &mut buf {
        *c = c.exp();
    }
    ifft.process(&mut buf);

    let mut out = Vec::with_capacity(n);
    for c in buf.iter().take(n) {
        let v = c.re * inv_len;
        if !v.is_finite() {
            bail!(
                "minimum-phase conversion went non-finite. \
                 Use --phase linear, or a shorter --quality."
            );
        }
        out.push(v);
    }

    let sum_in: f64 = h.iter().sum();
    let sum_out: f64 = out.iter().sum();
    if sum_out.abs() < 1e-18 || !sum_out.is_finite() {
        bail!("minimum-phase filter has no DC gain");
    }
    let g = sum_in / sum_out;
    for v in &mut out {
        *v *= g;
    }
    Ok(out)
}

fn choose_fft_len(n: usize) -> Result<usize> {
    // 4× padding is the usual cepstrum practice. Fall back to 2× if 4×
    // would exceed a few hundred megabytes, and refuse beyond that.
    for mult in [4usize, 2] {
        let need = n.saturating_mul(mult);
        let len = need.next_power_of_two();
        if len >= n.saturating_mul(2) && len <= 8_388_608 {
            return Ok(len);
        }
    }
    bail!(
        "minimum-phase FFT for a {n}-tap filter does not fit in memory. \
         Use --phase linear or a shorter --quality."
    );
}
