//! Quantiser with optional TPDF dither and noise shaping.
//!
//! TPDF (two independent uniform draws, subtracted) is triangular over ±1 LSB
//! and is the right default at every rate. It linearises the quantiser and
//! leaves a white floor. At 24-bit that floor sits near −141 dBFS, which is
//! the honest point of diminishing returns.
//!
//! `shaped` at 44.1 kHz output uses the 9-tap F-weighted error-feedback filter
//! published by Robert Wannamaker, "Psychoacoustically Optimal Noise Shaping",
//! JAES vol. 40 no. 7/8 (1992). Those coefficients are for 44.1 kHz only.
//! At 48 / 96 / 192 kHz this tool does **not** pretend they still land on the
//! equal-loudness curve. It uses a first-order highpass (`y = x + (1 − a z⁻¹) q`)
//! whose boost is mostly ultrasonic:
//!
//! | output | a    | roughly, at 4 kHz | at 18 kHz |
//! |--------|------|-------------------|-----------|
//! | 48 kHz | 0.50 | −4 dB             | +3 dB     |
//! | 96 kHz | 0.70 | −9 dB             | −0.2 dB   |
//! | 192 kHz| 0.85 | −14 dB            | −5 dB     |
//!
//! The feedback structure matches the standard one: the coefficient on the
//! most recent quantisation error is `coefs[0]`, and
//! `NTF(z) = 1 − Σ coefs[k] z^{−(k+1)}`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DitherMode {
    None,
    Tpdf,
    Shaped,
}

pub struct Quantizer {
    mode: DitherMode,
    scale: f64,
    min_i: f64,
    max_i: f64,
    coefs: Vec<f64>,
    errors: Vec<f64>,
    rng: u64,
    pub clips: u64,
}

impl Quantizer {
    pub fn new(mode: DitherMode, bits: u16, channel: usize, rate: u32) -> Self {
        let coefs = match mode {
            DitherMode::Shaped => shaper_coefs(rate).to_vec(),
            DitherMode::None | DitherMode::Tpdf => Vec::new(),
        };
        let n = coefs.len();
        let scale = (1u32 << (bits - 1)) as f64;
        Self {
            mode,
            scale,
            min_i: -scale,
            max_i: scale - 1.0,
            coefs,
            errors: vec![0.0; n],
            rng: seed_for(channel),
            clips: 0,
        }
    }

    pub fn shaper_name(mode: DitherMode, rate: u32) -> String {
        match mode {
            DitherMode::None => "none (round to nearest)".to_string(),
            DitherMode::Tpdf => "tpdf".to_string(),
            DitherMode::Shaped if rate == 44_100 => {
                "shaped: Wannamaker F-weight, 9-tap, 44.1 kHz".to_string()
            }
            DitherMode::Shaped => {
                format!(
                    "shaped: first-order highpass a={:.2} at {rate} Hz",
                    shaper_coefs(rate)[0]
                )
            }
        }
    }

    /// `sample` is a float in nominal ±1. Returns the integer PCM code.
    pub fn quantize(&mut self, sample: f64) -> i32 {
        let x = sample * self.scale;
        let mut feedback = 0.0;
        for (c, e) in self.coefs.iter().zip(self.errors.iter()) {
            feedback += c * e;
        }
        let shaped = x - feedback;
        let dither = match self.mode {
            DitherMode::None => 0.0,
            DitherMode::Tpdf | DitherMode::Shaped => self.tpdf(),
        };
        let mut q = (shaped + dither).round();
        let mut clipped = false;
        if q < self.min_i {
            q = self.min_i;
            clipped = true;
        } else if q > self.max_i {
            q = self.max_i;
            clipped = true;
        }
        if clipped {
            self.clips += 1;
        }
        if !self.errors.is_empty() {
            let err = q - shaped;
            for i in (1..self.errors.len()).rev() {
                self.errors[i] = self.errors[i - 1];
            }
            self.errors[0] = err;
        }
        q as i32
    }

    fn tpdf(&mut self) -> f64 {
        self.uniform() - self.uniform()
    }

    fn uniform(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        let u = x >> 11;
        (u as f64 + 0.5) / (1u64 << 53) as f64
    }
}

fn shaper_coefs(rate: u32) -> &'static [f64] {
    match rate {
        44_100 => &[
            2.412, -3.370, 3.937, -4.174, 3.353, -2.205, 1.281, -0.569, 0.0847,
        ],
        48_000 => &[0.50],
        96_000 => &[0.70],
        192_000 => &[0.85],
        _ => &[0.60],
    }
}

fn seed_for(channel: usize) -> u64 {
    let mut z = 0x9E37_79B9_7F4A_7C15u64 ^ (channel as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let z = z ^ (z >> 31);
    if z == 0 {
        0xA5A5_A5A5_A5A5_A5A5
    } else {
        z
    }
}
