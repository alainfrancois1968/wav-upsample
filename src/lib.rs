//! High-quality offline WAV resampler.
//!
//! See the README for why this is a single rational polyphase windowed-sinc
//! (Kaiser, Gaussian, or Blackman-Harris), optionally minimum-phase, with
//! Rayon across channels and across time slices.

#![deny(unsafe_code)]

pub mod design;
pub mod dither;
pub mod minphase;
pub mod resample;
pub mod wavio;

pub use design::{Phase, QualitySpec, Window};
pub use dither::DitherMode;
pub use wavio::{convert_file, Settings};

#[cfg(test)]
mod tests;
