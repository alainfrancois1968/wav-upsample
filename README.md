# wav-upsample

Offline WAV resampler. One rational-ratio polyphase FIR, double precision,
Rayon across channels and across time. Output rates are 44.1, 48, 96 and
192 kHz. Output bit depth is 16 or 24. Input can be 8/16/24/32-bit PCM or
32-bit float, at whatever rate the file actually is. If the output rate is
lower than the input, the same filter is the anti-alias filter, so
downsampling is not a special case.

This is inspired by the *ideas* in HQPlayer's poly-sinc family (a long
windowed sinc, a Gaussian window when you care about ringing, linear versus
minimum phase, and a real dither/noise-shaper choice). It is not HQPlayer.
There is no DSD modulator, no closed-form modulator, and the noise shaper is
the published Wannamaker F-weight curve at 44.1 kHz rather than a clone of
NS9 / LNS15.

## Build

```text
cargo build --release
cargo test --release
./target/release/wav-upsample --help
```

Rust 1.74 or newer. Release mode matters; the default quality is a few
hundred taps per output sample and debug builds are several times slower.

```text
./target/release/wav-upsample \
  -i cd.wav -o up.wav \
  -r 96000 -b 24 \
  --filter gauss --quality high --phase minimum --dither tpdf
```

`--info` designs the filter and prints the ratio, transition width and tap
count without writing audio.

## What was considered, and what this program actually does

| Approach | Why it loses for this job | Why it would win |
|---|---|---|
| Linear / cubic / Catmull-Rom | Images and aliases sit 40–70 dB down. Fine for graphics, not for 24-bit audio. | Tiny latency, tiny CPU. |
| Cascaded half-band FIR (the fast 2× trick SoX uses internally) | 44.1 ↔ 48 is not a power of two, so you still need a second, ugly rational stage. Two transition bands, two places to get the gain wrong. | Much cheaper for pure 2× / 4×. |
| FFT overlap-add of one long FIR | Same filter, more bookkeeping. Block edges, overlap, and a second code path for minimum phase. Easy to ship a click. | Wins once the kernel is tens of thousands of taps *and* you are single-threaded. |
| IIR (Butterworth / elliptic) | No exact linear phase. Elliptic filters ring too, and the phase wiggle is the part people are trying to escape. | Very few coefficients. |
| Single windowed-sinc polyphase, ratio L/M | This one. Every output sample is a dot product of 100–1500 input samples. At 192 kHz stereo that is a few hundred million MACs per second, which a desktop CPU does faster than realtime once Rayon is involved. | One transition band, exact linear phase if you want it, DC gain you can prove, and a minimum-phase twin with the same magnitude. |

The ratio is reduced with `gcd`. 44.1 kHz → 96 kHz is 320/147. 44.1 → 48 is
160/147. 48 → 192 is 4/1. The lowpass is designed at the high rate
`L · fs_in` with its cutoff in the middle of the transition, then split into
`L` phases. The passband edge is a fraction of the *lower* Nyquist, so
upsampling keeps (and slightly apodises) the original band, and downsampling
cuts at the new Nyquist. Kaiser length and β come from the Oppenheim / Schafer
estimate `N ≈ (A − 8) / (2.285 · Δω)`, so asking for more attenuation or a
narrower transition spends taps in a way that actually buys that attenuation,
instead of an arbitrary "quality" enum with a hidden kernel.

Each phase is normalised to sum to 1. A raw windowed sinc is already very
close; the trim removes the tiny gain difference between fractional delays,
which on a non-integer ratio (44.1 → 48) would otherwise be a slow amplitude
wobble.

Linear phase is the symmetric kernel, recentred so the file is not shifted by
half the filter. The residual error is at most half a high-rate sample
(`≤ 0.5/L` input samples) when the prototype length is even. Minimum phase is
the same magnitude response pushed through a real-cepstrum conversion, then
aligned on its peak. Pre-ring goes away; post-ring stays. That is the useful
part of the poly-sinc-mp idea, without claiming the same coefficients.

Internal processing is f64. The output stage scales by `2^(bits−1)`, adds
dither, and rounds. Triangular (TPDF) dither is two uniform draws subtracted,
so it spans ±1 LSB. It is the right default at every rate. `shaped` at
44.1 kHz output is Wannamaker's 9-tap F-weighted error feedback (JAES 1992);
at 48/96/192 kHz that curve is the wrong shape in hertz, so the shaper is a
short first-order highpass whose boost sits above the audible band. There is
no attempt to reproduce HQPlayer's noise shapers.

Rayon is used in two places. Filter coefficients are filled in parallel.
Audio is split per channel and, inside a channel, into 8192-sample output
slices. Each slice only reads the input span it needs, so there is no shared
mutable state and the result does not depend on thread count. Dither state is
sequential and runs after the parallel resample, one channel at a time, so
the noise sequence is deterministic.

## Presets

| quality | passband | design stopband | character |
|---|---|---|---|
| `short` | 90% of the lower Nyquist | 100 dB | fast, less ringing |
| `high` | 95.5% | 140 dB | default, about 24-bit clean |
| `extreme` | 98.5% (~21.7 kHz on a CD) | 170 dB | long; linear phase will pre-ring |

| filter | window | what you are choosing |
|---|---|---|
| `kaiser` | Kaiser | steepest stopband for the tap budget |
| `gauss` | Gaussian | the time-frequency compromise. Less ringing, a softer knee. This is the poly-sinc-gauss slot |
| `blackman-harris` | 4-term Blackman–Harris | a fixed ~92 dB floor. Length still follows `--quality` / `--atten`, but the window will not go deeper than its own sidelobes |

`--bandwidth` and `--atten` override the preset. Attenuation above ~170 dB
with a very narrow transition asks for thousands of taps and will be refused
rather than silently truncated.

`--headroom 1` (decibels of gain cut before the quantiser) is the escape
hatch when ringing drives intersample peaks over 0 dBFS. The tool never
hides clipping: it prints the peak, the number of samples over full scale,
and the number of integer codes that saturated.

## Practical limits

* Classic WAV is 4 GB. A 192 kHz 24-bit stereo file crosses that a bit under
  an hour. The program stops before writing a broken header.
* The whole input is held as f64 (about 8 bytes × channels × frames) so the
  parallel slices can read freely. Output is streamed in blocks of 65536
  frames.
* `--phase minimum` on `--quality extreme` runs a multi-million-point FFT.
  It is allowed, and it is slow. `--info` is the way to see the size first.
* Gaussian and Blackman-Harris do **not** reach the Kaiser attenuation
  number. The number still sets how long the kernel is.

## Examples

```text
# CD to 96 kHz / 24-bit, default Kaiser, linear phase, TPDF
wav-upsample -i album.wav -o album-96.wav -r 96k -b 24

# Shorter ringing, minimum phase, for material that is mostly transient
wav-upsample -i album.wav -o album-96.wav -r 96000 -b 24 \
  --filter gauss --quality short --phase minimum

# 16-bit target. Shaped dither earns its keep here, and only at 44.1 kHz
# is it the full F-weight curve.
wav-upsample -i hires.wav -o cd.wav -r 44100 -b 16 --dither shaped --headroom 1

# See the kernel before spending the CPU
wav-upsample -i album.wav -r 192000 -b 24 --quality extreme --info
```

## Tests

`cargo test` covers the awkward 160/147 ratio (DC gain and a 1 kHz sine),
image rejection on a 2× upsample, alias rejection when downsampling a 30 kHz
tone into 48 kHz, minimum-phase magnitude match, impulse alignment, TPDF
statistics, and a real WAV round trip through `hound` at 24-bit.

## License

MIT. Wannamaker's nine coefficients are numerical results from the 1992 JAES
paper, cited in `src/dither.rs`.
