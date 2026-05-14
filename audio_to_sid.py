#!/usr/bin/env python3
"""
audio_to_sid.py — Convert an audio file to a PSID-format C64 SID file.

Pipeline (Mahoney-inspired):
  1. Load audio; HPSS harmonic/percussive separation
  2. FFT autocorrelation pitch tracking per band (bass/mid/high)
  3. rmsLow > rmsHigh voiced gate — silent when high-freq energy dominates
  4. Per-frame waveform matching: synthesize triangle/sawtooth/pulse at the
     detected pitch and pick the best cross-correlation match
  5. analyze_instrument() for ADSR selection per voice
  6. Generate 6502 machine code that drives the SID chip each 50 Hz frame
  7. Write a valid PSID v2 file

  Waveform mapping:
    triangle  (0x10) — soft/few harmonics  → vocals, pads
    sawtooth  (0x20) — rich harmonics      → bass guitar, strings, brass
    pulse     (0x40) — hollow/nasal        → electric guitar, lead synth
    noise     (0x80) — unpitched transient → percussion (when detected)

Usage:
  python3 audio_to_sid.py <input_audio> <output.sid> [--title "Title"] [--author "Author"]

Supported input formats: anything librosa can load (wav, mp3, ogg, flac, …).
Output: PSID v2 file playable in SIDPLAY2, JSIDPLAY2, or real C64 hardware.
"""

import argparse
import struct
import numpy as np
import librosa
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------------------------
# SID chip constants (PAL C64)
# ---------------------------------------------------------------------------
PAL_CLOCK = 985_248
FRAME_RATE = 50

def hz_to_sid_freq(hz: float) -> int:
    if hz <= 0:
        return 0
    return max(0, min(0xFFFF, int(round(hz * (1 << 24) / PAL_CLOCK))))

def midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))

def hz_to_midi(hz: float) -> int:
    if hz <= 0:
        return 0
    return int(round(69 + 12 * np.log2(hz / 440.0)))

# ---------------------------------------------------------------------------
# SID register map
# ---------------------------------------------------------------------------
SID_BASE   = 0xD400
FREQ_LO    = [0x00, 0x07, 0x0E]
FREQ_HI    = [0x01, 0x08, 0x0F]
PW_LO      = [0x02, 0x09, 0x16]
PW_HI      = [0x03, 0x0A, 0x17]
CTRL       = [0x04, 0x0B, 0x12]
ATDEC      = [0x05, 0x0C, 0x13]
SUSTREL    = [0x06, 0x0D, 0x14]
VOL_FILTER = 0x18

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = 22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

# ---------------------------------------------------------------------------
# Voice extraction  (replaces the old band-split + piptrack approach)
# ---------------------------------------------------------------------------

def _lowpass(y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff / (sr / 2), btype='low', output='sos')
    return sosfilt(sos, y)

def _bandpass(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    nyq = sr / 2
    sos = butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')
    return sosfilt(sos, y)

def _highpass(y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff / (sr / 2), btype='high', output='sos')
    return sosfilt(sos, y)


# ---------------------------------------------------------------------------
# Mahoney-inspired helpers: autocorrelation pitch, RMS gate, waveform matching
# ---------------------------------------------------------------------------

def _synthesize_waveform(wave: str, freq: float, sr: int, n_samples: int) -> np.ndarray:
    """One period of a SID waveform (triangle/sawtooth/pulse), zero-mean."""
    phase = (np.arange(n_samples) * freq / sr) % 1.0
    if wave == 'triangle':
        y = 1.0 - 4.0 * np.abs(phase - 0.5)
    elif wave == 'sawtooth':
        y = 2.0 * phase - 1.0
    else:  # pulse 50%
        y = np.where(phase < 0.5, 1.0, -1.0).astype(float)
    return y - float(np.mean(y))


def _autocorr_pitch(y: np.ndarray, sr: int, fmin: float, fmax: float,
                    hop_length: int, n_frames: int) -> list[float]:
    """
    Per-frame pitch via FFT autocorrelation (Mahoney-style).
    Uses a 2× overlap window and parabolic interpolation on the peak.
    Returns Hz per frame; 0.0 = unvoiced (autocorr peak < 30% of zero-lag).
    """
    lag_min = max(1, int(sr / fmax))
    lag_max = int(sr / fmin)
    win_size = hop_length * 2

    pitches: list[float] = []
    for i in range(n_frames):
        start = i * hop_length
        end   = start + win_size
        if end > len(y):
            pitches.append(0.0)
            continue

        frame = y[start:end].copy()
        frame -= np.mean(frame)
        frame *= np.hanning(len(frame))

        # FFT-based autocorrelation (O(n log n))
        n_fft = 1 << int(np.ceil(np.log2(2 * win_size)))
        F = np.fft.rfft(frame, n=n_fft)
        r = np.fft.irfft(F * np.conj(F))[:win_size]

        r0 = r[0]
        if r0 < 1e-10 or lag_max >= win_size:
            pitches.append(0.0)
            continue

        r_search  = r[lag_min:lag_max]
        peak_idx  = int(np.argmax(r_search))
        peak_val  = float(r_search[peak_idx])

        if peak_val / r0 < 0.30:
            pitches.append(0.0)
            continue

        lag = float(lag_min + peak_idx)
        # Parabolic interpolation for sub-sample accuracy
        if 0 < peak_idx < len(r_search) - 1:
            a, b, g = r_search[peak_idx-1], r_search[peak_idx], r_search[peak_idx+1]
            denom = a - 2*b + g
            if abs(denom) > 1e-10:
                lag += 0.5 * (a - g) / denom

        pitches.append(float(sr) / lag)
    return pitches


def _voiced_rms_gate(y: np.ndarray, sr: int, hop_length: int,
                     n_frames: int, split_hz: float = 3000.0) -> np.ndarray:
    """
    Mahoney-style voiced gate: voiced when rmsLow > rmsHigh and above 3% of
    the peak rmsLow. Smooths isolated voiced/silent single frames.
    """
    low  = _lowpass(y,  sr, split_hz)
    high = _highpass(y, sr, split_hz)

    voiced  = np.zeros(n_frames, dtype=bool)
    rms_low = np.zeros(n_frames)

    for i in range(n_frames):
        s, e = i * hop_length, min((i + 1) * hop_length, len(y))
        rl = float(np.sqrt(np.mean(low[s:e]  ** 2) + 1e-12))
        rh = float(np.sqrt(np.mean(high[s:e] ** 2) + 1e-12))
        rms_low[i] = rl
        voiced[i]  = rl > rh

    threshold = float(np.max(rms_low)) * 0.03
    voiced &= rms_low > threshold

    # Remove isolated voiced single frames; fill isolated silent single frames
    for i in range(1, n_frames - 1):
        if voiced[i] and not voiced[i-1] and not voiced[i+1]:
            voiced[i] = False
    for i in range(1, n_frames - 1):
        if not voiced[i] and voiced[i-1] and voiced[i+1]:
            voiced[i] = True

    return voiced


def _match_waveform_per_frame(y: np.ndarray, sr: int, notes: list[int],
                               hop_length: int) -> list[int]:
    """
    For each voiced (non-zero) note, synthesize triangle/sawtooth/pulse at
    that pitch and pick the waveform with the highest zero-lag normalized
    cross-correlation against the actual audio frame.
    Returns a SID waveform byte (0x10/0x20/0x40) per frame.
    Unvoiced frames return 0x10 as a neutral placeholder.
    """
    WAVES = [('triangle', 0x10), ('sawtooth', 0x20), ('pulse', 0x40)]
    ctrl: list[int] = []

    for i, midi in enumerate(notes):
        if midi == 0:
            ctrl.append(0x10)
            continue

        hz  = midi_to_hz(midi)
        s   = i * hop_length
        e   = min(s + hop_length, len(y))
        frm = y[s:e].copy()
        frm -= np.mean(frm)
        n   = len(frm)
        pk  = np.max(np.abs(frm))
        if n == 0 or pk < 1e-8:
            ctrl.append(0x10)
            continue

        frm_n = frm / pk
        best_byte, best_score = 0x10, -np.inf
        for name, byte in WAVES:
            synth = _synthesize_waveform(name, hz, sr, n)
            sp = np.max(np.abs(synth))
            if sp < 1e-8:
                continue
            score = float(np.dot(frm_n, synth / sp)) / n
            if score > best_score:
                best_score, best_byte = score, byte

        ctrl.append(best_byte)
    return ctrl


def analyze_instrument(y: np.ndarray, sr: int, band: str) -> dict:
    """
    Classify the dominant instrument in y using spectral features and return
    a SID voice config: waveform byte, atdec byte, sustrel byte, name string.

    band: 'bass' | 'mid' | 'high'

    Decision axes:
      pluck_score  — onset peak / mean onset: high = plucked/percussive attack
      centroid_n   — spectral centroid normalised to Nyquist: high = bright
      flatness     — 0 = tonal, 1 = noisy

    SID waveforms (no noise):
      0x10 triangle  — soft, few harmonics → vocals, pads, sub bass
      0x20 sawtooth  — rich harmonics     → bass guitar, strings, brass
      0x40 pulse     — hollow/nasal       → electric guitar, lead synth, keys
    """
    if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
        return {'waveform': 0x10, 'atdec': 0x09, 'sustrel': 0xA4, 'name': 'silent'}

    onset_env   = librosa.onset.onset_strength(y=y, sr=sr)
    onset_mean  = np.mean(onset_env) + 1e-6
    pluck_score = np.max(onset_env) / onset_mean   # > 4 = plucked

    centroid_n = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))) / (sr / 2)
    flatness   = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    is_plucked  = pluck_score > 4.0
    is_bright   = centroid_n  > 0.08   # 0.08*nyquist ≈ 880Hz — covers guitar/keys mid range
    is_tonal    = flatness    < 0.10
    is_perc     = flatness    > 0.25 and pluck_score > 6.0  # noisy transients, not pitched

    # Percussion: noise waveform, sharp gate burst (fast attack/decay, no sustain)
    if is_perc:
        return {'waveform': 0x80, 'atdec': 0x03, 'sustrel': 0x03, 'name': 'percussion'}

    if band == 'bass':
        if is_plucked and is_tonal:
            # Bass guitar: sawtooth, fast pluck (attack=0 ~2ms, decay=8 ~100ms,
            # sustain=6, release=4 ~38ms)
            return {'waveform': 0x20, 'atdec': 0x08, 'sustrel': 0x64, 'name': 'bass guitar'}
        elif not is_plucked and centroid_n < 0.10:
            # Organ/synth bass: sawtooth, slow attack, full sustain
            return {'waveform': 0x20, 'atdec': 0x50, 'sustrel': 0xF4, 'name': 'synth/organ bass'}
        else:
            return {'waveform': 0x20, 'atdec': 0x18, 'sustrel': 0x82, 'name': 'bass (generic)'}

    elif band == 'mid':
        if is_plucked and is_bright:
            # Electric guitar: pulse, fast attack, medium sustain
            return {'waveform': 0x40, 'atdec': 0x06, 'sustrel': 0x74, 'name': 'electric guitar'}
        elif is_plucked and not is_bright:
            # Acoustic guitar / piano: pulse, fast attack, shorter sustain
            return {'waveform': 0x40, 'atdec': 0x09, 'sustrel': 0x54, 'name': 'acoustic guitar/piano'}
        elif not is_plucked and not is_bright:
            # Vocals / soft lead: triangle, gentle attack
            return {'waveform': 0x10, 'atdec': 0x32, 'sustrel': 0xB5, 'name': 'vocals/soft lead'}
        else:
            # Bright sustained: brass / organ / strings → sawtooth
            return {'waveform': 0x20, 'atdec': 0x50, 'sustrel': 0xF4, 'name': 'organ/brass/strings'}

    else:  # 'high'
        if is_plucked:
            return {'waveform': 0x40, 'atdec': 0x09, 'sustrel': 0x84, 'name': 'plucked high lead'}
        elif is_bright:
            return {'waveform': 0x20, 'atdec': 0x19, 'sustrel': 0xA4, 'name': 'bright high lead'}
        else:
            return {'waveform': 0x10, 'atdec': 0x19, 'sustrel': 0xA4, 'name': 'soft high lead'}


def _autocorr_to_notes(pitches: list[float], voiced: np.ndarray,
                       midi_lo: int, midi_hi: int) -> list[int]:
    """Convert autocorr Hz list + voiced mask to clamped MIDI notes (0 = silence)."""
    notes = []
    for hz, is_voiced in zip(pitches, voiced):
        if is_voiced and hz > 0:
            midi = hz_to_midi(hz)
            while midi < midi_lo:
                midi += 12
            while midi > midi_hi:
                midi -= 12
            notes.append(max(midi_lo, min(midi_hi, midi)))
        else:
            notes.append(0)
    return notes


def _smooth_notes(notes: list[int], min_hold: int = 4) -> list[int]:
    """Prevent rapid jitter: once a note starts, hold it for at least min_hold frames."""
    out = list(notes)
    i = 0
    while i < len(out):
        if out[i] != 0:
            end = min(i + min_hold, len(out))
            for j in range(i + 1, end):
                if out[j] == 0:
                    out[j] = out[i]   # extend into short silences
            i = end
        else:
            i += 1
    return out


def _quantize_to_grid(notes: list[int], onset_frames: set[int],
                      frames_per_cell: int) -> list[int]:
    """
    Snap notes onto a musical time grid.

    Each grid cell spans `frames_per_cell` frames (≈ one 16th note at the
    detected tempo).  Within each cell the most-common non-zero note wins;
    a note change is only accepted when there is an onset in that cell,
    otherwise the previous note carries forward.  This turns the raw
    50 Hz per-frame stream into a note-event sequence that sounds like a
    real SID composition rather than a continuous pitch tracker.
    """
    from collections import Counter
    n = len(notes)
    out = [0] * n
    prev = 0
    cell_size = max(1, frames_per_cell)

    for cell_start in range(0, n, cell_size):
        cell_end = min(cell_start + cell_size, n)
        cell_notes = [notes[f] for f in range(cell_start, cell_end)]

        non_zero = [x for x in cell_notes if x != 0]
        candidate = Counter(non_zero).most_common(1)[0][0] if non_zero else 0

        # Only change pitch when an onset lands in this cell
        cell_has_onset = any(f in onset_frames for f in range(cell_start, cell_end))
        if candidate != 0 and cell_has_onset:
            prev = candidate
        elif candidate == 0 and cell_has_onset:
            prev = 0   # onset with no pitch = explicit silence

        for f in range(cell_start, cell_end):
            out[f] = prev

    return out


def _suppress_overlapping_voices(melody: list[int], high: list[int],
                                 octave_threshold: int = 12) -> list[int]:
    """
    Mute Voice 2 (high lead) whenever it is within `octave_threshold`
    semitones of Voice 0 (melody).  Melody always wins; Voice 2 only plays
    when it has clearly independent content above the melody.
    """
    out = list(high)
    for i, (m, h) in enumerate(zip(melody, high)):
        if m != 0 and h != 0 and abs(h - m) < octave_threshold:
            out[i] = 0
    return out


def extract_voices(y: np.ndarray, sr: int,
                   hop_length: int, n_frames: int) -> tuple:
    """
    Return:
      melody, bass, high  — per-frame MIDI note lists (0 = silence)
      voice_ctrl          — per-frame SID waveform bytes, one list per voice
      voice_configs       — ADSR config dicts, one per voice

    Pipeline (Mahoney-inspired):
      1. HPSS harmonic/percussive separation
      2. FFT autocorrelation pitch tracking per band
      3. rmsLow/rmsHigh voiced gate (Mahoney's silence criterion)
      4. Per-frame waveform matching: synthesize triangle/sawtooth/pulse and
         pick whichever correlates best with the actual audio at that pitch
      5. analyze_instrument() used only for ADSR selection
    """
    from collections import Counter

    print("  Separating harmonic / percussive components…")
    harmonic, _ = librosa.effects.hpss(y, margin=3.0)

    bass_sig = _lowpass(harmonic, sr, 450.0)
    mid_sig  = _bandpass(harmonic, sr, 450.0, 2000.0)

    # Global voiced gate on harmonic signal (Mahoney: silent when high freq dominates)
    print("  Computing voiced gate (rmsLow vs rmsHigh)…")
    gate = _voiced_rms_gate(harmonic, sr, hop_length, n_frames, split_hz=3000.0)

    # --- Voice 1: bass, C1–C3 ---
    print("  Tracking bass (C1–C3) via autocorrelation…")
    p_bass   = _autocorr_pitch(bass_sig, sr, librosa.note_to_hz('C1'),
                                librosa.note_to_hz('C3'), hop_length, n_frames)
    v_bass   = gate & np.array([p > 0 for p in p_bass])
    bass     = _smooth_notes(_autocorr_to_notes(p_bass, v_bass, 24, 48), min_hold=4)

    # --- Voice 0: melody, C3–C6 ---
    print("  Tracking melody (C3–C6) via autocorrelation…")
    p_mel    = _autocorr_pitch(harmonic, sr, librosa.note_to_hz('C3'),
                               librosa.note_to_hz('C6'), hop_length, n_frames)
    v_mel    = gate & np.array([p > 0 for p in p_mel])
    melody   = _smooth_notes(_autocorr_to_notes(p_mel, v_mel, 48, 84), min_hold=4)

    # --- Voice 2: high lead, C4–C7 (longer hold to suppress jitter) ---
    print("  Tracking high lead (C4–C7) via autocorrelation…")
    p_high   = _autocorr_pitch(mid_sig, sr, librosa.note_to_hz('C4'),
                               librosa.note_to_hz('C7'), hop_length, n_frames)
    v_high   = gate & np.array([p > 0 for p in p_high])
    high     = _smooth_notes(_autocorr_to_notes(p_high, v_high, 60, 96), min_hold=6)

    # --- Musical time grid quantization ---
    print("  Detecting tempo and onset grid…")
    tempo, _ = librosa.beat.beat_track(y=harmonic, sr=sr, hop_length=hop_length)
    tempo_val = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
    # frames per 16th note = (beats/min → beats/frame) / 4 subdivisions
    frames_per_beat = (sr / hop_length) * (60.0 / max(tempo_val, 1.0))
    frames_per_16th = max(1, int(round(frames_per_beat / 4)))
    print(f"    Tempo={tempo_val:.1f} BPM  →  {frames_per_beat:.1f} frames/beat  "
          f"→  {frames_per_16th} frames/16th-note")

    onset_raw    = librosa.onset.onset_detect(y=harmonic, sr=sr,
                                              hop_length=hop_length, units='frames')
    onset_frames = set(int(f) for f in onset_raw)

    melody = _quantize_to_grid(melody, onset_frames, frames_per_16th)
    bass   = _quantize_to_grid(bass,   onset_frames, frames_per_16th)
    high   = _quantize_to_grid(high,   onset_frames, frames_per_16th)

    # Voice density control: suppress high lead when it overlaps melody
    high = _suppress_overlapping_voices(melody, high)

    n_overlap = sum(1 for m, h in zip(melody, high) if m != 0 and h != 0)
    n_total   = sum(1 for m in melody if m != 0)
    pct = 100 * n_overlap / max(n_total, 1)
    print(f"    Voice 2 active simultaneously with melody: {pct:.0f}% of melody frames")

    # Per-frame waveform matching (Mahoney's correlateWaveforms, scaled to 3 SID waveforms)
    print("  Matching waveforms per frame…")
    ctrl_mel  = _match_waveform_per_frame(harmonic, sr, melody, hop_length)
    ctrl_bass = _match_waveform_per_frame(bass_sig, sr, bass,   hop_length)
    ctrl_high = _match_waveform_per_frame(mid_sig,  sr, high,   hop_length)

    # analyze_instrument for ADSR only; override its waveform with the matched dominant
    print("  Analysing instruments for ADSR…")
    cfg_mid  = analyze_instrument(harmonic, sr, 'mid')
    cfg_bass = analyze_instrument(bass_sig, sr, 'bass')
    cfg_high = analyze_instrument(mid_sig,  sr, 'high')

    for cfg, ctrl in [(cfg_mid, ctrl_mel), (cfg_bass, ctrl_bass), (cfg_high, ctrl_high)]:
        voiced_ctrl = [c for c in ctrl if c != 0x10] or [0x10]
        cfg['waveform'] = Counter(voiced_ctrl).most_common(1)[0][0]

    print(f"    Voice 0 melody: {cfg_mid['name']}  dominant waveform={hex(cfg_mid['waveform'])}")
    print(f"    Voice 1 bass:   {cfg_bass['name']}  dominant waveform={hex(cfg_bass['waveform'])}")
    print(f"    Voice 2 high:   {cfg_high['name']}  dominant waveform={hex(cfg_high['waveform'])}")

    return melody, bass, high, [ctrl_mel, ctrl_bass, ctrl_high], [cfg_mid, cfg_bass, cfg_high]


def build_sid_program(voice_notes: list[list[int]], total_frames: int,
                      voice_configs: list[dict] | None = None,
                      voice_ctrl: list[list[int]] | None = None) -> tuple[bytes, int, int]:
    """
    Build 6502 machine code for a PSID tune.

    voice_notes: 3 lists of MIDI notes (0=silence), one per frame.
    Returns (machine_code_bytes, init_addr, play_addr).

    Memory layout (all relative to LOAD_ADDR = $1000):
      $1000        INIT routine  (~72 bytes)
      $1000+init   PLAY routine  (~160 bytes)
      $2000        NOTE_TABLE    (total_frames × 9 bytes)
    """
    LOAD_ADDR  = 0x1000
    NOTE_TABLE = 0x2000   # fixed: well above PLAY, plenty of room

    def lo(addr): return addr & 0xFF
    def hi(addr): return (addr >> 8) & 0xFF

    # Resolve waveforms from voice_configs (or sensible defaults)
    defaults = [
        {'waveform': 0x10, 'atdec': 0x09, 'sustrel': 0xA4},  # triangle melody
        {'waveform': 0x20, 'atdec': 0x08, 'sustrel': 0x64},  # sawtooth bass
        {'waveform': 0x40, 'atdec': 0x09, 'sustrel': 0x84},  # pulse high
    ]
    cfgs = voice_configs if voice_configs else defaults
    waveforms = [cfgs[v]['waveform'] for v in range(3)]
    gate_on   = [w | 0x01 for w in waveforms]
    gate_off  = [w & ~0x01 for w in waveforms]

    # ------------------------------------------------------------------
    # Note data blob at NOTE_TABLE
    # Each frame: 3 voices × 3 bytes = (freq_lo, freq_hi, gate_ctrl)
    # ------------------------------------------------------------------
    melody, bass, harmony = voice_notes   # unpack named voices

    note_data = bytearray()
    for frame in range(total_frames):
        for v, voice in enumerate([melody, bass, harmony]):
            midi = voice[frame] if frame < len(voice) else 0
            # Use per-frame matched waveform if available, else fall back to voice default
            if voice_ctrl and frame < len(voice_ctrl[v]):
                w = voice_ctrl[v][frame]
            else:
                w = waveforms[v]
            if midi > 0:
                reg = hz_to_sid_freq(midi_to_hz(midi))
                note_data += bytes([reg & 0xFF, (reg >> 8) & 0xFF, w | 0x01])
            else:
                note_data += bytes([0, 0, w & ~0x01])

    # ------------------------------------------------------------------
    # INIT routine at LOAD_ADDR
    # Zero-page vars:  $02/$03 = 16-bit frame counter
    #                  $04–$07 = scratch for PLAY
    # ------------------------------------------------------------------
    init_code = bytearray()
    init_code += bytes([0xA9, 0x00,
                        0x8D, lo(SID_BASE + VOL_FILTER), hi(SID_BASE + VOL_FILTER)])
    for v in range(3):
        init_code += bytes([0xA9, gate_off[v],
                            0x8D, lo(SID_BASE + CTRL[v]), hi(SID_BASE + CTRL[v])])
    for v in range(3):
        ad = cfgs[v]['atdec']
        sr_ = cfgs[v]['sustrel']
        init_code += bytes([0xA9, ad,  0x8D, lo(SID_BASE + ATDEC[v]),   hi(SID_BASE + ATDEC[v]),
                            0xA9, sr_, 0x8D, lo(SID_BASE + SUSTREL[v]), hi(SID_BASE + SUSTREL[v])])
        # Set 50% pulse width for any pulse-waveform voice
        if waveforms[v] == 0x40:
            init_code += bytes([0xA9, 0x00, 0x8D, lo(SID_BASE + PW_LO[v]), hi(SID_BASE + PW_LO[v]),
                                0xA9, 0x08, 0x8D, lo(SID_BASE + PW_HI[v]), hi(SID_BASE + PW_HI[v])])
    init_code += bytes([0xA9, 0x00, 0x85, 0x02, 0x85, 0x03])   # zero frame counter
    init_code += bytes([0xA9, 0x0F,
                        0x8D, lo(SID_BASE + VOL_FILTER), hi(SID_BASE + VOL_FILTER)])
    init_code += bytes([0x60])   # RTS

    INIT_ADDR = LOAD_ADDR
    PLAY_ADDR = LOAD_ADDR + len(init_code)

    # ------------------------------------------------------------------
    # PLAY routine at PLAY_ADDR
    #
    # Frame check with short branches + nearby JMP for out-of-range case:
    #   LDA $03 : CMP #hi_byte
    #     BCC  ok          ; hi < total_hi → not finished
    #     BNE  goto_done   ; hi > total_hi → finished (short branch)
    #   LDA $02 : CMP #lo_byte
    #     BCC  ok          ; lo < total_lo → not finished
    #   goto_done: JMP done_abs
    #   ok:  ... compute address, drive SID, increment counter ...
    #         RTS
    #   done_abs: silence voices + RTS
    # ------------------------------------------------------------------
    total_hi = (total_frames >> 8) & 0xFF
    total_lo = total_frames & 0xFF

    play_code = bytearray()
    pc = [0]

    def emit(*b):
        play_code.extend(b)
        pc[0] += len(b)

    def fix_branch(pos, target):
        rel = target - (pos + 2)
        assert -128 <= rel <= 127, f"Branch out of range: pos={pos} target={target} rel={rel}"
        play_code[pos + 1] = rel & 0xFF

    emit(0xA5, 0x03)            # LDA $03  (frame hi)
    emit(0xC9, total_hi)        # CMP #hi
    bcc1_pos = pc[0];  emit(0x90, 0x00)   # BCC ok
    bne_pos  = pc[0];  emit(0xD0, 0x00)   # BNE goto_done
    emit(0xA5, 0x02)            # LDA $02  (frame lo)
    emit(0xC9, total_lo)        # CMP #lo
    bcc2_pos = pc[0];  emit(0x90, 0x00)   # BCC ok

    goto_done_offset = pc[0]    # BNE branches here
    jmp_done_pos = pc[0];  emit(0x4C, 0x00, 0x00)   # JMP done (abs, patched later)

    fix_branch(bne_pos, goto_done_offset)

    ok_offset = pc[0]
    fix_branch(bcc1_pos, ok_offset)
    fix_branch(bcc2_pos, ok_offset)

    # Compute pointer: $06/$07 = NOTE_TABLE + frame*9
    # frame (16-bit) in $02/$03; scratch in $04–$07
    emit(0xA5, 0x02); emit(0x85, 0x04)     # $04 = frame lo
    emit(0xA5, 0x03); emit(0x85, 0x05)     # $05 = frame hi
    emit(0xA5, 0x04); emit(0x85, 0x06)     # $06 = frame lo  (will become frame*8)
    emit(0xA5, 0x05); emit(0x85, 0x07)     # $07 = frame hi
    for _ in range(3):                      # $06/$07 <<= 1  (three times = *8)
        emit(0x06, 0x06)                    # ASL $06
        emit(0x26, 0x07)                    # ROL $07
    emit(0x18)                              # CLC
    emit(0xA5, 0x06); emit(0x65, 0x04); emit(0x85, 0x06)   # $06 += $04  (+ frame lo)
    emit(0xA5, 0x07); emit(0x65, 0x05); emit(0x85, 0x07)   # $07 += $05  (+ frame hi)
    NOTE_LO = NOTE_TABLE & 0xFF
    NOTE_HI = (NOTE_TABLE >> 8) & 0xFF
    emit(0x18)
    emit(0xA5, 0x06); emit(0x69, NOTE_LO); emit(0x85, 0x06)
    emit(0xA5, 0x07); emit(0x69, NOTE_HI); emit(0x85, 0x07)

    # Drive SID voices via ($06),Y indirect indexed
    for v in range(3):
        base_y = v * 3
        emit(0xA0, base_y);     emit(0xB1, 0x06)   # LDY #n : LDA ($06),Y
        emit(0x8D, lo(SID_BASE + FREQ_LO[v]), hi(SID_BASE + FREQ_LO[v]))
        emit(0xA0, base_y + 1); emit(0xB1, 0x06)
        emit(0x8D, lo(SID_BASE + FREQ_HI[v]), hi(SID_BASE + FREQ_HI[v]))
        emit(0xA0, base_y + 2); emit(0xB1, 0x06)
        emit(0x8D, lo(SID_BASE + CTRL[v]),    hi(SID_BASE + CTRL[v]))

    # Increment 16-bit frame counter
    emit(0xE6, 0x02)            # INC $02
    emit(0xD0, 0x03)            # BNE +3 (skip INC $03 + NOP)
    emit(0xE6, 0x03)            # INC $03
    emit(0xEA)                  # NOP  ← branch target

    emit(0x60)                  # RTS

    # done: silence all voices, RTS
    done_offset = pc[0]
    for v in range(3):
        emit(0xA9, gate_off[v])
        emit(0x8D, lo(SID_BASE + CTRL[v]), hi(SID_BASE + CTRL[v]))
    emit(0x60)                  # RTS

    # Patch JMP done with absolute address
    done_abs = PLAY_ADDR + done_offset
    play_code[jmp_done_pos + 1] = done_abs & 0xFF
    play_code[jmp_done_pos + 2] = (done_abs >> 8) & 0xFF

    # ------------------------------------------------------------------
    # Assemble: init | play | padding to NOTE_TABLE | note_data
    # ------------------------------------------------------------------
    prog = bytearray(init_code) + bytearray(play_code)
    note_offset = NOTE_TABLE - LOAD_ADDR
    assert len(prog) <= note_offset, (
        f"Code ({len(prog)} bytes) overruns NOTE_TABLE at offset {note_offset:#x}")
    while len(prog) < note_offset:
        prog += bytes([0xEA])   # NOP fill
    prog += note_data

    return bytes(prog), INIT_ADDR, PLAY_ADDR


# ---------------------------------------------------------------------------
# PSID v2 header
# ---------------------------------------------------------------------------

def build_psid(machine_code: bytes,
               load_addr: int,
               init_addr: int,
               play_addr: int,
               title: str = "Converted",
               author: str = "audio_to_sid",
               released: str = "2024") -> bytes:
    """Build a PSID v2 binary."""
    # PSID header is 124 bytes (v2)
    MAGIC  = b'PSID'
    VERSION = 2
    DATA_OFFSET = 0x7C          # 124 = standard v2 header length
    SONGS = 1
    START_SONG = 1
    SPEED = 0                   # bit 0: 0 = 50Hz VBI
    FLAGS = 0x0002              # PAL

    title_b    = title[:31].encode('ascii', errors='replace').ljust(32, b'\x00')
    author_b   = author[:31].encode('ascii', errors='replace').ljust(32, b'\x00')
    released_b = released[:31].encode('ascii', errors='replace').ljust(32, b'\x00')

    header = struct.pack('>4sHHHHHIHHI',
        MAGIC,
        VERSION,
        DATA_OFFSET,
        load_addr,
        init_addr,
        play_addr,
        SONGS | (START_SONG << 16),
        SPEED,
        FLAGS,
        0,          # startPage (v2 extra, unused)
    )
    # The struct above is 4+2+2+2+2+2+4+2+2+4 = 26 bytes; we need 124 total
    # Let's build it properly field by field:
    hdr = bytearray(DATA_OFFSET)
    hdr[0:4]   = b'PSID'
    struct.pack_into('>H', hdr, 4, VERSION)
    struct.pack_into('>H', hdr, 6, DATA_OFFSET)
    struct.pack_into('>H', hdr, 8, load_addr)
    struct.pack_into('>H', hdr, 10, init_addr)
    struct.pack_into('>H', hdr, 12, play_addr)
    struct.pack_into('>H', hdr, 14, SONGS)
    struct.pack_into('>H', hdr, 16, START_SONG)
    struct.pack_into('>I', hdr, 18, SPEED)
    hdr[22:54]  = title_b
    hdr[54:86]  = author_b
    hdr[86:118] = released_b
    struct.pack_into('>H', hdr, 118, FLAGS)
    struct.pack_into('>H', hdr, 120, 0)   # startPage
    struct.pack_into('>H', hdr, 122, 0)   # pageLength / secondSidAddress

    return bytes(hdr) + machine_code


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Convert audio to C64 PSID SID file')
    parser.add_argument('input',  help='Input audio file (wav/mp3/ogg/flac/…)')
    parser.add_argument('output', help='Output .sid file')
    parser.add_argument('--title',    default='Converted', help='SID tune title')
    parser.add_argument('--author',   default='audio_to_sid', help='Author name')
    parser.add_argument('--released', default='2024', help='Release year/info')
    parser.add_argument('--max-frames', type=int, default=6000,
                        help='Maximum frames to encode (default 6000 = 2 minutes at 50Hz)')
    args = parser.parse_args()

    SR = 22050
    # hop_length sized so that one librosa frame ≈ one SID frame (1/50s)
    HOP = SR // FRAME_RATE    # 441 samples

    print(f"Loading {args.input}…")
    y, sr = load_audio(args.input, sr=SR)
    duration_s = len(y) / sr
    n_frames = min(int(duration_s * FRAME_RATE), args.max_frames)
    print(f"  Duration: {duration_s:.1f}s  →  {n_frames} SID frames")

    print("Extracting voices…")
    melody, bass, high, voice_ctrl, voice_configs = extract_voices(
        y, sr, hop_length=HOP, n_frames=n_frames)
    voice_notes = (melody, bass, high)

    print(f"  Encoding {n_frames} frames ({n_frames/FRAME_RATE:.1f}s)…")

    LOAD_ADDR = 0x1000

    machine_code, init_addr, play_addr = build_sid_program(
        voice_notes, n_frames, voice_configs, voice_ctrl)
    psid_data = build_psid(
        machine_code,
        load_addr  = LOAD_ADDR,
        init_addr  = init_addr,
        play_addr  = play_addr,
        title      = args.title,
        author     = args.author,
        released   = args.released,
    )

    with open(args.output, 'wb') as f:
        f.write(psid_data)

    print(f"Written {len(psid_data)} bytes to {args.output}")
    print(f"  Load: ${LOAD_ADDR:04X}  Init: ${init_addr:04X}  Play: ${play_addr:04X}")
    print("Done. Open in SIDPLAY2/JSIDPLAY2 or run on real hardware.")


if __name__ == '__main__':
    main()
