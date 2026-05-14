#!/usr/bin/env python3
"""
audio_to_sid.py — Convert an audio file to a PSID-format C64 SID file.

Pipeline:
  1. Load audio; separate harmonic and percussive components (HPSS)
  2. Split harmonic signal into bass / mid / high bands
  3. Analyse each band with spectral features to identify the dominant
     instrument and choose the best SID waveform + ADSR automatically
  4. pyin pitch tracking per band → three voice note lists
  5. Smooth/quantise notes; map to SID register values
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


def _f0_to_notes(f0: np.ndarray, voiced: np.ndarray,
                 midi_lo: int, midi_hi: int, n_frames: int) -> list[int]:
    """Convert pyin f0 array to a list of clamped MIDI notes (0 = silence)."""
    notes = []
    for i in range(min(len(f0), n_frames)):
        if voiced[i] and f0[i] > 0 and not np.isnan(f0[i]):
            midi = hz_to_midi(f0[i])
            # Shift octaves until in range rather than hard-clamp
            while midi < midi_lo:
                midi += 12
            while midi > midi_hi:
                midi -= 12
            midi = max(midi_lo, min(midi_hi, midi))
        else:
            midi = 0
        notes.append(midi)
    while len(notes) < n_frames:
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


def extract_voices(y: np.ndarray, sr: int,
                   hop_length: int, n_frames: int) -> tuple[list, list, list, list]:
    """
    Return three per-frame MIDI note lists (0 = silence) and a list of three
    voice config dicts (waveform, atdec, sustrel, name) chosen by instrument
    analysis of each frequency band:

      voice 0 — mid band  C3–C6  (melody / lead)
      voice 1 — bass band C1–C3  (bass instrument)
      voice 2 — high band C4–C7  (upper harmonic / counter-melody)
    """
    print("  Separating harmonic / percussive components…")
    harmonic, _ = librosa.effects.hpss(y, margin=3.0)

    bass_sig = _lowpass(harmonic, sr, 450.0)
    mid_sig  = _bandpass(harmonic, sr, 450.0, 2000.0)
    high_sig = _highpass(harmonic, sr, 2000.0)

    # --- Instrument analysis per band ---
    print("  Analysing instruments…")
    cfg_mid  = analyze_instrument(mid_sig,  sr, 'mid')
    cfg_bass = analyze_instrument(bass_sig, sr, 'bass')
    cfg_high = analyze_instrument(high_sig, sr, 'high')
    print(f"    Voice 0 (mid):  {cfg_mid['name']}")
    print(f"    Voice 1 (bass): {cfg_bass['name']}")
    print(f"    Voice 2 (high): {cfg_high['name']}")

    # --- Voice 0: mid-range melody, C3–C6 ---
    print("  Tracking melody (C3–C6)…")
    f0_mel, voiced_mel, _ = librosa.pyin(
        harmonic,
        fmin=librosa.note_to_hz('C3'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr, hop_length=hop_length)
    melody = _smooth_notes(_f0_to_notes(f0_mel, voiced_mel, 48, 84, n_frames), min_hold=4)

    # --- Voice 1: bass, C1–C3 ---
    print("  Tracking bass (C1–C3)…")
    f0_bass, voiced_bass, _ = librosa.pyin(
        bass_sig,
        fmin=librosa.note_to_hz('C1'),
        fmax=librosa.note_to_hz('C3'),
        sr=sr, hop_length=hop_length)
    bass = _smooth_notes(_f0_to_notes(f0_bass, voiced_bass, 24, 48, n_frames), min_hold=4)

    # --- Voice 2: high-mid counter-melody, C4–C7 ---
    # Use voiced_prob > 0.6 to suppress low-confidence spurious detections,
    # and a longer min_hold to reduce note jitter on this voice.
    print("  Tracking high lead (C4–C7)…")
    f0_high, voiced_high, voiced_prob = librosa.pyin(
        mid_sig,
        fmin=librosa.note_to_hz('C4'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop_length)
    voiced_high_strict = voiced_high & (voiced_prob > 0.6)
    high = _smooth_notes(_f0_to_notes(f0_high, voiced_high_strict, 60, 96, n_frames), min_hold=6)

    # voice_configs ordered to match (voice0, voice1, voice2)
    return melody, bass, high, [cfg_mid, cfg_bass, cfg_high]


def build_sid_program(voice_notes: list[list[int]], total_frames: int,
                      voice_configs: list[dict] | None = None) -> tuple[bytes, int, int]:
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
            if midi > 0:
                reg = hz_to_sid_freq(midi_to_hz(midi))
                note_data += bytes([reg & 0xFF, (reg >> 8) & 0xFF, gate_on[v]])
            else:
                note_data += bytes([0, 0, gate_off[v]])

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
    melody, bass, high, voice_configs = extract_voices(y, sr, hop_length=HOP, n_frames=n_frames)
    voice_notes = (melody, bass, high)

    print(f"  Encoding {n_frames} frames ({n_frames/FRAME_RATE:.1f}s)…")

    LOAD_ADDR = 0x1000

    machine_code, init_addr, play_addr = build_sid_program(voice_notes, n_frames, voice_configs)
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
