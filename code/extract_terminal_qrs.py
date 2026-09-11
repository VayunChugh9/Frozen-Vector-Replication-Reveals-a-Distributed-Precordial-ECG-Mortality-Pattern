"""
12-lead terminal-QRS feature extraction engine.

Generalises the published ECG-SCD aVL terminal-QRS algorithm to all 12 leads
and any sampling rate.

Reference implementation behavior:
  - median beat from aVL, min-max normalisation
  - Hamilton segmenter for R-peak
  - Q-peak = last sign-change before R within Q_WIN_MS ms
  - S-peak = first sign-change after R within S_WIN_MS ms
  - aVL_rs_diff  = mean(|Δ1 x[Q:S]|)  (mean abs first  difference)
  - aVL_rs_diff_2= mean(|Δ2 x[Q:S]|)  (mean abs second difference)

Adaptations:
  - All window sizes specified in ms; converted to samples at runtime.
  - Lead order read from HDF5 header attribute (or assumed canonical if absent).
  - NTUH gain correction (÷4.88) is NOT applied (confirmed MIMIC/CODE-15%).
  - QS morphology (no R-to-S interval) → NaN + qc_fail flag.
  - Staircase / low-variance signal → qc_flag = "low_var".

Confirmed lead order for CODE-15% HDF5:
  Index: 0=I  1=II  2=III  3=aVR  4=aVL  5=aVF  6=V1 … 11=V6
  (same as ECG-SCD x06 LEADS; opposite to MIMIC-IV-ECG which has aVL at 5)
"""

from __future__ import annotations
import numpy as np
import warnings
from typing import Optional

# ── canonical lead ordering (ECG-SCD / CODE-15% HDF5) ─────────────────────
LEADS_CANONICAL = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
AVL_IDX_CANONICAL = 4       # index in LEADS_CANONICAL

# ── algorithm constants (milliseconds) ────────────────────────────────────
BEAT_WIN_MS      = 500      # ±500 ms median-beat window (= HALF_WIN at 500 Hz)
Q_SEARCH_MS      = 80       # search 80 ms before R for Q-peak
S_SEARCH_MS      = 82       # search 82 ms after  R for S-peak
MIN_VARIANCE     = 1e-7     # signal variance floor for QC
EPS              = 1e-12    # normalisation epsilon


def ms_to_samples(ms: float, fs: float) -> int:
    """Convert milliseconds to integer sample count at sampling rate fs."""
    return max(1, int(round(ms * fs / 1000.0)))


def get_lead_idx(lead_names: list[str], target: str) -> int:
    """Case-insensitive lookup of a lead name in a list."""
    target_up = target.upper().replace(" ", "")
    for i, n in enumerate(lead_names):
        if n.upper().replace(" ", "") == target_up:
            return i
    raise ValueError(f"Lead '{target}' not found in {lead_names}")


def detect_r_peak(signal: np.ndarray, fs: float) -> Optional[int]:
    """
    Detect the dominant R-peak in a single-lead beat window.
    Uses biosppy Hamilton segmenter; falls back to argmax.
    Returns sample index or None on failure.
    """
    from biosppy.signals import ecg as bsp_ecg
    try:
        out = bsp_ecg.hamilton_segmenter(signal=signal.astype(float), sampling_rate=fs)
        rpeaks = out[0]
        if len(rpeaks) > 0:
            # Choose the peak closest to the window centre
            centre = len(signal) // 2
            return int(rpeaks[np.argmin(np.abs(rpeaks - centre))])
    except Exception:
        pass
    # Fallback: absolute maximum
    return int(np.argmax(np.abs(signal)))


def compute_rs_features(
    lead_signal: np.ndarray,
    fs: float,
    r_idx: Optional[int] = None,
) -> dict:
    """
    Compute rs_diff and rs_diff_2 for a single normalised median-beat lead.

    Parameters
    ----------
    lead_signal : 1-D array, one lead of the median beat (already ±BEAT_WIN_MS centred)
    fs          : sampling rate in Hz
    r_idx       : pre-computed R-peak index (optional; detected if None)

    Returns
    -------
    dict with keys: rs_diff, rs_diff_2, r_idx, q_idx, s_idx, qc_flag
      qc_flag: "" (ok), "no_r", "no_qs", "low_var", "nan_signal"
    """
    res = dict(rs_diff=np.nan, rs_diff_2=np.nan, r_idx=np.nan,
               q_idx=np.nan, s_idx=np.nan, qc_flag="")

    if np.any(~np.isfinite(lead_signal)):
        res["qc_flag"] = "nan_signal"
        return res

    # ── min-max normalise ────────────────────────────────────────────────
    sig_min, sig_max = lead_signal.min(), lead_signal.max()
    rng = sig_max - sig_min
    if rng < MIN_VARIANCE:
        res["qc_flag"] = "low_var"
        return res
    x = (lead_signal - sig_min) / (rng + EPS)

    # ── R-peak ───────────────────────────────────────────────────────────
    if r_idx is None:
        r_idx = detect_r_peak(x, fs)
    if r_idx is None:
        res["qc_flag"] = "no_r"
        return res
    res["r_idx"] = r_idx

    q_win = ms_to_samples(Q_SEARCH_MS, fs)
    s_win = ms_to_samples(S_SEARCH_MS, fs)

    # ── Q-peak: last sign change before R within Q_WIN_MS ────────────────
    q_start = max(0, r_idx - q_win)
    seg_pre = x[q_start:r_idx]
    if len(seg_pre) < 2:
        res["qc_flag"] = "no_qs"
        return res
    sign_changes_q = np.where(np.diff(np.sign(np.diff(seg_pre))) != 0)[0]
    if len(sign_changes_q) > 0:
        q_idx = q_start + sign_changes_q[-1] + 1
    else:
        q_idx = q_start  # fallback: use window start
    res["q_idx"] = q_idx

    # ── S-peak: first sign change after R within S_WIN_MS ────────────────
    s_end = min(len(x), r_idx + s_win)
    seg_post = x[r_idx:s_end]
    if len(seg_post) < 2:
        res["qc_flag"] = "no_qs"
        return res
    sign_changes_s = np.where(np.diff(np.sign(np.diff(seg_post))) != 0)[0]
    if len(sign_changes_s) > 0:
        s_idx = r_idx + sign_changes_s[0] + 1
    else:
        s_idx = s_end - 1   # fallback: use window end
    res["s_idx"] = s_idx

    # ── Features over R→S interval ────────────────────────────────────────
    if s_idx <= q_idx:
        res["qc_flag"] = "no_qs"
        return res

    rs_seg = x[q_idx:s_idx + 1]
    if len(rs_seg) < 2:
        res["qc_flag"] = "no_qs"
        return res

    d1 = np.abs(np.diff(rs_seg))
    d2 = np.abs(np.diff(d1)) if len(d1) > 1 else np.array([0.0])
    res["rs_diff"]   = float(np.mean(d1))
    res["rs_diff_2"] = float(np.mean(d2))
    return res


def extract_median_beat(
    signal: np.ndarray,
    fs: float,
    lead_idx: int = AVL_IDX_CANONICAL,
) -> np.ndarray:
    """
    Compute the median beat from a multi-lead ECG.

    Parameters
    ----------
    signal   : shape (n_samples, n_leads) or (n_leads, n_samples)
    fs       : sampling rate in Hz
    lead_idx : index of reference lead (aVL by default) for R-peak detection

    Returns
    -------
    median_beat : shape (2 * HALF_WIN, n_leads), centred on median R-peak
    """
    # Normalise to (n_samples, n_leads)
    if signal.ndim == 2 and signal.shape[0] < signal.shape[1]:
        signal = signal.T
    n_samp, n_leads = signal.shape

    half_win = ms_to_samples(BEAT_WIN_MS, fs)
    ref_lead = signal[:, lead_idx].astype(float)

    # Detect all R-peaks on the reference lead
    from biosppy.signals import ecg as bsp_ecg
    try:
        out = bsp_ecg.hamilton_segmenter(signal=ref_lead, sampling_rate=fs)
        rpeaks = out[0]
    except Exception:
        rpeaks = np.array([n_samp // 2])

    # Collect beat windows
    beats = []
    for rp in rpeaks:
        lo, hi = rp - half_win, rp + half_win
        if lo >= 0 and hi <= n_samp:
            beats.append(signal[lo:hi, :])

    if len(beats) == 0:
        # Fallback: use centre window
        mid = n_samp // 2
        lo, hi = max(0, mid - half_win), min(n_samp, mid + half_win)
        return signal[lo:hi, :]

    return np.median(np.stack(beats, axis=0), axis=0)


def extract_all_leads(
    signal: np.ndarray,
    fs: float,
    lead_names: Optional[list[str]] = None,
    avl_idx: Optional[int] = None,
) -> dict:
    """
    Extract 12-lead terminal-QRS features from one ECG recording.

    Parameters
    ----------
    signal     : (n_samples, 12) or (12, n_samples)
    fs         : sampling rate in Hz
    lead_names : list of 12 lead names; defaults to LEADS_CANONICAL
    avl_idx    : index of aVL in lead_names; auto-detected if None

    Returns
    -------
    flat dict:  {rs_diff_I, rs_diff_2_I, ..., rs_diff_V6, rs_diff_2_V6,
                 r_idx_aVL, q_idx_aVL, s_idx_aVL,
                 qc_fail_any (bool), qc_flags (str)}
    """
    if lead_names is None:
        lead_names = LEADS_CANONICAL
    if avl_idx is None:
        try:
            avl_idx = get_lead_idx(lead_names, "aVL")
        except ValueError:
            avl_idx = AVL_IDX_CANONICAL

    # Normalise shape
    if signal.ndim == 2 and signal.shape[0] < signal.shape[1]:
        signal = signal.T
    if signal.ndim != 2 or signal.shape[1] != len(lead_names):
        return {"qc_fail_any": True, "qc_flags": "bad_shape"}

    median_beat = extract_median_beat(signal, fs, lead_idx=avl_idx)

    # Detect R-peak once on aVL; share with all leads
    n_beat = median_beat.shape[0]
    avl_beat = median_beat[:, avl_idx].astype(float)
    avl_min, avl_max = avl_beat.min(), avl_beat.max()
    avl_rng = avl_max - avl_min
    if avl_rng > MIN_VARIANCE:
        avl_norm = (avl_beat - avl_min) / (avl_rng + EPS)
        r_idx_shared = detect_r_peak(avl_norm, fs)
    else:
        r_idx_shared = n_beat // 2

    result = {}
    qc_flags = []
    for i, lead in enumerate(lead_names):
        lead_sig = median_beat[:, i].astype(float)
        feats = compute_rs_features(lead_sig, fs, r_idx=r_idx_shared)
        result[f"rs_diff_{lead}"]   = feats["rs_diff"]
        result[f"rs_diff_2_{lead}"] = feats["rs_diff_2"]
        if lead == "aVL":
            result["r_idx_aVL"] = feats["r_idx"]
            result["q_idx_aVL"] = feats["q_idx"]
            result["s_idx_aVL"] = feats["s_idx"]
        if feats["qc_flag"]:
            qc_flags.append(f"{lead}:{feats['qc_flag']}")

    result["qc_fail_any"] = len(qc_flags) > 0
    result["qc_flags"]    = "|".join(qc_flags)
    return result


def process_hdf5_file(
    hdf5_path: str,
    fs: float = 400.0,
    lead_names: Optional[list[str]] = None,
    batch_size: int = 200,
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    Extract features from all ECGs in a CODE-15%-style HDF5 file.

    HDF5 structure assumed:
      tracings  dataset: shape (N, n_samples, n_leads)  dtype float32
      exam_id   dataset: shape (N,)                     dtype int64

    Returns
    -------
    pd.DataFrame with columns: exam_id, rs_diff_<lead>, rs_diff_2_<lead>, ...
    """
    import h5py
    import pandas as pd
    from tqdm import tqdm

    if lead_names is None:
        lead_names = LEADS_CANONICAL

    records = []
    with h5py.File(hdf5_path, "r") as hf:
        # Inspect keys
        keys = list(hf.keys())
        if verbose:
            print(f"  HDF5 keys: {keys}")
        # Find tracings and exam_id
        trace_key = None
        for k in ["tracings", "signals", "ecg", "data"]:
            if k in hf:
                trace_key = k; break
        if trace_key is None:
            trace_key = keys[0]

        exam_key = "exam_id" if "exam_id" in hf else None
        tracings = hf[trace_key]
        exam_ids = hf[exam_key][:] if exam_key else np.arange(tracings.shape[0])

        n_total = tracings.shape[0]
        if verbose:
            print(f"  Shape: {tracings.shape}  N={n_total:,}  fs={fs} Hz")

        # Verify lead order on first record (print aVL summary)
        if verbose and n_total > 0:
            sig0 = tracings[0].astype(np.float32)
            if sig0.shape[0] < sig0.shape[1]:
                sig0 = sig0.T
            avl_sig = sig0[:, lead_names.index("aVL")]
            print(f"  aVL [0] range: [{avl_sig.min():.3f}, {avl_sig.max():.3f}]  "
                  f"mean={avl_sig.mean():.3f}")

        for start in tqdm(range(0, n_total, batch_size), desc="  batches",
                          disable=not verbose, ncols=70):
            end = min(start + batch_size, n_total)
            batch_traces = tracings[start:end].astype(np.float32)
            batch_ids    = exam_ids[start:end]
            for j in range(end - start):
                sig = batch_traces[j]
                feats = extract_all_leads(sig, fs, lead_names=lead_names)
                feats["exam_id"] = int(batch_ids[j])
                records.append(feats)

    return pd.DataFrame(records)


if __name__ == "__main__":
    import argparse, pandas as pd
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5",   required=True)
    p.add_argument("--out",    required=True)
    p.add_argument("--fs",     type=float, default=400.0)
    p.add_argument("--batch",  type=int,   default=200)
    args = p.parse_args()
    df = process_hdf5_file(args.hdf5, fs=args.fs, batch_size=args.batch)
    df.to_parquet(args.out, index=False)
    print(f"Saved {len(df):,} records → {args.out}")
    qc_fail = df["qc_fail_any"].sum()
    print(f"QC fails: {qc_fail:,} ({qc_fail/len(df)*100:.1f}%)")
    print(df[[c for c in df.columns if "rs_diff_" in c and "rs_diff_2" not in c]].describe().T)
