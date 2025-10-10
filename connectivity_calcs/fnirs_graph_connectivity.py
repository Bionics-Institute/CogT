# %%
# fnirs_graph_connectivity.py
#
# Compute graph-theoretic connectivity metrics (clustering coefficient and characteristic
# path length) from fNIRS time-series, following Duan et al. (2012).
#
# Inputs:
#   - time_series: 2D numpy array of shape (n_channels, n_timepoints)
# Optional:
#   - fs: sampling rate in Hz (if you want to band-pass filter; otherwise set to None)
#   - band: (low, high) frequency band for band-pass filtering (default: 0.01–0.08 Hz)
#   - detrend: remove linear trend per channel (default: True)
#   - use_abs: use absolute correlations for thresholding (default: True)
#   - thresholds: iterable of percentage thresholds to keep strongest edges (default: range(10, 51))
#
# Outputs:
#   - metrics_df: pandas DataFrame with columns: ['threshold_pct', 'CP', 'LP', 'n_nodes', 'n_edges',
#         'is_connected', 'connected_fraction']
#   - corr_mat: the channel × channel correlation matrix used for thresholding
#
# Example:
#   from fnirs_graph_connectivity import compute_fnirs_graph_metrics
#   metrics, corr = compute_fnirs_graph_metrics(time_series, fs=10.0)
#
# Notes:
#   - For LP on potentially disconnected graphs, we compute average shortest-path length on the
#     largest connected component (LCC) and report the fraction of nodes in that component.
#   - Duan et al. used a “top-percentage” threshold on the correlation matrix to build binary graphs.
#   - If you also have an fMRI-derived correlation matrix mapped to the same channels, you can compute
#     BMS_CP and BMS_LP consistency scores with `compute_bms_against_reference`.
#
from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from scipy import signal


def _bandpass_filter(x: np.ndarray, fs: float, band: Tuple[float, float]) -> np.ndarray:
    """Zero-phase band-pass filter each row in x (channels × time)."""
    low, high = band
    nyq = 0.5 * fs
    if low <= 0 or high >= nyq:
        raise ValueError(f"Band {band} must be within (0, {nyq}) for fs={fs}.")
    # 2nd-order Butterworth in each direction → effectively 4th order
    sos = signal.butter(2, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x, axis=1)


def _detrend(x: np.ndarray) -> np.ndarray:
    return signal.detrend(x, axis=1, type="linear")


def _corrcoef_channels(x: np.ndarray) -> np.ndarray:
    """Correlation matrix (channels × channels) for time-series (channels × time)."""
    # subtract mean per channel to be safe
    xm = x - x.mean(axis=1, keepdims=True)
    # avoid all-zero variance channels
    std = xm.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    xn = xm / std
    return np.corrcoef(xn)


def _threshold_top_percentage(corr: np.ndarray, pct: float, use_abs: bool = True) -> np.ndarray:
    """
    Returns a binary symmetric adjacency (no self-loops) by keeping the top `pct` percent
    of (upper-triangle) correlation values.
    """
    if not (0 < pct <= 100):
        raise ValueError("pct must be in (0, 100].")
    n = corr.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = np.abs(corr[iu]) if use_abs else corr[iu]
    # rank edges by value
    k_keep = max(1, int(np.floor(len(vals) * (pct / 100.0))))
    # np.argpartition for efficiency
    if k_keep < len(vals):
        thresh_idx = np.argpartition(vals, -k_keep)[-k_keep:]
    else:
        thresh_idx = np.arange(len(vals))
    mask = np.zeros_like(vals, dtype=bool)
    mask[thresh_idx] = True
    # Build adjacency
    A = np.zeros((n, n), dtype=int)
    A[iu] = mask.astype(int)
    A = A + A.T
    np.fill_diagonal(A, 0)
    return A


def _graph_metrics_from_adj(A: np.ndarray) -> Tuple[float, float, int, int, bool, float]:
    """
    Compute clustering coefficient (CP) and characteristic path length (LP) from adjacency.
    LP is computed on the largest connected component (LCC) if the graph is disconnected.
    Returns: CP, LP, n_nodes, n_edges, is_connected, connected_fraction
    """
    G = nx.from_numpy_array(A)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Clustering coefficient (average over nodes)
    CP = float(np.mean(list(nx.clustering(G).values()))) if n_nodes > 1 else 0.0

    # Characteristic path length
    if nx.is_connected(G):
        LP = nx.average_shortest_path_length(G)
        is_conn = True
        frac = 1.0
    else:
        # compute on the largest connected component
        components = list(nx.connected_components(G))
        lcc = max(components, key=len)
        frac = len(lcc) / n_nodes if n_nodes else 0.0
        G_lcc = G.subgraph(lcc).copy()
        # If lcc has a single node, path length is 0 by definition; else compute
        LP = 0.0 if G_lcc.number_of_nodes() <= 1 else nx.average_shortest_path_length(G_lcc)
        is_conn = False

    return CP, LP, n_nodes, n_edges, is_conn, frac


def compute_fnirs_graph_metrics(
    time_series: np.ndarray,
    fs: Optional[float] = None,
    band: Tuple[float, float] = (0.01, 0.08),
    detrend: bool = True,
    use_abs: bool = True,
    thresholds: Iterable[int] = range(10, 51),
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute graph-theoretic metrics from fNIRS time series.
    Parameters:
      time_series: array of shape (n_channels, n_timepoints)
      fs: sampling rate in Hz. If provided, band-pass filter is applied using `band`.
      band: (low, high) in Hz for band-pass filtering (default 0.01–0.08 Hz)
      detrend: whether to linearly detrend each channel (default True)
      use_abs: threshold by absolute correlations (default True). Set False to use signed r.
      thresholds: collection of integer percentages to keep strongest edges (10–50 recommended)
    Returns:
      metrics_df: DataFrame with metrics across thresholds
      corr_mat: correlation matrix used for graph construction
    """
    if time_series.ndim != 2:
        raise ValueError("time_series must be 2D (n_channels, n_timepoints).")
    x = time_series.astype(float, copy=True)

    if detrend:
        x = _detrend(x)
    if fs is not None:
        x = _bandpass_filter(x, fs=fs, band=band)

    corr_mat = _corrcoef_channels(x)

    records = []
    for t in thresholds:
        A = _threshold_top_percentage(corr_mat, pct=float(t), use_abs=use_abs)
        CP, LP, n_nodes, n_edges, is_conn, frac = _graph_metrics_from_adj(A)
        records.append(
            dict(
                threshold_pct=t,
                CP=CP,
                LP=LP,
                n_nodes=n_nodes,
                n_edges=n_edges,
                is_connected=is_conn,
                connected_fraction=frac,
            )
        )
    metrics_df = pd.DataFrame.from_records(records).sort_values("threshold_pct").reset_index(drop=True)
    return metrics_df, corr_mat


def compute_bms_against_reference(
    cp_ref: pd.Series,
    lp_ref: pd.Series,
    metrics_df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Compute BMS (Between-Modality Similarity) scores for CP and LP against a reference
    (e.g., fMRI-derived values at the same thresholds). Thresholds must match.
    BMS = mean_t (1 - |metric_ref(t) - metric_fnirs(t)| / metric_ref(t))
    Returns: (BMS_CP, BMS_LP)
    """
    df = metrics_df.set_index("threshold_pct").copy()
    common = sorted(set(cp_ref.index).intersection(df.index))
    if not common:
        raise ValueError("No overlapping thresholds between reference and fnirs metrics.")
    bms_cp_vals, bms_lp_vals = [], []
    for t in common:
        ref_cp = float(cp_ref.loc[t])
        ref_lp = float(lp_ref.loc[t])
        fn_cp = float(df.loc[t, "CP"])
        fn_lp = float(df.loc[t, "LP"])
        # Guard against division by zero
        if ref_cp == 0 or ref_lp == 0:
            continue
        bms_cp_vals.append(1.0 - abs(ref_cp - fn_cp) / ref_cp)
        bms_lp_vals.append(1.0 - abs(ref_lp - fn_lp) / ref_lp)
    if not bms_cp_vals or not bms_lp_vals:
        raise ValueError("Reference metrics contain zeros or no overlap after filtering.")
    return float(np.mean(bms_cp_vals)), float(np.mean(bms_lp_vals))


# ---------- Demo with synthetic data (safe to run) ----------
if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n_channels, n_time = 48, 600
    fs = 10.0  # Hz (example fNIRS sampling rate)
    # Create weakly correlated block-structure signals (two communities)
    base1 = signal.sosfiltfilt(
        signal.butter(2, [0.01/(0.5*fs), 0.08/(0.5*fs)], btype="bandpass", output="sos"),
        rng.normal(size=n_time)
    )
    base2 = signal.sosfiltfilt(
        signal.butter(2, [0.01/(0.5*fs), 0.08/(0.5*fs)], btype="bandpass", output="sos"),
        rng.normal(size=n_time)
    )
    X = np.zeros((n_channels, n_time))
    for ch in range(n_channels):
        base = base1 if ch < n_channels // 2 else base2
        noise = rng.normal(scale=0.5, size=n_time)
        X[ch] = base + noise
    dlpfc_idx = [12, 13, 14, 15, 16, 17, 18]   # example indices
    ts_dlpfc = X[dlpfc_idx, :]   
    metrics, corr = compute_fnirs_graph_metrics(ts_dlpfc, fs=fs, band=(0.01, 0.08), detrend=True, use_abs=True)
    # Save outputs for inspection
    out_path_metrics = "/mnt/data/fnirs_graph_metrics_demo.csv"
    out_path_corr = "/mnt/data/fnirs_corr_matrix_demo.npy"
    metrics.to_csv(out_path_metrics, index=False)
    np.save(out_path_corr, corr)
    print("Saved demo metrics to:", out_path_metrics)
    print("Saved corr matrix to:", out_path_corr)
