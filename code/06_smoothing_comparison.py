"""
06_smoothing_comparison.py — Signal Smoothing (Savitzky-Golay) Comparison

Purpose:
  Compare RF performance WITH vs WITHOUT Savitzky-Golay smoothing on RSSI,
  across all imbalance strategies (baseline, SMOTE, RUS, class_weight).

  Savitzky-Golay filter: fits a low-degree polynomial to a sliding window
  of RSSI values, preserving trend while reducing high-frequency noise.
  Applied PER-BEACON on the timestamp-sorted RSSI time series BEFORE
  sliding window feature extraction.

  This addresses "signal smoothing techniques" mentioned in the abstract,
  and does NOT create synthetic samples (unlike SMOTE).

Strategies (8 total):
  [U]nsmoothed: Baseline, SMOTE, RUS, class_weight (V2)
  [S]moothed:   Baseline+SG, SMOTE+SG, RUS+SG, class_weight+SG (V3)

Usage: python code/06_smoothing_comparison.py
"""

import pandas as pd
import numpy as np
import time, os, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, '.cache')
WINDOW = 5

print("=" * 70)
print("06: Signal Smoothing Comparison (Savitzky-Golay)")
print("=" * 70)
t0 = time.time()

if not HAS_SCIPY:
    print("ERROR: scipy not installed. Run: pip install scipy")
    exit(1)

# ---------------------------------------------------------------------------
# 1. Load raw per-timestamp BLE
# ---------------------------------------------------------------------------
os.makedirs(CACHE, exist_ok=True)
X_raw_file = os.path.join(CACHE, 'X_raw.npy')
X_smooth_file = os.path.join(CACHE, 'X_smooth.npy')
y_file = os.path.join(CACHE, 'y.npy')
le_file = os.path.join(CACHE, 'classes.npy')

if (os.path.exists(X_raw_file) and os.path.exists(X_smooth_file)
        and os.path.exists(y_file) and os.path.exists(le_file)):
    print("Loading cached data...")
    X_raw = np.load(X_raw_file)
    X_smooth = np.load(X_smooth_file)
    y = np.load(y_file)
    classes = np.load(le_file, allow_pickle=True)
    le = LabelEncoder(); le.classes_ = classes; y_e = le.transform(y)
    n_classes = len(classes)
else:
    print("Loading BLE data...")
    BLE_DATA = os.path.join(BASE, 'ble_train_4d.csv')
    LABEL_FILE = os.path.join(BASE, 'cleaned_labels.csv')

    reader = pd.read_csv(BLE_DATA,
        usecols=lambda c: c in ['timestamp'] + [f'RSSI_{i}' for i in range(1, 26)],
        dtype={f'RSSI_{i}': 'float32' for i in range(1, 26)},
        parse_dates=['timestamp'], chunksize=100000, low_memory=False)

    chunks = []
    for chunk in reader:
        rssi = [f'RSSI_{i}' for i in range(1, 26)]
        chunk[rssi] = chunk[rssi].replace(0, np.nan)
        chunks.append(chunk.groupby('timestamp')[rssi].mean())

    ble_ts = pd.concat(chunks)
    ble_ts = ble_ts[~ble_ts.index.duplicated(keep='first')].fillna(-100)
    timestamps = ble_ts.index

    # --- Apply Savitzky-Golay filter per beacon ---
    rssi_cols = [f'RSSI_{i}' for i in range(1, 26)]
    ble_smooth = ble_ts.copy()
    for col in rssi_cols:
        ble_smooth[col] = savgol_filter(ble_smooth[col].values, window_length=5, polyorder=2)

    # --- Merge labels ---
    labels = pd.read_csv(LABEL_FILE, parse_dates=['started_at', 'finished_at'])
    ble_ts = ble_ts.reset_index()
    ble_smooth = ble_smooth.reset_index()

    for df in [ble_ts, ble_smooth]:
        df['room'] = 'unknown'
        for _, r in labels.iterrows():
            m = (df['timestamp'] >= r['started_at']) & (df['timestamp'] <= r['finished_at'])
            df.loc[m, 'room'] = r['room']

    train_raw = ble_ts[ble_ts['room'] != 'unknown'].copy()
    train_smooth = ble_smooth[ble_smooth['room'] != 'unknown'].copy()
    del ble_ts, ble_smooth, labels, reader, chunks

    def sliding_window_features(values, rooms_arr):
        X_out, y_out = [], []
        for i in range(len(values) - WINDOW + 1):
            w = values[i:i+WINDOW]
            f = []
            for j in range(25):
                c = w[:, j]
                f += [float(np.mean(c)), float(np.std(c)), float(np.min(c)), float(np.max(c))]
            X_out.append(f)
            y_out.append(rooms_arr[i + WINDOW // 2])
        return np.array(X_out), np.array(y_out)

    rssi_cols_list = rssi_cols
    X_raw, y = sliding_window_features(train_raw[rssi_cols_list].values, train_raw['room'].values)
    X_smooth, y2 = sliding_window_features(train_smooth[rssi_cols_list].values, train_smooth['room'].values)
    assert list(y) == list(y2), "Labels mismatch after smoothing"
    del train_raw, train_smooth

    le = LabelEncoder()
    y_e = le.fit_transform(y)
    n_classes = len(le.classes_)

    np.save(X_raw_file, X_raw)
    np.save(X_smooth_file, X_smooth)
    np.save(y_file, y)
    np.save(le_file, le.classes_)
    print(f"Cached to {CACHE}/")

# ---------------------------------------------------------------------------
# 2. Class distribution info
# ---------------------------------------------------------------------------
uniq, cnts = np.unique(y, return_counts=True)
imbalance_ratio = max(cnts) / min(cnts)
print(f"\nSamples: {len(X_raw)}, Features: {X_raw.shape[1]}, Classes: {n_classes}")
print(f"Imbalance ratio: {imbalance_ratio:.1f}x (max/min)")

rare_names = ['503', '505', '510', '516', '517', '518']
rare_ids = [list(le.classes_).index(r) for r in le.classes_ if r in rare_names]

# ---------------------------------------------------------------------------
# 3. CV evaluation
# ---------------------------------------------------------------------------
def run_cv_fold(tr, va, X, y, sampler_type, cw):
    X_tr, y_tr = X[tr], y[tr]
    if sampler_type == 'smote':
        from collections import Counter
        orig = Counter(y_tr)
        strategy = {c: max(2000, n) for c, n in orig.items()}
        sm = SMOTE(sampling_strategy=strategy, k_neighbors=1, random_state=42)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
    elif sampler_type == 'rus':
        rus = RandomUnderSampler(random_state=42)
        X_tr, y_tr = rus.fit_resample(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=100, max_depth=20,
        class_weight=cw, random_state=42, n_jobs=1)
    rf.fit(X_tr, y_tr)
    p_val = rf.predict(X[va])
    p_tr = rf.predict(X_tr)
    wf = f1_score(y[va], p_val, average='weighted')
    mf = f1_score(y[va], p_val, average='macro')
    twf = f1_score(y_tr, p_tr, average='weighted')
    rare_mask = np.isin(y[va], rare_ids)
    rf1 = f1_score(y[va][rare_mask], p_val[rare_mask], average='macro') if rare_mask.sum() > 0 else 0.0
    del rf, X_tr, y_tr, p_val, p_tr; gc.collect()
    return wf, mf, twf, rf1

def evaluate_strategy(X, y, name, sampler=None, cw=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    wf_s, mf_s, rare_s, train_s = [], [], [], []
    for tr, va in skf.split(X, y):
        wf, mf, twf, rf1 = run_cv_fold(tr, va, X, y, sampler, cw)
        wf_s.append(wf); mf_s.append(mf); train_s.append(twf); rare_s.append(rf1)
    return {'name': name, 'W-F1': np.mean(wf_s), 'M-F1': np.mean(mf_s),
            'rare_F1': np.mean(rare_s), 'train_W-F1': np.mean(train_s),
            'overfit_gap': np.mean(train_s) - np.mean(wf_s)}

# ---------------------------------------------------------------------------
# 4. Run all strategies on both unsmoothed and smoothed data
# ---------------------------------------------------------------------------
strategies = [
    ('[U] Baseline (no balancing)',    None, None),
    ('[U] SMOTE (oversampling)',       'smote', None),
    ('[U] RandomUnderSampler',         'rus', None),
    ("[U] class_weight (V2)",          None, 'balanced_subsample'),
    ('[S] Baseline + SG smooth',       None, None),
    ('[S] SMOTE + SG smooth',          'smote', None),
    ('[S] RUS + SG smooth',            'rus', None),
    ("[S] class_weight + SG smooth (V3)", None, 'balanced_subsample'),
]

print(f"\n{'='*70}")
print(f"{'Strategy':<38} {'W-F1':>8} {'M-F1':>8} {'Rare-F1':>8} {'Train':>8} {'Gap':>9}")
print(f"{'='*70}")

results = []
for name, sampler, cw in strategies:
    is_smooth = name.startswith('[S]')
    X_use = X_smooth if is_smooth else X_raw
    if sampler == 'smote' and not HAS_IMBLEARN:
        print(f"{name:<38} {'SKIP':>8} (install imbalanced-learn)")
        continue
    r = evaluate_strategy(X_use, y_e, name, sampler, cw)
    results.append(r)
    gap = r['overfit_gap']
    gap_str = f"OVERFIT!" if gap > 0.12 else f"{gap:.4f}"
    print(f"{r['name']:<38} {r['W-F1']:>8.4f} {r['M-F1']:>8.4f} {r['rare_F1']:>8.4f} {r['train_W-F1']:>8.4f} {gap_str:>9}")
    gc.collect()

print(f"{'='*70}")

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for r in results:
    note = ""
    if r['overfit_gap'] > 0.12:
        note = " OVERFIT"
    if '[S]' in r['name']:
        note += " (smoothed)"
    best_marker = " <<< BEST" if (r['W-F1'] >= max([x['W-F1'] for x in results]) - 0.001) else ""
    print(f"  {r['name']:<38}: W-F1={r['W-F1']:.4f} M-F1={r['M-F1']:.4f} Rare-F1={r['rare_F1']:.4f} Gap={r['overfit_gap']:.4f}{note}{best_marker}")

# Compare SG lift
print(f"\n{'='*70}")
print("SG FILTER LIFT (Smoothed - Unsmoothed)")
print(f"{'='*70}")
pairs = [
    ("[U] Baseline (no balancing)",    "[S] Baseline + SG smooth"),
    ("[U] SMOTE (oversampling)",       "[S] SMOTE + SG smooth"),
    ("[U] RandomUnderSampler",         "[S] RUS + SG smooth"),
    ("[U] class_weight (V2)",          "[S] class_weight + SG smooth (V3)"),
]
for u_key, s_key in pairs:
    u = next(r for r in results if r['name'] == u_key)
    s = next(r for r in results if r['name'] == s_key)
    dw = s['W-F1'] - u['W-F1']
    dm = s['M-F1'] - u['M-F1']
    dr = s['rare_F1'] - u['rare_F1']
    label_short = u_key.replace('[U] ', '')
    print(f"  {label_short:<30}: W-F1 {dw:+.4f}  M-F1 {dm:+.4f}  Rare-F1 {dr:+.4f}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
