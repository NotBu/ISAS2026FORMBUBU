"""
09b_lodo_experiments.py — LODO with RSSI ranking + z-score normalization
"""

import pandas as pd, numpy as np, time, os, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.signal import savgol_filter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW = 5
t0 = time.time()

print("=" * 70)
print("09b: LODO Experiments — ranking + z-score + XGBoost")
print("=" * 70)

# --- Load & preprocess (same as V3) ---
print("\nLoading data...")
reader = pd.read_csv(os.path.join(BASE, 'ble_train_4d.csv'),
    usecols=lambda c: c in ['timestamp']+[f'RSSI_{i}' for i in range(1,26)],
    dtype={'timestamp': str, **{f'RSSI_{i}':'float32' for i in range(1,26)}},
    chunksize=25000, low_memory=False)
chunks = []
for chunk in reader:
    r = [f'RSSI_{i}' for i in range(1,26)]
    chunk[r] = chunk[r].replace(0, np.nan)
    chunks.append(chunk.groupby('timestamp')[r].mean())
ble = pd.concat(chunks)
ble.index = pd.to_datetime(ble.index)
ble = ble[~ble.index.duplicated(keep='first')].fillna(-100).sort_index()

rssi_cols = [f'RSSI_{i}' for i in range(1, 26)]
for col in rssi_cols:
    ble[col] = savgol_filter(ble[col].values, window_length=5, polyorder=2)

labels = pd.read_csv(os.path.join(BASE, 'cleaned_labels.csv'), parse_dates=['started_at','finished_at'])
ble = ble.reset_index()
bu = pd.Timedelta(seconds=4)
ble['room'] = 'unknown'
for _, r in labels.iterrows():
    m = (ble['timestamp']>=r['started_at']-bu)&(ble['timestamp']<=r['finished_at']+bu)
    ble.loc[m,'room'] = r['room']
train = ble[ble['room']!='unknown'].copy()
del ble, labels, reader, chunks

v = train[rssi_cols].values
rooms = train['room'].values
timestamps = train['timestamp'].values
le = LabelEncoder()
le.fit(rooms)
days = pd.Series(timestamps).dt.date
unique_days = sorted(days.unique())
N = len(v)
HW = WINDOW // 2

def sliding_features(values):
    X = []
    for i in range(len(values)-WINDOW+1):
        w = values[i:i+WINDOW]
        f = []
        for j in range(25):
            c = w[:,j]
            f += [np.mean(c), np.std(c), np.min(c), np.max(c)]
        X.append(f)
    return np.array(X)

def lodo_score(X, y_labels, day_vals, clf_fn):
    scores = []
    for day in unique_days:
        va = day_vals == day
        tr = ~va
        clf = clf_fn()
        clf.fit(X[tr], le.transform(y_labels[tr]))
        p = clf.predict(X[va])
        scores.append(f1_score(le.transform(y_labels[va]), p, average='weighted'))
    return scores

# Alignment after sliding window: skip first HW and last HW timestamps
y_aligned = rooms[HW:N-HW] if N > 2*HW else rooms
d_aligned = days.values[HW:N-HW] if N > 2*HW else days.values

# --- 1. Baseline ---
print("\n--- 1. Baseline (SG + RF) ---")
X_base = sliding_features(v)
print(f"  X: {X_base.shape}, y: {len(y_aligned)}")
scores = lodo_score(X_base, y_aligned, d_aligned,
    lambda: RandomForestClassifier(n_estimators=100, max_depth=20,
        class_weight='balanced_subsample', random_state=42, n_jobs=1))
print(f"  LODO W-F1: {np.mean(scores):.4f} ({np.std(scores):.4f})")

# --- 2. RSSI ranking ---
print("\n--- 2. RSSI Ranking ---")
rank_v = np.zeros_like(v)
for i in range(len(v)):
    r = np.argsort(np.argsort(-v[i])) + 1
    rank_v[i] = r
X_rank = sliding_features(rank_v)
scores = lodo_score(X_rank, y_aligned, d_aligned,
    lambda: RandomForestClassifier(n_estimators=100, max_depth=20,
        class_weight='balanced_subsample', random_state=42, n_jobs=1))
print(f"  RF  LODO W-F1: {np.mean(scores):.4f} ({np.std(scores):.4f})")

try:
    import xgboost as xgb
    scores = []
    for day in unique_days:
        va = d_aligned == day
        tr = ~va
        y_tr = le.transform(y_aligned[tr])
        # manually map to contiguous range, then map back
        uniq = np.unique(y_tr)
        mapping = {o: n for n, o in enumerate(uniq)}
        inv_map = {n: o for o, n in mapping.items()}
        y_mapped = np.array([mapping[v] for v in y_tr])
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=6,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=42)
        clf.fit(X_rank[tr], y_mapped)
        p_mapped = clf.predict(X_rank[va])
        p = np.array([inv_map[v] for v in p_mapped])
        scores.append(f1_score(le.transform(y_aligned[va]), p, average='weighted'))
    print(f"  XGB LODO W-F1: {np.mean(scores):.4f} ({np.std(scores):.4f})")
    has_xgb = True
except ImportError:
    print("  XGBoost not installed, skip")
    has_xgb = False

# --- 3. Z-score per beacon ---
print("\n--- 3. Z-score per beacon ---")
v_float = v.astype(np.float64)

def lodo_zscore(clf_fn):
    scores = []
    for day in unique_days:
        va = days.values == day
        tr = ~va
        mu = np.nanmean(v_float[tr], axis=0)
        sd = np.nanstd(v_float[tr], axis=0)
        sd[sd == 0] = 1
        v_norm = (v_float - mu) / sd
        X_n = sliding_features(v_norm)
        clf = clf_fn()
        clf.fit(X_n[tr[HW:N-HW]], le.transform(y_aligned[tr[HW:N-HW]]))
        p = clf.predict(X_n[va[HW:N-HW]])
        scores.append(f1_score(le.transform(y_aligned[va[HW:N-HW]]), p, average='weighted'))
    return scores

scores = lodo_zscore(lambda: RandomForestClassifier(n_estimators=100, max_depth=20,
    class_weight='balanced_subsample', random_state=42, n_jobs=1))
print(f"  RF  LODO W-F1: {np.mean(scores):.4f} ({np.std(scores):.4f})")

if has_xgb:
    scores = []
    for day in unique_days:
        va = days.values == day
        tr = ~va
        mu = np.nanmean(v_float[tr], axis=0)
        sd = np.nanstd(v_float[tr], axis=0)
        sd[sd == 0] = 1
        v_norm = (v_float - mu) / sd
        X_n = sliding_features(v_norm)
        y_tr = le.transform(y_aligned[tr[HW:N-HW]])
        uniq = np.unique(y_tr)
        mapping = {o: n for n, o in enumerate(uniq)}
        inv_map = {n: o for o, n in mapping.items()}
        y_mapped = np.array([mapping[v] for v in y_tr])
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=6,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=42)
        clf.fit(X_n[tr[HW:N-HW]], y_mapped)
        p_mapped = clf.predict(X_n[va[HW:N-HW]])
        p = np.array([inv_map[v] for v in p_mapped])
        scores.append(f1_score(le.transform(y_aligned[va[HW:N-HW]]), p, average='weighted'))
    print(f"  XGB LODO W-F1: {np.mean(scores):.4f} ({np.std(scores):.4f})")

print(f"\nTime: {time.time()-t0:.1f}s")
