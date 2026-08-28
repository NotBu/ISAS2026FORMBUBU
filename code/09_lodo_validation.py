"""
09_lodo_validation.py — Leave-One-Day-Out CV on V3 pipeline

Compares LODO vs StratifiedKFold for the paper.
"""

import pandas as pd, numpy as np, time, os, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.signal import savgol_filter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW = 5
t0 = time.time()

print("=" * 60)
print("09: Leave-One-Day-Out CV (V3 pipeline)")
print("=" * 60)

# Load preprocessed data (same as V3)
print("\nLoading BLE data...")
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

# SG smoothing
rssi_cols = [f'RSSI_{i}' for i in range(1, 26)]
for col in rssi_cols:
    ble[col] = savgol_filter(ble[col].values, window_length=5, polyorder=2)

# Merge labels
labels = pd.read_csv(os.path.join(BASE, 'cleaned_labels.csv'), parse_dates=['started_at','finished_at'])
ble = ble.reset_index()
bu = pd.Timedelta(seconds=4)
ble['room'] = 'unknown'
for _, r in labels.iterrows():
    m = (ble['timestamp']>=r['started_at']-bu)&(ble['timestamp']<=r['finished_at']+bu)
    ble.loc[m,'room'] = r['room']
train = ble[ble['room']!='unknown'].copy()
del ble, labels, reader, chunks

# Sliding window features (same as V3)
rssi = [f'RSSI_{i}' for i in range(1,26)]
v = train[rssi].values
rooms = train['room'].values
timestamps = train['timestamp'].values
X, y, ts = [], [], []
for i in range(len(v)-WINDOW+1):
    w = v[i:i+WINDOW]
    f = []
    for j in range(25):
        c = w[:,j]
        f += [np.mean(c),np.std(c),np.min(c),np.max(c)]
    X.append(f)
    y.append(rooms[i+WINDOW//2])
    ts.append(timestamps[i+WINDOW//2])
X, y, ts = np.array(X), np.array(y), np.array(ts)
del v, rooms, timestamps, train

le = LabelEncoder()
y_e = le.fit_transform(y)
n_classes = len(le.classes_)
print(f"Samples: {len(X)}, Features: {X.shape[1]}, Classes: {n_classes}")

# Day assignment
days = pd.Series(pd.to_datetime(ts)).dt.date
unique_days = sorted(days.unique())
print(f"Days in data: {unique_days}")

# LODO CV
print("\n[1/2] Leave-One-Day-Out CV:")
lodo_wf, lodo_mf = [], []
all_t_lodo, all_p_lodo = [], []
for day in unique_days:
    va_mask = days == day
    tr_mask = ~va_mask
    rf = RandomForestClassifier(n_estimators=100, max_depth=20,
        class_weight='balanced_subsample', random_state=42, n_jobs=1)
    rf.fit(X[tr_mask], y_e[tr_mask])
    p = rf.predict(X[va_mask])
    all_t_lodo.extend(y_e[va_mask]); all_p_lodo.extend(p)
    wf = f1_score(y_e[va_mask], p, average='weighted')
    mf = f1_score(y_e[va_mask], p, average='macro')
    lodo_wf.append(wf); lodo_mf.append(mf)
    cnt = va_mask.sum()
    print(f"  Holdout {day}: W-F1={wf:.4f} M-F1={mf:.4f} (n={cnt})")

print(f"\n  LODO CV: W-F1={np.mean(lodo_wf):.4f} (±{np.std(lodo_wf):.4f})")
print(f"            M-F1={np.mean(lodo_mf):.4f} (±{np.std(lodo_mf):.4f})")

print(f"\nPer-class LODO report:")
print(classification_report(all_t_lodo, all_p_lodo, target_names=le.classes_, digits=4))

# StratifiedKFold comparison (3-fold, matching V3)
print("-" * 60)
print("[2/2] StratifiedKFold (3-fold) for comparison:")
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
skf_wf, skf_mf = [], []
all_t_skf, all_p_skf = [], []
for tr, va in skf.split(X, y_e):
    rf = RandomForestClassifier(n_estimators=100, max_depth=20,
        class_weight='balanced_subsample', random_state=42, n_jobs=1)
    rf.fit(X[tr], y_e[tr])
    p = rf.predict(X[va])
    all_t_skf.extend(y_e[va]); all_p_skf.extend(p)
    skf_wf.append(f1_score(y_e[va], p, average='weighted'))
    skf_mf.append(f1_score(y_e[va], p, average='macro'))
    print(f"  Fold {len(skf_wf)}: W-F1={skf_wf[-1]:.4f} M-F1={skf_mf[-1]:.4f}")

print(f"\n  SKF CV: W-F1={np.mean(skf_wf):.4f} (±{np.std(skf_wf):.4f})")
print(f"          M-F1={np.mean(skf_mf):.4f} (±{np.std(skf_mf):.4f})")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"                    W-F1        M-F1")
print(f"  LODO              {np.mean(lodo_wf):.4f} (±{np.std(lodo_wf):.4f})    {np.mean(lodo_mf):.4f} (±{np.std(lodo_mf):.4f})")
print(f"  StratifiedKFold   {np.mean(skf_wf):.4f} (±{np.std(skf_wf):.4f})    {np.mean(skf_mf):.4f} (±{np.std(skf_mf):.4f})")
print(f"  Δ (LODO - SKF)    {np.mean(lodo_wf)-np.mean(skf_wf):+.4f}         {np.mean(lodo_mf)-np.mean(skf_mf):+.4f}")
print(f"\nTotal time: {time.time()-t0:.1f}s")
