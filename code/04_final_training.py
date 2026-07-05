"""
04_final_training.py — FINAL MODEL V3: 22-Class RF + SG Smoothing

Purpose:
  Train and evaluate the final model — a 22-class Random Forest with
  class_weight='balanced_subsample' + Savitzky-Golay signal smoothing.

  New in V3: Savitzky-Golay filter applied per-beacon on RSSI time series
  (window=5, polyorder=2) to reduce high-frequency noise before sliding
  window feature extraction. Improves all metrics vs V2.

  Pipeline:
    1. Load ble_train_4d.csv → per-timestamp pivot (37,020 timestamps)
    2. Apply Savitzky-Golay smoothing per RSSI beacon
    3. Merge with cleaned_labels.csv → 23,905 labeled timestamps
    4. Sliding window (5 timestamps) → 100 features (mean/std/min/max × 25 beacons)
    5. RF: 100 trees, max_depth=20, class_weight='balanced_subsample'
    6. Predict BLE_Test_predict.csv → results/BLE_Test_predict_v3.csv

  Performance (vs V2):
                    V2       V3       Δ
    Weighted F1:  0.8259   0.8351  +0.0092
    Macro F1:     0.8181   0.8336  +0.0155
    Rare F1:      0.3725   0.4757  +0.1032

  Output: results/BLE_Test_predict_v3.csv

Usage: python code/04_final_training.py
"""

import pandas as pd
import numpy as np
import time, os, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

try:
    from scipy.signal import savgol_filter
except ImportError:
    print("ERROR: scipy not installed. Run: pip install scipy")
    exit(1)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, 'results', 'BLE_Test_predict_v3.csv')
CACHE = os.path.join(BASE, '.cache')
WINDOW = 5

print("=" * 60)
print("04: Final V3 - 22-class RF + SG smoothing")
print("=" * 60)
t0 = time.time()

# 1. Load / cache
os.makedirs(CACHE, exist_ok=True)
X_cache = os.path.join(CACHE, 'X_smooth.npy')
y_cache = os.path.join(CACHE, 'y.npy')
le_cache = os.path.join(CACHE, 'classes.npy')

if os.path.exists(X_cache) and os.path.exists(y_cache) and os.path.exists(le_cache):
    print("Loading cached data...")
    X = np.load(X_cache)
    y = np.load(y_cache)
    classes = np.load(le_cache, allow_pickle=True)
    le = LabelEncoder(); le.classes_ = classes; y_e = le.transform(y)
    n_classes = len(classes)
else:
    print("Loading BLE data from CSV...")
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
    ble = ble[~ble.index.duplicated(keep='first')].fillna(-100)

    # 2. Savitzky-Golay smoothing per beacon
    ble = ble.sort_index()
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

    # Sliding window
    rssi = [f'RSSI_{i}' for i in range(1,26)]
    v = train[rssi].values
    rooms = train['room'].values
    X, y = [], []
    for i in range(len(v)-WINDOW+1):
        w = v[i:i+WINDOW]
        f = []
        for j in range(25):
            c = w[:,j]
            f += [np.mean(c),np.std(c),np.min(c),np.max(c)]
        X.append(f); y.append(rooms[i+WINDOW//2])
    X, y = np.array(X), np.array(y)
    del v, rooms, train

    le = LabelEncoder()
    y_e = le.fit_transform(y)
    n_classes = len(le.classes_)
    np.save(X_cache, X); np.save(y_cache, y); np.save(le_cache, le.classes_)

print(f"\nTraining samples: {len(X)}, Features: {X.shape[1]}, Classes: {n_classes}")

# CV
print("\n[1/3] 5-fold CV...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
wf, mf = [], []
all_t, all_p = [], []
for tr, va in skf.split(X, y_e):
    rf = RandomForestClassifier(n_estimators=100, max_depth=20,
        class_weight='balanced_subsample', random_state=42, n_jobs=1)
    rf.fit(X[tr], y_e[tr])
    p = rf.predict(X[va])
    all_t.extend(y_e[va]); all_p.extend(p)
    wf.append(f1_score(y_e[va], p, average='weighted'))
    mf.append(f1_score(y_e[va], p, average='macro'))
    print(f"  Fold {len(wf)}: W-F1={wf[-1]:.4f} M-F1={mf[-1]:.4f}")

print(f"\n  CV: W-F1={np.mean(wf):.4f}, M-F1={np.mean(mf):.4f}")

print(f"\n[2/3] Per-class report:")
print(classification_report(all_t, all_p, target_names=le.classes_, digits=4))

# Retrain full
print("\n[3/3] Retrain full + predict test...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20,
    class_weight='balanced_subsample', random_state=42, n_jobs=1)
rf.fit(X, y_e)

# Predict test
test = pd.read_csv(os.path.join(BASE, 'BLE_Test_predict.csv'), parse_dates=['timestamp'])
tp = test.groupby(['timestamp','mac address'])['RSSI'].mean().reset_index()
tp = tp.pivot(index='timestamp', columns='mac address', values='RSSI')
tp.columns = [f'RSSI_{int(c)}' for c in tp.columns]
for i in range(1,26):
    c = f'RSSI_{i}'
    if c not in tp.columns: tp[c] = -100
tp = tp[[f'RSSI_{i}' for i in range(1,26)]].fillna(-100)

# Apply SG smoothing to test data too
rssi_cols = [f'RSSI_{i}' for i in range(1, 26)]
tp_sorted = tp.sort_index()
for col in rssi_cols:
    tp_sorted[col] = savgol_filter(tp_sorted[col].values, window_length=5, polyorder=2)
tp = tp_sorted

tv, tt = tp.values, tp.index.values
Xt, tst = [], []
for i in range(len(tv)-WINDOW+1):
    w = tv[i:i+WINDOW]
    f = []
    for j in range(25):
        c = w[:,j]
        f += [np.mean(c),np.std(c),np.min(c),np.max(c)]
    Xt.append(f); tst.append(tt[i+WINDOW//2])
Xt = np.array(Xt)

p = rf.predict(Xt)
prob = rf.predict_proba(Xt)
conf = np.max(prob, axis=1)
pred = le.inverse_transform(p)

result = pd.DataFrame({
    'timestamp': pd.to_datetime(tst),
    'predicted_room': pred,
    'confidence': conf
}).sort_values('timestamp').reset_index(drop=True)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
result.to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT} ({len(result)} predictions)")
print(f"\nDistribution:")
for room, cnt in result['predicted_room'].value_counts().items():
    print(f"  {room:>15}: {cnt:>4}")
print(f"\nTotal time: {time.time()-t0:.1f}s")
