"""
04_final_training.py — FINAL MODEL: 22-Class RF for All Rooms

Purpose:
  Train and evaluate the final model — a 22-class Random Forest with
  class_weight='balanced_subsample' (no grouping).
  
  This model predicts ALL rooms on the 5th floor including
  6 rare rooms that were previously grouped into "other_room":
    503, 505, 510, 516, 517, 518 (each with only 23-44 training samples).
  
  Pipeline:
    1. Load ble_train_4d.csv → per-timestamp pivot (37,020 timestamps)
    2. Merge with cleaned_labels.csv → 23,905 labeled timestamps
    3. Sliding window (5 timestamps) → 100 features (mean/std/min/max × 25 beacons)
    4. RF: 100 trees, max_depth=20, class_weight='balanced_subsample'
    5. Predict BLE_Test_predict.csv → results/BLE_Test_predict_v5.csv

  Performance:
    CV Weighted F1 = 0.8244, Macro F1 = 0.8158
    Room number accuracy = 93.19% (1983/2128)
    
  Output: results/BLE_Test_predict_v5.csv

Usage: python code/05_final_prediction.py
"""

import pandas as pd
import numpy as np
import time, os, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, 'results', 'BLE_Test_predict_v5.csv')
WINDOW = 5

print("=" * 60)
print("05: Final - 22-class RF (all rooms, no grouping)")
print("=" * 60)
t0 = time.time()

# 1. Load
reader = pd.read_csv(os.path.join(BASE, 'ble_train_4d.csv'),
    usecols=lambda c: c in ['timestamp']+[f'RSSI_{i}' for i in range(1,26)],
    dtype={f'RSSI_{i}':'float32' for i in range(1,26)},
    parse_dates=['timestamp'], chunksize=200000, low_memory=False)
chunks = []
for chunk in reader:
    r = [f'RSSI_{i}' for i in range(1,26)]
    chunk[r] = chunk[r].replace(0, np.nan)
    chunks.append(chunk.groupby('timestamp')[r].mean())
ble = pd.concat(chunks)
ble = ble[~ble.index.duplicated(keep='first')].fillna(-100)

labels = pd.read_csv(os.path.join(BASE, 'cleaned_labels.csv'), parse_dates=['started_at','finished_at'])
ble = ble.reset_index()
ble['room'] = 'unknown'
for _, r in labels.iterrows():
    m = (ble['timestamp']>=r['started_at'])&(ble['timestamp']<=r['finished_at'])
    ble.loc[m,'room'] = r['room']
train = ble[ble['room']!='unknown'].copy()

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

# Encode all 22 classes
le = LabelEncoder()
y_e = le.fit_transform(y)
n_classes = len(le.classes_)
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

# Per-class from all folds
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
