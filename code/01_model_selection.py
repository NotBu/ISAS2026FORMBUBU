"""
01_model_selection.py — Strategy Selection for Location Prediction

Purpose:
  Compare 3 strategies for handling imbalanced room data:
    A:  RF + class_weight='balanced_subsample' (treats rare rooms fairly)
    D:  RF without class_weight (focuses on common rooms)
    A+D: RF + class_weight + confidence threshold filtering
  
  Outputs comparison table and retrains best model → results/BLE_Test_predict_v4.csv

Result (from experiment):
  Best: Strategy D (no class_weight), threshold=0.0 → Weighted F1 = 0.8315
  17 classes (16 common rooms + other_room for rooms <50 samples)

Usage: python code/02_strategy_comparison.py
"""

import pandas as pd
import numpy as np
import time, os, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLE_DATA = os.path.join(BASE, 'ble_train_4d.csv')
LABEL_FILE = os.path.join(BASE, 'cleaned_labels.csv')
TEST_FILE = os.path.join(BASE, 'BLE_Test_predict.csv')
OUTPUT_FILE = os.path.join(BASE, 'results', 'BLE_Test_predict_v4.csv')

WINDOW = 5
MIN_SAMPLES = 50
THRESHOLDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]

print("=" * 70)
print("02: Compare Strategies A, D, A+D")
print("=" * 70)
t0 = time.time()

# 1. Load data
reader = pd.read_csv(BLE_DATA,
    usecols=lambda c: c in ['timestamp'] + [f'RSSI_{i}' for i in range(1, 26)],
    dtype={f'RSSI_{i}': 'float32' for i in range(1, 26)},
    parse_dates=['timestamp'], chunksize=200000, low_memory=False)

chunks = []
for chunk in reader:
    rssi = [f'RSSI_{i}' for i in range(1, 26)]
    chunk[rssi] = chunk[rssi].replace(0, np.nan)
    chunks.append(chunk.groupby('timestamp')[rssi].mean())

ble_ts = pd.concat(chunks)
ble_ts = ble_ts[~ble_ts.index.duplicated(keep='first')].fillna(-100)

labels = pd.read_csv(LABEL_FILE, parse_dates=['started_at', 'finished_at'])
ble_ts = ble_ts.reset_index()
ble_ts['room'] = 'unknown'
for _, r in labels.iterrows():
    m = (ble_ts['timestamp'] >= r['started_at']) & (ble_ts['timestamp'] <= r['finished_at'])
    ble_ts.loc[m, 'room'] = r['room']
train = ble_ts[ble_ts['room'] != 'unknown'].copy()

# Sliding window
rssi = [f'RSSI_{i}' for i in range(1, 26)]
v = train[rssi].values
rooms = train['room'].values
X, y = [], []
for i in range(len(v) - WINDOW + 1):
    w = v[i:i+WINDOW]
    f = []
    for j in range(25):
        c = w[:, j]
        f += [np.mean(c), np.std(c), np.min(c), np.max(c)]
    X.append(f); y.append(rooms[i + WINDOW // 2])
X, y = np.array(X), np.array(y)

# Group rare
uniq, cnt = np.unique(y, return_counts=True)
rare = set(uniq[cnt < MIN_SAMPLES])
y_g = np.array([r if r not in rare else 'other_room' for r in y])
le = LabelEncoder()
y_e = le.fit_transform(y_g)

def cv_score(X, y, use_cw, th):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    wf, mf, ac = [], [], []
    for tr, va in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=100, max_depth=20,
            class_weight='balanced_subsample' if use_cw else None,
            random_state=42, n_jobs=1)
        rf.fit(X[tr], y[tr])
        p = rf.predict(X[va])
        if th > 0:
            prob = rf.predict_proba(X[va])
            idx = list(le.classes_).index('other_room') if 'other_room' in le.classes_ else None
            if idx is not None:
                p[np.max(prob, axis=1) < th] = idx
        wf.append(f1_score(y[va], p, average='weighted'))
        mf.append(f1_score(y[va], p, average='macro'))
        ac.append(accuracy_score(y[va], p))
    return np.mean(wf), np.mean(mf), np.mean(ac)

results = []
for name, cw in [('A - class_weight', True), ('D - no_weight', False)]:
    print(f"\n  {name}:")
    for th in THRESHOLDS:
        w, m, a = cv_score(X, y_e, cw, th)
        results.append({'Strategy': name, 'Threshold': th, 'W-F1': round(w,4), 'M-F1': round(m,4), 'Acc': round(a,4)})
        if th == 0.0 or cw:
            print(f"    Thresh={th}: W-F1={w:.4f} M-F1={m:.4f} Acc={a:.4f}")
    if not cw:
        for th in THRESHOLDS[1:]:
            w, m, a = cv_score(X, y_e, cw, th)
            print(f"    Thresh={th}: W-F1={w:.4f} M-F1={m:.4f} Acc={a:.4f}")

df = pd.DataFrame(results)
print("\n" + "=" * 70)
print(df.to_string(index=False))
best = df.loc[df['W-F1'].idxmax()]
print(f"\nBest: {best['Strategy']} @ thresh={best['Threshold']} -> W-F1={best['W-F1']}")

# Retrain best
use_cw = 'class_weight' in best['Strategy']
th = best['Threshold']
rf = RandomForestClassifier(n_estimators=100, max_depth=20,
    class_weight='balanced_subsample' if use_cw else None,
    random_state=42, n_jobs=1)
rf.fit(X, y_e)

# Predict test
test = pd.read_csv(TEST_FILE, parse_dates=['timestamp'])
tp = test.groupby(['timestamp', 'mac address'])['RSSI'].mean().reset_index()
tp = tp.pivot(index='timestamp', columns='mac address', values='RSSI')
tp.columns = [f'RSSI_{int(c)}' for c in tp.columns]
for i in range(1, 26):
    c = f'RSSI_{i}'
    if c not in tp.columns: tp[c] = -100
tp = tp[[f'RSSI_{i}' for i in range(1, 26)]].fillna(-100)

tv, tt = tp.values, tp.index.values
Xt, tst = [], []
for i in range(len(tv) - WINDOW + 1):
    w = tv[i:i+WINDOW]
    f = []
    for j in range(25):
        c = w[:, j]
        f += [np.mean(c), np.std(c), np.min(c), np.max(c)]
    Xt.append(f); tst.append(tt[i + WINDOW // 2])
Xt = np.array(Xt)

p = rf.predict(Xt)
prob = rf.predict_proba(Xt)
conf = np.max(prob, axis=1)
pred = le.inverse_transform(p)
if th > 0:
    idx = list(le.classes_).index('other_room')
    p[conf < th] = idx
    pred = le.inverse_transform(p)

result = pd.DataFrame({
    'timestamp': pd.to_datetime(tst),
    'predicted_room': pred,
    'confidence': conf
}).sort_values('timestamp').reset_index(drop=True)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
result.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE} ({len(result)} predictions)")
print(f"Total time: {time.time()-t0:.1f}s")
