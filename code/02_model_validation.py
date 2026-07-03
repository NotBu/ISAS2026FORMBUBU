"""
02_model_validation.py — Model Quality Diagnostic

Purpose:
  Analyze model performance in detail:
    1. Per-class precision, recall, F1-score from 5-fold CV
    2. Confusion matrix for room number classes
    3. Temporal consistency check on test predictions
       (room change frequency, confidence distribution, transitions)
  
  Use this to identify which rooms the model struggles with.

Usage: python code/03_diagnostic_report.py
"""

import pandas as pd
import numpy as np
import time, os, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLE_DATA = os.path.join(BASE, 'ble_train_4d.csv')
LABEL_FILE = os.path.join(BASE, 'cleaned_labels.csv')
TEST_RESULT = os.path.join(BASE, 'results', 'BLE_Test_predict_v4.csv')

WINDOW = 5
MIN_SAMPLES = 50

print("=" * 70)
print("03: Diagnosis - Per-class Metrics + Temporal Check")
print("=" * 70)
t0 = time.time()

# 1. Load
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

# Group
uniq, cnt = np.unique(y, return_counts=True)
rare = set(uniq[cnt < MIN_SAMPLES])
y_g = np.array([r if r not in rare else 'other_room' for r in y])
le = LabelEncoder()
y_e = le.fit_transform(y_g)

# 2. CV with per-class report
print(f"\n[1/3] CV 5-fold per-class report...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_true, all_pred = [], []
for tr, va in skf.split(X, y_e):
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=1)
    rf.fit(X[tr], y_e[tr])
    all_true.extend(y_e[va])
    all_pred.extend(rf.predict(X[va]))

print(classification_report(all_true, all_pred, target_names=le.classes_, digits=4))

# 3. Confusion matrix for room numbers
print(f"\n[2/3] Room number confusion matrix...")
cm = confusion_matrix(all_true, all_pred)
room_cls = [c for c in le.classes_ if c not in ['cafeteria','kitchen','nurse station','hallway','cleaning','other_room']]
room_idx = [list(le.classes_).index(c) for c in room_cls]

total, correct = 0, 0
for i, rc in enumerate(room_cls):
    ri = list(le.classes_).index(rc)
    row_total = cm[ri][room_idx].sum() + cm[ri][list(le.classes_).index('other_room')]
    corr = cm[ri, ri]
    total += row_total
    correct += corr
    print(f"  {rc:>5}: {corr}/{row_total} = {corr/max(row_total,1):.0%} (other={cm[ri,list(le.classes_ ).index('other_room')]})")

print(f"\n  Room number accuracy: {correct}/{total} = {correct/max(total,1):.4f}")

# 4. Temporal check on test predictions
print(f"\n[3/3] Test prediction temporal check...")
try:
    r = pd.read_csv(TEST_RESULT, parse_dates=['timestamp']).sort_values('timestamp')
    changes = (r['predicted_room'] != r['predicted_room'].shift(1)).sum()
    print(f"  Predictions: {len(r)}, changes: {changes} ({changes/len(r):.1%})")
    print(f"  Avg between changes: every {len(r)/max(changes,1):.1f} predictions")
    # Confidence by type
    room_nums = [c for c in r['predicted_room'].unique() if c not in ['cafeteria','kitchen','nurse station','hallway','cleaning','other_room']]
    r['type'] = r['predicted_room'].apply(lambda x: 'room_number' if x in room_nums else x)
    for t in sorted(r['type'].unique()):
        sub = r[r['type'] == t]
        print(f"  {t:>20}: n={len(sub):4d}, mean_conf={sub['confidence'].mean():.3f}")
    print(f"  First 15 transitions:")
    prev = None
    cnt = 0
    for _, row in r.iterrows():
        if row['predicted_room'] != prev and prev is not None and cnt < 15:
            print(f"    {row['timestamp']} : {prev} -> {row['predicted_room']} (conf={row['confidence']:.3f})")
            cnt += 1
        prev = row['predicted_room']
except FileNotFoundError:
    print(f"  No test result found at {TEST_RESULT}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
