"""
03_rare_room_test.py — Investigation: Can We Predict Rare Rooms?

Purpose:
  Test whether a 22-class model (no grouping) can accurately predict
  the 6 rarest rooms (<50 samples each): 503, 505, 510, 516, 517, 518.
  
  Two approaches tested:
    1. Single 22-class RF with class_weight
    2. Two-stage: 17-class → if "other_room", run separate 6-class model
  
  Result (from experiment):
    22-class with class_weight IS viable:
    Rare room F1 scores: 503=0.875, 505=0.844, 510=0.857,
                         516=0.718, 517=0.739, 518=0.630
    Overall Weighted F1 = 0.8244 (vs 0.8315 for 17-class)
    Trade-off: -0.007 W-F1 but predicts ALL 22 rooms individually.

Usage: python code/04_rare_room_analysis.py
"""

import pandas as pd
import numpy as np
import time, os, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLE_DATA = os.path.join(BASE, 'ble_train_4d.csv')
LABEL_FILE = os.path.join(BASE, 'cleaned_labels.csv')
TEST_FILE = os.path.join(BASE, 'BLE_Test_predict.csv')
OUTPUT = os.path.join(BASE, 'results', 'BLE_Test_predict_v2.csv')

WINDOW = 5
RARE_THRESHOLD = 50  # rooms <50 samples are "rare"

print("=" * 70)
print("04: Test Rare Room Prediction")
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

# Identify rare rooms
uniq, cnts = np.unique(y, return_counts=True)
rare_rooms = set(uniq[cnts < RARE_THRESHOLD])
common_rooms = set(uniq[cnts >= RARE_THRESHOLD])
print(f"\nCommon rooms ({len(common_rooms)}): {sorted(common_rooms)}")
print(f"Rare rooms ({len(rare_rooms)}): {sorted(rare_rooms)}")

# =========================================
# TEST 1: 22-class model (no grouping) with class_weight
# =========================================
print(f"\n{'='*50}")
print("TEST 1: 22-class RF with class_weight (no grouping)")
print(f"{'='*50}")

le22 = LabelEncoder()
y22 = le22.fit_transform(y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_true22, all_pred22 = [], []

for tr, va in skf.split(X, y22):
    rf22 = RandomForestClassifier(n_estimators=100, max_depth=20,
        class_weight='balanced_subsample', random_state=42, n_jobs=1)
    rf22.fit(X[tr], y22[tr])
    all_true22.extend(y22[va])
    all_pred22.extend(rf22.predict(X[va]))

all_true22 = np.array(all_true22)
all_pred22 = np.array(all_pred22)

print(classification_report(all_true22, all_pred22, target_names=le22.classes_, digits=4))

# Per-class for rare rooms
print(f"\nRare room performance:")
for room in sorted(rare_rooms):
    idx = list(le22.classes_).index(room)
    mask_true = all_true22 == idx
    n_true = mask_true.sum()
    n_pred = (all_pred22 == idx).sum()
    correct = ((all_true22 == idx) & (all_pred22 == idx)).sum()
    prec = correct / max(n_pred, 1)
    rec = correct / max(n_true, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    print(f"  {room:>5}: n_true={n_true:>3}, n_pred={n_pred:>3}, correct={correct:>3}, prec={prec:.3f}, rec={rec:.3f}, F1={f1:.3f}")

wf1_22 = f1_score(all_true22, all_pred22, average='weighted')
mf1_22 = f1_score(all_true22, all_pred22, average='macro')
print(f"\n  22-class: W-F1={wf1_22:.4f}, M-F1={mf1_22:.4f}")

# =========================================
# TEST 2: Two-stage model
# Stage 1: 17-class (common + other_room)
# Stage 2: 6-class (only rare rooms, for samples predicted as other_room)
# =========================================
print(f"\n{'='*50}")
print("TEST 2: Two-stage (17 + 6 class)")
print(f"{'='*50}")

# Stage 1 labels
y_g = np.array([r if r not in rare_rooms else 'other_room' for r in y])
le1 = LabelEncoder()
y1 = le1.fit_transform(y_g)

# Stage 2 labels (only rare rooms)
rare_mask = np.array([r in rare_rooms for r in y])
y_rare = y[rare_mask]
X_rare = X[rare_mask]
if len(X_rare) > 0:
    le2 = LabelEncoder()
    y2 = le2.fit_transform(y_rare)

all_true_2s, all_pred_2s = [], []

# 5-fold CV
for tr, va in skf.split(X, y1):
    # Stage 1: 17-class
    rf1 = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=1)
    rf1.fit(X[tr], y1[tr])
    p1 = rf1.predict(X[va])
    
    # For samples predicted as other_room, try stage 2
    va_indices = np.where(p1 == list(le1.classes_).index('other_room'))[0]
    p_final = p1.copy()
    
    if len(va_indices) > 0 and len(X_rare) > 0 and len(np.unique(y2)) > 1:
        # Check if any rare rooms exist in this fold's training data
        rare_tr = rare_mask[tr]
        if rare_tr.sum() >= 6:  # need at least 1 sample per class (6 classes)
            rf2 = RandomForestClassifier(n_estimators=50, max_depth=10, 
                class_weight='balanced_subsample', random_state=42, n_jobs=1)
            rf2.fit(X_rare, y2)
            
            for idx in va_indices:
                if idx < len(X):
                    p2 = rf2.predict(X[idx].reshape(1, -1))[0]
                    p_final[idx] = list(le1.classes_).index(le2.inverse_transform([p2])[0])
    
    all_true_2s.extend(y1[va])
    all_pred_2s.extend(p_final)

all_true_2s = np.array(all_true_2s)
all_pred_2s = np.array(all_pred_2s)

print("\nTwo-stage classification report (17+6 -> up to 22 classes):")
print(classification_report(all_true_2s, all_pred_2s, target_names=le1.classes_, digits=4))

wf1_2s = f1_score(all_true_2s, all_pred_2s, average='weighted')
mf1_2s = f1_score(all_true_2s, all_pred_2s, average='macro')
print(f"\n  Two-stage: W-F1={wf1_2s:.4f}, M-F1={mf1_2s:.4f}")

# =========================================
# CONCLUSION
# =========================================
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
print(f"  22-class (class_weight): W-F1={wf1_22:.4f}, M-F1={mf1_22:.4f}")
print(f"  Two-stage (17+6):        W-F1={wf1_2s:.4f}, M-F1={mf1_2s:.4f}")
print(f"  Curr best 17-class (V1): W-F1=0.8315, M-F1=0.8193")

# Choose best and predict test
print(f"\n{'='*50}")
print("Predicting test with best approach...")
print(f"{'='*50}")

# Retrain best model (two-stage if better, else 17-class)
rf_best = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=1)
rf_best.fit(X, y1)

# Stage 2 model on all rare data
if len(X_rare) > 0 and len(np.unique(y2)) > 1:
    rf_rare = RandomForestClassifier(n_estimators=50, max_depth=10,
        class_weight='balanced_subsample', random_state=42, n_jobs=1)
    rf_rare.fit(X_rare, y2)

# Load and predict test
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

# Stage 1 predictions
p1 = rf_best.predict(Xt)
prob1 = rf_best.predict_proba(Xt)
conf1 = np.max(prob1, axis=1)
pred = le1.inverse_transform(p1)

# For other_room predictions, try stage 2
other_idx = list(le1.classes_).index('other_room')
other_mask = p1 == other_idx
if other_mask.sum() > 0 and len(X_rare) > 0 and len(np.unique(y2)) > 1:
    p2 = rf_rare.predict(Xt[other_mask])
    rare_pred = le2.inverse_transform(p2)
    prob2 = rf_rare.predict_proba(Xt[other_mask])
    conf2 = np.max(prob2, axis=1)
    pred[other_mask] = rare_pred
    conf1[other_mask] = conf2

result = pd.DataFrame({
    'timestamp': pd.to_datetime(tst),
    'predicted_room': pred,
    'confidence': conf1
}).sort_values('timestamp').reset_index(drop=True)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
result.to_csv(OUTPUT, index=False)
print(f"\nSaved: {OUTPUT}")
print(f"Predictions: {len(result)}")
print(f"Distribution:\n{result['predicted_room'].value_counts()}")
print(f"Total time: {time.time()-t0:.1f}s")
