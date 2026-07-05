"""
08_relabeling.py — Relabeling for Indoor Localization (Garcia & Inoue, 2024)

Purpose:
  Implement signal-pattern-based relabeling to augment minority (rare) rooms:
    1. Compute "signal profile" of each minority room (mean RSSI vector)
    2. For each majority room sample, compute cosine similarity vs each minority profile
    3. If similarity > threshold, duplicate that sample with minority room label
    4. Train RF (same as V3) and compare F1

  Reference: sensors-24-00319 — "Relabeling for Indoor Localization Using Stationary
  Beacons in Nursing Care Facilities"

Usage: python code/08_relabeling.py
"""

import pandas as pd
import numpy as np
import time, os, gc, warnings
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
WINDOW = 5
RARE_ROOMS = ['503', '505', '510', '516', '517', '518']
SIMILARITY_THRESHOLD = 0.90  # cosine similarity threshold for relabeling

print("=" * 60)
print("08: Relabeling (Garcia & Inoue 2024)")
print("=" * 60)
t0 = time.time()

# 1. Load data
print("Loading BLE data...")
reader = pd.read_csv(os.path.join(BASE, 'ble_train_4d.csv'),
    usecols=lambda c: c in ['timestamp']+[f'RSSI_{i}' for i in range(1,26)],
    dtype={'timestamp': str, **{f'RSSI_{i}':'float32' for i in range(1,26)}},
    chunksize=25000, low_memory=False)
chunks = []
for chunk in reader:
    r = [f'RSSI_{i}' for i in range(1,26)]
    chunk[r] = chunk[r].replace(0, np.nan)
    chunks.append(chunk.groupby('timestamp')[r].mean())
ble_full = pd.concat(chunks)
ble_full.index = pd.to_datetime(ble_full.index)
ble_full = ble_full[~ble_full.index.duplicated(keep='first')].fillna(-100)
ble_full = ble_full.sort_index()
print(f"  BLE timestamps: {len(ble_full)}")

# 2. Merge labels (with 4s buffer)
labels = pd.read_csv(os.path.join(BASE, 'cleaned_labels.csv'), parse_dates=['started_at','finished_at'])
bu = pd.Timedelta(seconds=4)
ble_full = ble_full.reset_index()
ble_full['room'] = 'unknown'
for _, r in labels.iterrows():
    m = (ble_full['timestamp']>=r['started_at']-bu)&(ble_full['timestamp']<=r['finished_at']+bu)
    ble_full.loc[m,'room'] = r['room']

# 3. Apply SG smoothing
rssi_cols = [f'RSSI_{i}' for i in range(1, 26)]
ble_sorted = ble_full.sort_values('timestamp')
for col in rssi_cols:
    ble_sorted[col] = savgol_filter(ble_sorted[col].values, window_length=5, polyorder=2)
ble_sorted = ble_sorted.reset_index(drop=True)

# 4. Count room samples (before relabeling)
room_counts = ble_sorted[ble_sorted['room']!='unknown']['room'].value_counts()
print(f"\nRoom distribution (before relabeling):")
for r, c in room_counts.items():
    print(f"  {str(r):>15}: {c:>5}")

rare_rooms = set(r for r in RARE_ROOMS if r in room_counts.index)
print(f"\nRare rooms: {sorted(rare_rooms)}")
print(f"Majority rooms: {sorted(set(room_counts.index) - rare_rooms)}")

# 5. Compute signal profile for each minority room
train = ble_sorted[ble_sorted['room']!='unknown'].copy()
profiles = {}
for room in rare_rooms:
    mask = train['room'] == room
    if mask.sum() < 3:
        continue
    vals = train.loc[mask, rssi_cols].values
    profiles[room] = {
        'mean': np.mean(vals, axis=0),
        'std': np.std(vals, axis=0),
        'n': mask.sum()
    }
    print(f"  Profile {room}: mean={np.mean(vals):.1f}, n={mask.sum()}")

# 6. Relabeling: find majority samples that match minority profiles
offset = 100.0  # shift RSSI to positive for cosine similarity

# Precompute normalized minority profiles
minority_profiles = {}
for room, prof in profiles.items():
    vec = prof['mean'] + offset
    minority_profiles[room] = vec / np.linalg.norm(vec)

# For each majority room, find matchable samples
majority_mask = ~train['room'].isin(rare_rooms)
majority = train[majority_mask].copy()
relabeled = []

print(f"\nScanning {len(majority)} majority samples for relabeling...")
for idx, (_, row) in enumerate(majority.iterrows()):
    rssi_vec = row[rssi_cols].values + offset
    nrm = np.linalg.norm(rssi_vec)
    if nrm == 0:
        continue
    rssi_norm = rssi_vec / nrm

    best_match = None
    best_sim = 0
    for room, mn in minority_profiles.items():
        sim = float(np.dot(rssi_norm, mn))
        if sim > best_sim:
            best_sim = sim
            best_match = room

    if best_sim >= SIMILARITY_THRESHOLD and best_match is not None:
        new_row = row.to_dict()
        new_row['room'] = best_match
        new_row['relabeled'] = True
        relabeled.append(new_row)

    if (idx + 1) % 5000 == 0:
        print(f"  Scanned {idx+1}/{len(majority)}...")

print(f"  Found {len(relabeled)} samples to relabel")

# 7. Build augmented dataset
if len(relabeled) > 0:
    df_relabeled = pd.DataFrame(relabeled)
    train_aug = pd.concat([train, df_relabeled], ignore_index=True)
else:
    train_aug = train.copy()

train_aug = train_aug.sort_values('timestamp').reset_index(drop=True)

# 8. Sliding window
v = train_aug[rssi_cols].values
rooms = train_aug['room'].values
X, y = [], []
for i in range(len(v)-WINDOW+1):
    w = v[i:i+WINDOW]
    f = []
    for j in range(25):
        c = w[:,j]
        f += [np.mean(c),np.std(c),np.min(c),np.max(c)]
    X.append(f); y.append(rooms[i+WINDOW//2])
X, y = np.array(X), np.array(y)

del v, rooms, train, train_aug, majority, ble_sorted, ble_full
gc.collect()

# 9. Encode + CV
le = LabelEncoder()
y_e = le.fit_transform(y)
n_classes = len(le.classes_)
print(f"\nTraining samples after relabel: {len(X)}, Features: {X.shape[1]}, Classes: {n_classes}")

print("\nRoom distribution (after relabeling + sliding window):")
uniq, cnts = np.unique(y, return_counts=True)
for r, c in sorted(zip(uniq, cnts), key=lambda x: -x[1]):
    print(f"  {str(r):>15}: {c:>5}")

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

# Rare room F1
rare_names = ['503','505','510','516','517','518']
rare_ids = [list(le.classes_).index(r) for r in le.classes_ if r in rare_names]
rare_mask = np.isin(all_t, rare_ids)
if rare_mask.sum() > 0:
    rf1 = f1_score(np.array(all_t)[rare_mask], np.array(all_p)[rare_mask], average='macro')
    print(f"\n  Rare room F1: {rf1:.4f}")

# 10. Predict test
print("\n[3/3] Retrain full + predict test...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20,
    class_weight='balanced_subsample', random_state=42, n_jobs=1)
rf.fit(X, y_e)

test = pd.read_csv(os.path.join(BASE, 'BLE_Test_predict.csv'), parse_dates=['timestamp'])
tp = test.groupby(['timestamp','mac address'])['RSSI'].mean().reset_index()
tp = tp.pivot(index='timestamp', columns='mac address', values='RSSI')
tp.columns = [f'RSSI_{int(c)}' for c in tp.columns]
for i in range(1,26):
    c = f'RSSI_{i}'
    if c not in tp.columns: tp[c] = -100
tp = tp[[f'RSSI_{i}' for i in range(1,26)]].fillna(-100)

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

out_path = os.path.join(BASE, 'results', 'BLE_Test_predict_v4_relabel.csv')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
result.to_csv(out_path, index=False)
print(f"Saved: {out_path} ({len(result)} predictions)")
print(f"\nDistribution:")
for room, cnt in result['predicted_room'].value_counts().items():
    print(f"  {room:>15}: {cnt:>4}")
print(f"\nTotal time: {time.time()-t0:.1f}s")
