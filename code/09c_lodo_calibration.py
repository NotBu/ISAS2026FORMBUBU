"""
09c_lodo_calibration.py — LODO + cross-day calibration
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

print("=" * 70)
print("09c: LODO + Cross-Day Calibration")
print("=" * 70)

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

# add day column
train['day'] = train['timestamp'].dt.date
v = train[rssi_cols].values.astype(np.float64)
rooms = train['room'].values
days = train['day'].values
le = LabelEncoder()
le.fit(rooms)

# per-day per-beacon mean
day_beacon_mean = train.groupby('day')[rssi_cols].mean()
unique_days = sorted(day_beacon_mean.index)
print(f"Days: {unique_days}")
print(f"Samples: {len(train)}, Beacons: 25")

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

# Baseline LODO (same as before)
X_base = sliding_features(v)
y_aligned = rooms[HW:len(rooms)-HW] if len(rooms) > 2*HW else rooms
d_aligned = days[HW:len(days)-HW] if len(days) > 2*HW else days

print("\n--- Baseline LODO ---")
base_scores = []
for day in unique_days:
    va = d_aligned == day
    tr = ~va
    rf = RandomForestClassifier(n_estimators=100, max_depth=20,
        class_weight='balanced_subsample', random_state=42, n_jobs=1)
    rf.fit(X_base[tr], le.transform(y_aligned[tr]))
    p = rf.predict(X_base[va])
    s = f1_score(le.transform(y_aligned[va]), p, average='weighted')
    base_scores.append(s)
    print(f"  Holdout {day}: W-F1={s:.4f}")
print(f"  LODO W-F1: {np.mean(base_scores):.4f} ({np.std(base_scores):.4f})")

# --- Cross-Day Calibration ---
# Idea: align all train days to a "reference" day's RSSI distribution
# per beacon: shift = mean(beacon, reference_day) - mean(beacon, source_day)
# Then each day's beacon values get shifted to match the reference day

print("\n--- Cross-Day Calibration LODO ---")
cal_scores = []
for day in unique_days:  # held-out day
    ref_days = [d for d in unique_days if d != day]  # training days
    # pick first training day as reference for alignment
    ref_day = ref_days[0]
    ref_mean = day_beacon_mean.loc[ref_day].values.astype(np.float64)

    # calibrate: shift all training days to match reference day
    v_cal = v.copy()
    for d in ref_days:
        shift = ref_mean - day_beacon_mean.loc[d].values.astype(np.float64)
        mask = days == d
        v_cal[mask] += shift[np.newaxis, :]
    # held-out day stays raw (simulating real scenario where we can't calibrate it)

    X_cal = sliding_features(v_cal)
    # align
    va_mask = days[HW:len(days)-HW] == day if len(days) > 2*HW else days == day
    tr_mask = ~va_mask
    rf = RandomForestClassifier(n_estimators=100, max_depth=20,
        class_weight='balanced_subsample', random_state=42, n_jobs=1)
    rf.fit(X_cal[tr_mask], le.transform(y_aligned[tr_mask]))
    p = rf.predict(X_cal[va_mask])
    s = f1_score(le.transform(y_aligned[va_mask]), p, average='weighted')
    cal_scores.append(s)
    print(f"  Holdout {day}: W-F1={s:.4f}")

print(f"  Calibrated LODO W-F1: {np.mean(cal_scores):.4f} ({np.std(cal_scores):.4f})")
print(f"  Δ vs Baseline: {np.mean(cal_scores)-np.mean(base_scores):+.4f}")

# --- Per-fold detail ---
print("\n" + "=" * 70)
print("PER-FOLD COMPARISON")
print("=" * 70)
for i, day in enumerate(unique_days):
    print(f"  {day}: Baseline={base_scores[i]:.4f}  Calibrated={cal_scores[i]:.4f}  Δ={cal_scores[i]-base_scores[i]:+.4f}")

print(f"\nTime: {time.time()-t0:.1f}s")
