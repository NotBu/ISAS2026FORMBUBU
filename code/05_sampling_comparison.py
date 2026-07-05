"""
05_sampling_comparison.py — Imbalanced Learning Strategy Comparison

Purpose:
  Compare 4 strategies for handling severe class imbalance (max 407x):
    1. Baseline: RF with no balancing
    2. SMOTE: Synthetic Minority Oversampling
    3. RandomUnderSampler: Random undersampling of majority
    4. class_weight='balanced_subsample': RF built-in (V2 final model)
  
  Results show that SMOTE overfits severely on this RSSI spatial data,
  while class_weight provides the best trade-off.

Usage: python code/05_sampling_comparison.py
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, '.cache')
WINDOW = 5

print("=" * 70)
print("05: Imbalanced Learning Strategy Comparison")
print("=" * 70)
t0 = time.time()

# Load or preprocess data
os.makedirs(CACHE, exist_ok=True)
X_file = os.path.join(CACHE, 'X.npy')
y_file = os.path.join(CACHE, 'y.npy')
le_file = os.path.join(CACHE, 'classes.npy')

if os.path.exists(X_file) and os.path.exists(y_file) and os.path.exists(le_file):
    print("Loading cached data...")
    X = np.load(X_file)
    y = np.load(y_file)
    classes = np.load(le_file, allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = classes
    y_e = le.transform(y)
    n_classes = len(classes)
else:
    BLE_DATA = os.path.join(BASE, 'ble_train_4d.csv')
    LABEL_FILE = os.path.join(BASE, 'cleaned_labels.csv')
    
    print("Loading BLE data (this may take a moment)...")
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
    
    labels = pd.read_csv(LABEL_FILE, parse_dates=['started_at', 'finished_at'])
    ble_ts = ble_ts.reset_index()
    ble_ts['room'] = 'unknown'
    for _, r in labels.iterrows():
        m = (ble_ts['timestamp'] >= r['started_at']) & (ble_ts['timestamp'] <= r['finished_at'])
        ble_ts.loc[m, 'room'] = r['room']
    train = ble_ts[ble_ts['room'] != 'unknown'].copy()
    del ble_ts, labels, reader, chunks
    
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
    del v, rooms, train
    
    le = LabelEncoder()
    y_e = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    np.save(X_file, X)
    np.save(y_file, y)
    np.save(le_file, le.classes_)
    print(f"Cached to {CACHE}/")

uniq, cnts = np.unique(y, return_counts=True)
imbalance_ratio = max(cnts) / min(cnts)
print(f"\nSamples: {len(X)}, Features: {X.shape[1]}, Classes: {n_classes}")
print(f"Imbalance ratio: {imbalance_ratio:.1f}x (max/min)")
print("\nClass distribution:")
for r, c in sorted(zip(uniq, cnts), key=lambda x: -x[1]):
    pct = c / len(y) * 100
    print(f"  {str(r):>15}: {c:>5} ({pct:>5.1f}%)")

# Rare room indices (6 rooms with <50 samples)
rare_names = ['503', '505', '510', '516', '517', '518']
rare_ids = [list(le.classes_).index(r) for r in le.classes_ if r in rare_names]

def run_cv_fold(tr, va, X, y, sampler_type, cw):
    """Run a single CV fold with optional sampling. Returns metrics."""
    X_tr, y_tr = X[tr], y[tr]
    
    if sampler_type == 'smote':
        target_per_class = 2000
        from collections import Counter
        orig_counts = Counter(y_tr)
        smote_strategy = {cls: max(target_per_class, count)
                          for cls, count in orig_counts.items()}
        sm = SMOTE(sampling_strategy=smote_strategy, k_neighbors=1, random_state=42)
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
    
    del rf, X_tr, y_tr, p_val, p_tr
    gc.collect()
    
    return wf, mf, twf, rf1

def evaluate_strategy(X, y, name, sampler=None, cw=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    wf_scores, mf_scores, rare_scores, train_scores = [], [], [], []
    
    for tr, va in skf.split(X, y):
        wf, mf, twf, rf1 = run_cv_fold(tr, va, X, y, sampler, cw)
        wf_scores.append(wf)
        mf_scores.append(mf)
        train_scores.append(twf)
        rare_scores.append(rf1)
    
    return {
        'name': name,
        'W-F1': np.mean(wf_scores),
        'M-F1': np.mean(mf_scores),
        'rare_F1': np.mean(rare_scores),
        'train_W-F1': np.mean(train_scores),
        'overfit_gap': np.mean(train_scores) - np.mean(wf_scores)
    }

print(f"\n{'='*70}")
print(f"{'Strategy':<35} {'W-F1':>8} {'M-F1':>8} {'Rare-F1':>8} {'Train':>8} {'Gap':>9}")
print(f"{'='*70}")

results = []
strategies = [
    ('Baseline (no balancing)', None, None),
    ('SMOTE (oversampling)', 'smote', None),
    ('RandomUnderSampler', 'rus', None),
    ("class_weight='balanced_subsample' (V2)", None, 'balanced_subsample'),
]

for name, sampler, cw in strategies:
    if sampler == 'smote' and not HAS_IMBLEARN:
        print(f"{name:<35} {'SKIP':>8} (install imbalanced-learn)")
        continue
    r = evaluate_strategy(X, y_e, name, sampler, cw)
    results.append(r)
    gap = r['overfit_gap']
    gap_str = f"OVERFIT!" if gap > 0.12 else f"{gap:.4f}"
    print(f"{r['name']:<35} {r['W-F1']:>8.4f} {r['M-F1']:>8.4f} {r['rare_F1']:>8.4f} {r['train_W-F1']:>8.4f} {gap_str:>9}")
    gc.collect()

print(f"{'='*70}")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for r in results:
    note = ""
    if r['overfit_gap'] > 0.12:
        note = " OVERFIT - synthetic RSSI invalid"
    elif 'class_weight' in r['name']:
        note = " SELECTED (V2)"
    elif r['overfit_gap'] < 0.10 and r['rare_F1'] > 0.6:
        note = " competitive"
    print(f"  {r['name']:<35}: W-F1={r['W-F1']:.4f} M-F1={r['M-F1']:.4f} Rare-F1={r['rare_F1']:.4f} Gap={r['overfit_gap']:.4f}{note}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
