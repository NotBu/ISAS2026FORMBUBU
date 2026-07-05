"""
07_delay_analysis.py — Analyze Label Delay vs BLE Signal

Purpose:
  Find the actual delay between when a caregiver ENTERS a room
  (detected by BLE signal change) and when they RECORD the label
  (started_at). This determines the optimal buffer for time correction.

Method:
  1. For each label, compute "room signature" = mean RSSI vector
     across all timestamps inside [started_at, finished_at]
  2. Scan backwards from started_at, compute cosine similarity
     between each timestamp's RSSI and the room signature
  3. Find the point where similarity crosses a threshold (entry point)
  4. delay = started_at - entry_time
  5. Histogram of all delays → median = adaptive buffer

Output:
  - results/delay_histogram.png
  - Console: median delay, stats per room

Usage: python code/07_delay_analysis.py
"""

import pandas as pd
import numpy as np
import time, os, warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, '.cache')
WINDOW = 5

print("=" * 60)
print("07: Label Delay Analysis (Adaptive Buffer)")
print("=" * 60)
t0 = time.time()

# 1. Load cached data if available, else load & cache
os.makedirs(CACHE, exist_ok=True)
ble_file = os.path.join(CACHE, 'ble_pivot.npy')
idx_file = os.path.join(CACHE, 'ble_index.npy')

if os.path.exists(ble_file) and os.path.exists(idx_file):
    print("Loading cached BLE data...")
    ble_values = np.load(ble_file)
    ble_index = np.load(idx_file, allow_pickle=True)
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
    ble_values = ble.values
    ble_index = np.array([t.to_datetime64() for t in ble.index])
    np.save(ble_file, ble_values)
    np.save(idx_file, ble_index)
    del ble

print(f"  BLE timestamps: {len(ble_index)}")

# 2. Load labels
labels = pd.read_csv(os.path.join(BASE, 'cleaned_labels.csv'),
    parse_dates=['started_at', 'finished_at'])
print(f"  Labels: {len(labels)}")

# 3. For each label, find delay
MAX_LOOKBACK = pd.Timedelta(seconds=120)
SIMILARITY_THRESHOLD = 0.85

delays = []
label_info = []

for idx, (_, row) in enumerate(labels.iterrows()):
    start = row['started_at']
    end = row['finished_at']
    room = row['room']
    duration = (end - start).total_seconds()

    # Skip very short labels (< 5s, probably noise)
    if duration < 5:
        continue

    # Get timestamps inside label period
    start_ts = start.to_datetime64()
    end_ts = end.to_datetime64()
    mask = (ble_index >= start_ts) & (ble_index <= end_ts)
    interior = ble_values[mask]

    if len(interior) < 3:
        continue

    # Room signature = mean RSSI vector inside label
    room_sig = np.mean(interior, axis=0)

    # Get timestamps before started_at
    lookback_start = (start - MAX_LOOKBACK).to_datetime64()
    before_mask = (ble_index >= lookback_start) & (ble_index < start_ts)
    before_vals = ble_values[before_mask]
    before_ts = ble_index[before_mask]

    if len(before_vals) < 3:
        continue

    # Compute cosine similarity (shift RSSI to positive: +100)
    offset = 100.0
    sig_pos = room_sig + offset
    sig_norm = sig_pos / np.linalg.norm(sig_pos)

    sims = []
    for v in before_vals:
        v_pos = v + offset
        v_norm = v_pos / np.linalg.norm(v_pos)
        sim = float(np.dot(sig_norm, v_norm))
        sims.append(sim)
    sims = np.array(sims)

    # Find entry: scan backwards, find where similarity drops below threshold
    # (i.e., the first point from the end where signal doesn't match the room)
    entry_idx = None
    for i in range(len(sims) - 1, -1, -1):
        if sims[i] < SIMILARITY_THRESHOLD:
            entry_idx = i + 1  # first timestamp that matches
            break

    if entry_idx is None or entry_idx >= len(before_ts):
        continue

    entry_time = pd.Timestamp(before_ts[entry_idx])
    delay_sec = (start - entry_time).total_seconds()

    if delay_sec < 0 or delay_sec > 120:
        continue

    delays.append(delay_sec)
    label_info.append({
        'room': room,
        'delay_sec': delay_sec,
        'duration_min': duration / 60,
        'similarity_at_start': sims[-1] if len(sims) > 0 else 0
    })

    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx+1}/{len(labels)} labels...")

print(f"\n  Found delays for {len(delays)}/{len(labels)} labels")
df_delays = pd.DataFrame(label_info)

if len(delays) == 0:
    print("No delays found. Try lowering SIMILARITY_THRESHOLD.")
    exit()

# 4. Statistics
delays_arr = np.array(delays)
print(f"\n{'='*60}")
print("DELAY STATISTICS")
print(f"{'='*60}")
print(f"  Median delay: {np.median(delays_arr):.1f}s")
print(f"  Mean delay:   {np.mean(delays_arr):.1f}s")
print(f"  Std delay:    {np.std(delays_arr):.1f}s")
print(f"  P25:          {np.percentile(delays_arr, 25):.1f}s")
print(f"  P75:          {np.percentile(delays_arr, 75):.1f}s")
print(f"  P90:          {np.percentile(delays_arr, 90):.1f}s")
print(f"  Min:          {np.min(delays_arr):.1f}s")
print(f"  Max:          {np.max(delays_arr):.1f}s")

# Per-room stats
print(f"\n{'='*60}")
print("TOP 10 ROOMS BY MEDIAN DELAY")
print(f"{'='*60}")
room_stats = df_delays.groupby('room')['delay_sec'].agg(['median','mean','count','std'])
room_stats = room_stats.sort_values('count', ascending=False)
print(room_stats.head(10).to_string())

# 5. Plot histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
ax.hist(delays_arr, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(np.median(delays_arr), color='red', linestyle='--',
           label=f"Median={np.median(delays_arr):.1f}s")
ax.axvline(1.5, color='green', linestyle=':',
           label=f"Current buffer=1.5s")
ax.set_xlabel('Delay (seconds)')
ax.set_ylabel('Number of labels')
ax.set_title(f'Label Delay Distribution (n={len(delays)})')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Boxplot by room
ax = axes[1]
top_rooms = room_stats.head(8).index.tolist()
room_data = [df_delays[df_delays['room'] == r]['delay_sec'].values for r in top_rooms]
bp = ax.boxplot(room_data, labels=top_rooms, vert=True, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.axhline(np.median(delays_arr), color='red', linestyle='--', alpha=0.7,
           label=f"Overall median={np.median(delays_arr):.1f}s")
ax.set_ylabel('Delay (seconds)')
ax.set_title('Delay by Room (top 8 most frequent)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
out_path = os.path.join(BASE, 'results', 'delay_histogram.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved histogram: {out_path}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
