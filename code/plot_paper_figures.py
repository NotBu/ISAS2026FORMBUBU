"""
plot_paper_figures.py — Figures for the paper
1. per_class_f1.png  — bar chart F1 per room
2. config_comparison.png — W-F1/M-F1 across V1-V5
3. confusion_matrix.png — confusion matrix V3
"""

import numpy as np, os, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

X = np.load(os.path.join(BASE, '.cache', 'X_smooth.npy'))
y_labels = np.load(os.path.join(BASE, '.cache', 'y.npy'), allow_pickle=True)
classes = np.load(os.path.join(BASE, '.cache', 'classes.npy'), allow_pickle=True)
le = LabelEncoder(); le.classes_ = classes; y_e = le.transform(y_labels)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_t, all_p = [], []
for tr, va in skf.split(X, y_e):
    rf = RandomForestClassifier(n_estimators=100, max_depth=20,
        class_weight='balanced_subsample', random_state=42, n_jobs=1)
    rf.fit(X[tr], y_e[tr])
    all_p.extend(rf.predict(X[va]))
    all_t.extend(y_e[va])
all_t, all_p = np.array(all_t), np.array(all_p)

# Per-class F1
per_class_f1 = f1_score(all_t, all_p, average=None)
room_names = le.classes_
order = np.argsort(per_class_f1)[::-1]

# --- Figure 1: Per-class F1 ---
fig, ax = plt.subplots(figsize=(8, 3.5))
colors = ['#e15759' if f1 < 0.75 else '#76b7b2' for f1 in per_class_f1[order]]
ax.bar(range(len(order)), per_class_f1[order], color=colors)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(room_names[order], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('F1 score')
ax.set_ylim(0, 1.0)
ax.axhline(0.835, color='gray', linestyle='--', linewidth=1)
ax.text(21, 0.84, 'macro F1 = 0.834', fontsize=8, ha='right')
ax.grid(axis='y', alpha=0.3)
ax.set_title('Per-class F1 (V3)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'per_class_f1.png'), dpi=200)
plt.close()
print('Saved per_class_f1.png')

# --- Figure 2: Config comparison ---
fig, ax = plt.subplots(figsize=(6.5, 3))
configs = ['V1\n17-class', 'V2\n22-class\nweight', 'V3\n+ SG', 'V3+4s\n+ buffer', 'V4\n+ relabel']
w_f1 = [0.8315, 0.8246, 0.8351, 0.8347, 0.8171]
m_f1 = [0.8193, 0.8191, 0.8336, 0.8350, 0.7713]
x = np.arange(len(configs))
w = 0.35
b1 = ax.bar(x - w/2, w_f1, w, label='Weighted F1', color='#4e79a7')
b2 = ax.bar(x + w/2, m_f1, w, label='Macro F1', color='#f28e2b')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=8)
ax.set_ylim(0.75, 0.85)
ax.set_ylabel('F1 score')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.set_title('Performance across configurations', fontsize=11)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{b.get_height():.4f}', ha='center', fontsize=7)
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{b.get_height():.4f}', ha='center', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'config_comparison.png'), dpi=200)
plt.close()
print('Saved config_comparison.png')

# --- Figure 3: Confusion matrix ---
cm = confusion_matrix(all_t, all_p)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues')
# annotate only if > 1% of row max
row_max = cm.max(axis=1, keepdims=True)
threshold = row_max * 0.01
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if cm[i, j] > 0:
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=6, color='black' if cm[i,j] < row_max[i]*0.6 else 'white')
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=90, fontsize=6)
ax.set_yticklabels(classes, fontsize=6)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion matrix (V3)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'confusion_matrix.png'), dpi=200)
plt.close()
print('Saved confusion_matrix.png')
