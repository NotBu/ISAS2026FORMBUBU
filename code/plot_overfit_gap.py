import matplotlib.pyplot as plt
import numpy as np

strategies = ['Baseline\n(no weight)', 'SMOTE', 'class_weight']
train_f1 = [0.9438, 0.9734, 0.9422]
val_f1   = [0.8299, 0.8274, 0.8351]
gaps     = [0.1139, 0.1459, 0.1071]

x = np.arange(len(strategies))
w = 0.3

fig, ax = plt.subplots(figsize=(5, 3))
bars1 = ax.bar(x - w/2, train_f1, w, label='Train W-F1', color='#4c72b0')
bars2 = ax.bar(x + w/2, val_f1,   w, label='Validation W-F1', color='#dd8452')

for i, (t, v, g) in enumerate(zip(train_f1, val_f1, gaps)):
    ax.text(i, max(t, v) + 0.01, f'gap={g:.4f}', ha='center', fontsize=8, fontstyle='italic')

ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.set_ylabel('Weighted F1')
ax.set_ylim(0.75, 1.0)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('overfit_gap.png', dpi=200)
print('Saved: overfit_gap.png')
