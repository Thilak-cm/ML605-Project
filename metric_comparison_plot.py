import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['RMSE', 'MAE']
with_optuna = [17.32, 15.39]
without_optuna = [192.81, 155.66]

r2_with = 0.8210
r2_without = 0.3210

x = np.arange(len(metrics))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot 1: RMSE and MAE
axes[0].bar(x - width/2, with_optuna, width, label='With Optuna', color='skyblue')
axes[0].bar(x + width/2, without_optuna, width, label='Without Optuna', color='salmon')
axes[0].set_ylabel('Error')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].set_title('Error Metrics (Lower is Better)')
for i, v in enumerate(with_optuna):
    axes[0].text(i - width/2, v + 5, f"{v:.2f}", ha='center')
for i, v in enumerate(without_optuna):
    axes[0].text(i + width/2, v + 5, f"{v:.2f}", ha='center')

# Plot 2: R2
axes[1].bar(['With Optuna', 'Without Optuna'], [r2_with, r2_without], color=['skyblue', 'salmon'])
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('R2 Score')
axes[1].set_title('R2 Score (Higher is Better)')
for i, v in enumerate([r2_with, r2_without]):
    axes[1].text(i, v + 0.03, f"{v:.2f}", ha='center')

# Place a single legend above both plots
fig.legend(['With Optuna', 'Without Optuna'],
           loc='upper right', bbox_to_anchor=(0.98, 1))

plt.suptitle('Model Performance Comparison')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
