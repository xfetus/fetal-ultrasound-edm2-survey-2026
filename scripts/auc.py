import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Example data: 1 = Real, 0 = Synthetic
ground_truth = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])

# Predictions from three raters
rater_preds = {
    'Rater 1': np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0]),
    'Rater 2': np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0]),
    'Rater 3': np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0])
}

# Compute AUC per rater
rater_aucs = {rater: roc_auc_score(ground_truth, preds) 
              for rater, preds in rater_preds.items()}

# Plot
plt.figure(figsize=(8, 5))
plt.bar(rater_aucs.keys(), rater_aucs.values(), color='skyblue')
plt.axhline(y=0.5, color='red', linestyle='--', label='Chance (AUC = 0.5)')
plt.ylabel('AUC')
plt.title('Real vs. Synthetic Discrimination AUC per Rater')
plt.ylim(0, 1)
plt.legend()
plt.show()
