import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

np.random.seed(42)

n_samples = 200
cgpa = np.round(np.random.uniform(5.5, 10.0, n_samples), 2)
resume_score = np.round(np.random.uniform(35, 100, n_samples), 0)

placed = []
for g, r in zip(cgpa, resume_score):
    if g >= 8.0 and r >= 75:
        placed.append(1)  
    elif g <= 6.5 and r <= 50:
        placed.append(0)  
    else:
        prob = 0.3 + 0.4 * ((g - 6.5) / 3.5) + 0.3 * ((r - 50) / 50)
        prob = np.clip(prob, 0, 1)
        placed.append(np.random.binomial(1, prob))

# Create DataFrame
df = pd.DataFrame({
    'cgpa': cgpa,
    'resume_score': resume_score,
    'placed': placed
})

# Save to CSV
df.to_csv('placement.csv', index=False)
