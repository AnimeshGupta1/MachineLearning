import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions


df = pd.read_csv('placement.csv')

sns.scatterplot(x='cgpa', y='resume_score', hue='placed', data=df)
plt.xlabel('CGPA')
plt.ylabel('Resume Score')
plt.title('Placement Scatter Plot')
plt.show()

X = df.iloc[:, 0:2]
y = df.iloc[:, -1]
p = Perceptron()
p.fit(X, y)
print("Weights:", p.coef_)
print("Intercept:", p.intercept_)
plot_decision_regions(X.values, y.values, clf=p, legend=2)
plt.xlabel('CGPA')
plt.ylabel('Resume Score')
plt.title('Perceptron Decision Regions')
plt.show()
