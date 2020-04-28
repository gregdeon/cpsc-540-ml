# Example of how to work with model predictions...

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results_csv = '../data/test_results_transformer_moves.csv'
# results_csv = '../data/test_results_stockfish_score.csv'
df = pd.read_csv(results_csv)

p_correct = df[df['move_played'] == True]['p_model']
avg_nll = np.average(-np.log(p_correct))
print('NLL:', avg_nll)

plt.hist(p_correct, color='k', bins=np.geomspace(1e-5, 1e0, 21), rwidth=0.8, zorder=10)
plt.xscale('log')
plt.grid()
plt.show()
