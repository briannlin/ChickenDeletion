import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

file_path = 'qtable_full_simplified_episode_results.csv'
df = pd.read_csv(file_path)

sns.set(style="whitegrid")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Epsilon vs Cumulative Reward (red)
sns.regplot(x='epsilon', y='cumulative_reward', data=df, order=5, scatter_kws={'s': 25}, line_kws={'color': 'red'}, ax=axes[0])
axes[0].invert_xaxis()
axes[0].set_title('Epsilon vs Cumulative Reward')
axes[0].set_xlabel('Epsilon')
axes[0].set_ylabel('Cumulative Reward')
axes[0].grid(True)
axes[0].set_xticks(np.arange(0, 1, 0.1))

# Epsilon vs Episode Length (green)
sns.regplot(x='epsilon', y='episode_length', data=df, order=5, scatter_kws={'s': 25}, line_kws={'color': 'green'}, ax=axes[1])
axes[1].invert_xaxis()
axes[1].set_title('Epsilon vs Episode Length')
axes[1].set_xlabel('Epsilon')
axes[1].set_ylabel('Episode Length')
axes[1].grid(True)
axes[1].set_xticks(np.arange(0, 1, 0.1))

plt.tight_layout(pad=1.5)
plt.show()
