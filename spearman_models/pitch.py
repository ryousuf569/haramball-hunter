import numpy as np
import matplotlib.pyplot as plt
from spearman_models.ppcf import PPCF_grid
from schema import player_dt

players = np.array([
    ([20.0, 30.0], [1.5, -0.5], 'attacker', 0),
    ([35.0, 40.0], [1.0, 0.2],  'attacker', 0),
    ([30.0, 34.0], [-0.5, 0.3], 'defender', 0),
    ([40.0, 45.0], [-1.0, -0.8],'defender', 0),
], dtype=player_dt)

''' 
FIFA Regulation 105 x 68 meters - Half Field 52.5 x 68 
Center Circle has a radius of 9.15m (52.5 + 9.15 = 61.65)
Field dimensions from just behind the attackers side of the circle: 62 x 68
We take those dimensions at a ratio of 2/1
''' 

cell_size = 2.0

x_range = (np.arange(31) + 0.5) * cell_size
y_range = (np.arange(34) + 0.5) * cell_size

X, Y = np.meshgrid(x_range, y_range, indexing="ij")

# Vectorized: flatten the grid to (n_cells, 2) targets, evaluate all cells and
# all players in one PPCF_grid call, then split by team and reshape to the grid.
# (This replaces the former nested for-loop over X.shape[0] x X.shape[1] cells.)
targets = np.stack([X.ravel(), Y.ravel()], axis=1)
result = PPCF_grid(targets, players) # (n_cells, n_players)

is_att = (players['team'] == 'attacker')
PC_att = result[:, is_att].sum(axis=1).reshape(X.shape)
PC_def = result[:, ~is_att].sum(axis=1).reshape(X.shape)

print("min/max PC_att:", PC_att.min(), PC_att.max())
print("min/max PC_def:", PC_def.min(), PC_def.max())
print("max deviation from 1:", np.max(np.abs(PC_att + PC_def - 1)))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(PC_att.T, origin='lower', extent=[0, 62, 0, 68],
                      cmap='Reds', vmin=0, vmax=1)
axes[0].set_title('Attacker Pitch Control')
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(PC_def.T, origin='lower', extent=[0, 62, 0, 68],
                      cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Defender Pitch Control')
axes[1].set_xlabel('x (m)')
axes[1].set_ylabel('y (m)')
plt.colorbar(im1, ax=axes[1])

# overlay player positions
for p in players:
    color = 'darkred' if p['team'] == 'attacker' else 'darkblue'
    marker = 'o' if p['team'] == 'attacker' else 's'
    for ax in axes:
        ax.scatter(p['position'][0], p['position'][1], c=color, marker=marker,
                   s=100, edgecolors='white', linewidths=1.5, zorder=5)
        ax.arrow(p['position'][0], p['position'][1],
                 p['velocity'][0], p['velocity'][1],
                 head_width=1.2, color=color, zorder=5)

plt.tight_layout()
plt.show()