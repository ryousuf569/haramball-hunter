from statsbombpy import sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COMP_ID = 55
SEASON_ID = 282
N_MATCHES = 8

BALL_X_MAX = 80.0
MIN_POSSESSION_SECONDS = 5.0
MIN_DEFENDERS = 8
BLOCK_DEPTH_MAX = 60.0

matches = sb.matches(competition_id=COMP_ID, season_id=SEASON_ID)
match_ids = sorted(matches['match_id'])[:N_MATCHES]

frame_rows = []
player_rows = []

for match_id in match_ids:
    events = sb.events(match_id=match_id)
    frames = sb.frames(match_id=match_id, fmt='dict')

    events = events[events['location'].notna()]
    events['abs_time'] = events['minute'] * 60 + events['second']
    events['poss_start'] = events.groupby('possession')['abs_time'].transform('min')
    events['poss_elapsed'] = events['abs_time'] - events['poss_start']
    ev = events.set_index('id')

    for frame in frames:
        eid = frame['event_uuid']
        if eid not in ev.index:
            continue

        e = ev.loc[eid]
        if e['team'] != e['possession_team']:
            continue

        ff = frame['freeze_frame']
        defenders = [p for p in ff if not p['teammate'] and not p['keeper']]
        attackers = [p for p in ff if p['teammate']]

        if len(defenders) < MIN_DEFENDERS:
            continue

        ball_x, ball_y = e['location'][0], e['location'][1]
        if ball_x > BALL_X_MAX or e['poss_elapsed'] < MIN_POSSESSION_SECONDS:
            continue

        dx = np.array([p['location'][0] for p in defenders])
        dy = np.array([p['location'][1] for p in defenders])

        if dx.mean() > BLOCK_DEPTH_MAX:
            continue

        order = np.argsort(dx)
        back_sel = order[:5]
        mid_sel = order[5:9]

        back_x, back_y = dx[back_sel], dy[back_sel]
        mid_x, mid_y = dx[mid_sel], dy[mid_sel]

        back_y_sorted = np.sort(back_y)
        mid_y_sorted = np.sort(mid_y)

        att_x = np.array([p['location'][0] for p in attackers]) if attackers else np.array([])

        frame_rows.append({
            'match_id': match_id,
            'event_id': eid,
            'minute': e['minute'],
            'type': e['type'],
            'poss_elapsed': e['poss_elapsed'],
            'ball_x': ball_x,
            'ball_y': ball_y,
            'n_def': len(defenders),
            'back_line_x': back_x.mean(),
            'mid_line_x': mid_x.mean(),
            'line_sep': mid_x.mean() - back_x.mean(),
            'back_centroid_y': back_y.mean(),
            'mid_centroid_y': mid_y.mean(),
            'back_width': back_y.max() - back_y.min(),
            'mid_width': mid_y.max() - mid_y.min(),
            'back_x_scatter': back_x.std(),
            'mid_x_scatter': mid_x.std(),
            'back_gap_min': np.diff(back_y_sorted).min(),
            'back_gap_max': np.diff(back_y_sorted).max(),
            'mid_gap_min': np.diff(mid_y_sorted).min(),
            'highest_att_x': att_x.min() if len(att_x) else np.nan,
        })

        for slot, (px, py) in enumerate(zip(back_x[np.argsort(back_y)], np.sort(back_y))):
            player_rows.append({'event_id': eid, 'match_id': match_id, 'line': 'back',
                                'slot': slot, 'x': px, 'y': py,
                                'ball_x': ball_x, 'ball_y': ball_y,
                                'dx_from_line': px - back_x.mean()})
        for slot, (px, py) in enumerate(zip(mid_x[np.argsort(mid_y)], np.sort(mid_y))):
            player_rows.append({'event_id': eid, 'match_id': match_id, 'line': 'mid',
                                'slot': slot + 5, 'x': px, 'y': py,
                                'ball_x': ball_x, 'ball_y': ball_y,
                                'dx_from_line': px - mid_x.mean()})

df = pd.DataFrame(frame_rows)
players = pd.DataFrame(player_rows)

print(f"low-block frames: {len(df)}   defender rows: {len(players)}")
print()
print("--- depth: back_line_x vs ball_x ---")
fit = np.polyfit(df['ball_x'], df['back_line_x'], 1)
print(f"back_line_x = {fit[0]:.3f} * ball_x + {fit[1]:.2f}   corr={df['ball_x'].corr(df['back_line_x']):.3f}")
print(f"line_sep (mid - back):  mean={df['line_sep'].mean():.2f}  sd={df['line_sep'].std():.2f}")
print()
print("--- lateral: centroid_y vs ball_y (defenders.py gain=0.2) ---")
gb = np.polyfit(df['ball_y'], df['back_centroid_y'] - 34, 1)
gm = np.polyfit(df['ball_y'] - 34, df['mid_centroid_y'] - 34, 1)
print(f"back gain={np.polyfit(df['ball_y'] - 34, df['back_centroid_y'] - 34, 1)[0]:.3f}   mid gain={gm[0]:.3f}")
print()
print("--- width / spacing (defenders.py assumes 40m back, gaps 8-12) ---")
print(df[['back_width', 'mid_width', 'back_gap_min', 'back_gap_max', 'mid_gap_min']].describe().loc[['mean', 'std', '25%', '50%', '75%']])
print()
print("--- NON-UNISON evidence: within-line x scatter (rigid model predicts 0) ---")
print(df[['back_x_scatter', 'mid_x_scatter']].describe().loc[['mean', 'std', '50%', 'max']])
print()
print("--- per-slot depth stagger (rigid model predicts identical) ---")
print(players.groupby('slot')['dx_from_line'].agg(['mean', 'std', 'count']))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

axes[0, 0].hist(df['back_line_x'], bins=40, color='steelblue')
axes[0, 0].set_title(f'Backline depth (n={len(df)})')
axes[0, 0].set_xlabel('back_line_x')

axes[0, 1].scatter(df['ball_x'], df['back_line_x'], s=4, alpha=0.2)
axes[0, 1].plot([0, 80], np.polyval(fit, [0, 80]), 'r-')
axes[0, 1].set_xlabel('ball_x')
axes[0, 1].set_ylabel('back_line_x')
axes[0, 1].set_title('Depth response to ball')

axes[0, 2].scatter(df['ball_y'], df['back_centroid_y'], s=4, alpha=0.2)
axes[0, 2].plot([0, 68], 34 + gb[0] * (np.array([0, 68]) - 34), 'r-')
axes[0, 2].set_xlabel('ball_y')
axes[0, 2].set_ylabel('back_centroid_y')
axes[0, 2].set_title(f'Lateral shift (gain={gb[0]:.2f})')

axes[1, 0].hist(df['back_x_scatter'], bins=40, color='darkorange')
axes[1, 0].set_title('Within-backline x scatter')
axes[1, 0].set_xlabel('sd of x within line (0 = perfect unison)')

axes[1, 1].hist(df['back_width'], bins=40, color='seagreen')
axes[1, 1].axvline(40, color='red', ls='--')
axes[1, 1].set_title('Backline width (red = current 40m)')
axes[1, 1].set_xlabel('back_width')

gaps = np.concatenate([df['back_gap_min'], df['back_gap_max']])
axes[1, 2].hist(gaps, bins=40, color='purple')
axes[1, 2].axvline(8, color='red', ls='--')
axes[1, 2].axvline(12, color='red', ls='--')
axes[1, 2].set_title('Backline gaps (red = snapback 8-12 band)')
axes[1, 2].set_xlabel('gap (m)')

plt.tight_layout()
plt.savefig('defenders/low_block_diagnostic.png', dpi=120)

df.to_csv('defenders/low_block_frames.csv', index=False)
players.to_csv('defenders/low_block_defenders.csv', index=False)
