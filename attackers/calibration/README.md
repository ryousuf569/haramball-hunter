# Attacker formation calibration

## What this is

`make_initial_world` used to place all ten attackers at fixed offsets from a
fixed point, so every episode started from exactly the same picture. It built an
RNG from the seed and then never used it for positions. An agent trained against
that can memorise one arrangement of players instead of learning anything about
breaking a low block, and any claim that it learned to break low blocks would be
unsupportable. What we calibrated here is the starting shape of the attacking
team, so that changing the seed actually gives a different, and still realistic,
opening position.

## How we did it

We reused the freeze frames that the defender calibration already pulled from
StatsBomb 360 (8 matches, 2032 low block frames, stored in
`defenders/calibration/low_block_defenders.csv`). Those files record the
attacking team's players as well as the defenders. We kept the 869 frames where
at least ten attackers were inside the camera view and took the ten most
advanced players in each, which drops the occasional deep straggler and gives
every frame the same number of slots. The calibration data is in the defending
team's frame, where attackers move toward x=0, so we rotated all of it 180
degrees into sim coordinates first, where attackers move toward x=105.

For each frame we reduced the ten players to four numbers: where the team's
centre of mass sat up the pitch, where it sat across the pitch, and how
stretched the shape was lengthwise and widthwise. Every player was then stored
as an offset from that centre, measured in units of that frame's stretch, so a
compact shape and a spread out shape become comparable. Players were sorted by
how far up the pitch they were, split into the 2-5-3 lines, then sorted left to
right inside each line, so slot 0 means the same role in every frame. We fitted
a normal distribution to each of the four frame numbers and to each of the ten
slot offsets, clipped to the 5th and 95th percentile of what actually occurred.
Sampling runs the same steps backwards: draw a centre, draw a stretch, draw the
ten offsets, undo the scaling. The two stretch numbers are drawn together rather
than independently, because in the real frames a shape that is compressed
lengthwise tends to be compressed widthwise too.

## Why it is set up this way

The attacking centre of mass sits about 2.6m behind the defensive back line in
the real frames, which is a tight and useful relationship, so we fitted it and
saved it. It is not the default though. `defenders.py` deliberately parks the
block 15m deeper than the real data says (`HARAM_DEPTH_OFFSET`), and anchoring
the attackers to that line would drag them 15m deeper too, leaving the forwards
camped in the six yard box and permanently offside. So by default the attacking
shape is placed at the depth real attackers actually take up, around x=66, and
the relative anchoring is available as an option for anyone who wants it. Worth
knowing about the fit: because the ten slots are drawn independently, the two
widest players rarely hit their extremes in the same draw, so sampled formations
come out about 10 percent narrower than real ones (about 46m across instead of
51m). Depth, centre position and per player scatter all match the data.

## Files

Run `python attackers/calibrate_attacker_formation.py` from the repo root to
regenerate everything here.

- `attacker_frame_params.csv` - the four frame level numbers, plus the gap to
  the defensive back line and the correlation between the two stretch numbers.
- `attacker_slot_params.csv` - mean and spread of each of the ten slot offsets.
- `attacker_shape_fit.png` - what the fitted quantities look like in the data.
  The old fixed starting depth is marked, and it sits near the middle of the
  real distribution, so the old position was a reasonable sample. It was just
  the only one.
- `attacker_slot_scatter.png` - the real per slot clouds against the fitted mean
  and spread.
- `attacker_samples.png` - six formations drawn from the finished model.
