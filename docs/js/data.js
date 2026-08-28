// Manifest + episode-binary access.
//
// Rollouts are pre-rendered by docs/binaries/recorder.py; nothing is
// simulated here. Per episode, starting at `offset`, with
// stride = 90 + nx*ny bytes per tick:
//
//   pos     int16[ticks*42]   offset              positions in cm, 21 x (x,y)
//   ball    int16[ticks*2]    offset + ticks*84
//   state   uint8[ticks]      offset + ticks*88   0 held 1 flight 2 loose 3 other
//   holder  int8[ticks]       offset + ticks*89   player row 0..20, -1 none
//   heat    uint8[ticks*1054] offset + ticks*90   attacker pitch control x 255
//
// Each episode block is ticks*stride bytes and stride is even, so the int16
// views are always 2-byte aligned and can be taken over the ArrayBuffer
// without copying.

export const BASE = 'binaries/';

export async function loadManifest() {
  const res = await fetch(BASE + 'manifest.json');
  if (!res.ok) throw new Error('manifest ' + res.status);
  return res.json();
}

// One policy's .bin, fetched at most once. Concurrent callers share a promise.
const buffers = new Map();

export function loadPolicy(policy) {
  let p = buffers.get(policy.file);
  if (!p) {
    p = fetch(BASE + policy.file)
      .then((res) => {
        if (!res.ok) throw new Error(policy.file + ' ' + res.status);
        return res.arrayBuffer();
      })
      .catch((err) => { buffers.delete(policy.file); throw err; });
    buffers.set(policy.file, p);
  }
  return p;
}

export function isLoaded(policy) {
  return buffers.has(policy.file);
}

// Typed views over one episode. No bytes are copied.
export function readEpisode(buffer, manifest, episode) {
  const { offset, ticks } = episode;
  const ncell = manifest.grid.nx * manifest.grid.ny;
  return {
    ticks,
    ncell,
    pos: new Int16Array(buffer, offset, ticks * 42),
    ball: new Int16Array(buffer, offset + ticks * 84, ticks * 2),
    state: new Uint8Array(buffer, offset + ticks * 88, ticks),
    holder: new Int8Array(buffer, offset + ticks * 89, ticks),
    heat: new Uint8Array(buffer, offset + ticks * 90, ticks * ncell),
    outcome: episode.outcome,
    seed: episode.seed,
  };
}

// Positions in metres for tick t: [x0, y0, x1, y1, ...] over 21 players.
export function playersAt(ep, t) {
  const out = new Float32Array(42);
  const base = t * 42;
  for (let i = 0; i < 42; i++) out[i] = ep.pos[base + i] / 100;
  return out;
}

export function ballAt(ep, t) {
  return [ep.ball[t * 2] / 100, ep.ball[t * 2 + 1] / 100];
}

// The 31x34 pitch-control field for tick t, indexed i*ny + j.
export function heatAt(ep, t) {
  return ep.heat.subarray(t * ep.ncell, (t + 1) * ep.ncell);
}

// ------------------------------------------------------------------ CSV ----

// Small, well-formed research CSVs: no quoted fields, no embedded newlines.
export function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(',');
  return lines.slice(1).map((line) => {
    const cells = line.split(',');
    const row = {};
    header.forEach((h, i) => { row[h] = cells[i]; });
    return row;
  });
}

export async function loadCSV(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(path + ' ' + res.status);
  return parseCSV(await res.text());
}
