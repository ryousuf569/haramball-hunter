// The dial: six recorded policies, one canvas, a playback loop.
//
// The environment ticks at 10 Hz (physics/engine.py DT = 0.1), so playback at
// 10 ticks/second is real time.

import { loadManifest, loadPolicy, isLoaded, readEpisode } from './data.js';
import { PitchRenderer } from './render.js';

const TICKS_PER_SEC = 10;

const el = {
  frame: document.getElementById('pitch-frame'),
  canvas: document.getElementById('pitch'),
  tag: document.getElementById('run-tag'),
  stops: document.getElementById('stops'),
  req: document.getElementById('r-req'),
  ach: document.getElementById('r-ach'),
  succ: document.getElementById('r-succ'),
  play: document.getElementById('play'),
  playIcon: document.getElementById('play-icon'),
  scrub: document.getElementById('scrub'),
  tick: document.getElementById('tick-count'),
  episode: document.getElementById('episode-btn'),
  outcome: document.getElementById('outcome'),
};

const ICON_PLAY = 'M3 2 L12 7 L3 12 Z';
const ICON_PAUSE = 'M3 2 H6 V12 H3 Z M8 2 H11 V12 H8 Z';

const state = {
  manifest: null,
  renderer: null,
  policyIndex: 0,
  episodeIndex: 0,
  episode: null,
  tick: 0,
  playing: true,
  lastFrame: 0,
  accumulator: 0,
};

const fmt2 = (v) => (v === null || v === undefined || Number.isNaN(v) ? '–' : v.toFixed(2));
const pct = (v) => (v === null || v === undefined || Number.isNaN(v) ? '–' : Math.round(v * 100) + '%');

// ------------------------------------------------------------- the dial ----

function buildStops(policies) {
  const inner = el.stops.querySelector('.stops-inner');
  policies.forEach((p, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'stop';
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', String(i === 0));
    b.tabIndex = -1;
    b.dataset.index = String(i);
    const label = p.requested === null ? 'Unc.' : p.requested.toFixed(2);
    b.setAttribute('aria-label',
      p.requested === null ? 'Unconstrained policy' : 'Requested rate ' + label);
    b.innerHTML = '<span class="dot"></span><span class="tag"></span>';
    b.querySelector('.tag').textContent = label;
    b.addEventListener('click', () => selectPolicy(i));
    inner.appendChild(b);
  });

  el.stops.addEventListener('keydown', (e) => {
    const n = policies.length;
    let next = null;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (state.policyIndex + 1) % n;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (state.policyIndex - 1 + n) % n;
    if (e.key === 'Home') next = 0;
    if (e.key === 'End') next = n - 1;
    if (next !== null) { e.preventDefault(); selectPolicy(next); }
  });
}

function markStops() {
  el.stops.querySelectorAll('.stop').forEach((b, i) => {
    b.setAttribute('aria-checked', String(i === state.policyIndex));
  });
}

function updateReadout() {
  const p = state.manifest.policies[state.policyIndex];
  el.req.textContent = p.requested === null ? 'none' : fmt2(p.requested);
  el.ach.textContent = fmt2(p.achieved);
  el.succ.textContent = pct(p.success);
  el.tag.textContent = p.run.toUpperCase();
}

// -------------------------------------------------------------- loading ----

async function selectPolicy(index) {
  state.policyIndex = index;
  state.episodeIndex = 0;
  markStops();
  updateReadout();

  const p = state.manifest.policies[index];
  if (!isLoaded(p)) el.frame.classList.add('loading');

  let buffer;
  try {
    buffer = await loadPolicy(p);
  } catch (err) {
    el.frame.classList.remove('loading');
    el.tag.textContent = 'RECORDING UNAVAILABLE';
    console.error(err);
    return;
  }

  if (state.policyIndex !== index) return; // a later click won the race
  el.frame.classList.remove('loading');
  state.buffer = buffer;
  showEpisode(0);
}

function showEpisode(i) {
  const p = state.manifest.policies[state.policyIndex];
  state.episodeIndex = i;
  state.episode = readEpisode(state.buffer, state.manifest, p.episodes[i]);
  state.tick = 0;
  state.accumulator = 0;
  el.scrub.max = String(state.episode.ticks - 1);
  el.scrub.value = '0';
  el.episode.textContent = 'Episode ' + (i + 1) + ' of ' + p.episodes.length + ' →';
  el.outcome.textContent = state.episode.outcome;
  updateTickLabel();
  state.renderer.drawFrame(state.episode, 0);
}

function updateTickLabel() {
  if (!state.episode) return;
  el.tick.textContent = 'tick ' + (state.tick + 1) + ' / ' + state.episode.ticks;
}

// ------------------------------------------------------------ playback -----

function loop(now) {
  requestAnimationFrame(loop);
  if (!state.episode) return;

  if (!state.playing) { state.lastFrame = now; return; }

  const dt = state.lastFrame ? Math.min((now - state.lastFrame) / 1000, 0.25) : 0;
  state.lastFrame = now;
  state.accumulator += dt * TICKS_PER_SEC;

  if (state.accumulator < 1) return;
  const advance = Math.floor(state.accumulator);
  state.accumulator -= advance;
  state.tick += advance;

  if (state.tick >= state.episode.ticks) {
    // Roll into the next episode so the dial keeps moving unattended.
    const p = state.manifest.policies[state.policyIndex];
    showEpisode((state.episodeIndex + 1) % p.episodes.length);
    return;
  }

  el.scrub.value = String(state.tick);
  updateTickLabel();
  state.renderer.drawFrame(state.episode, state.tick);
}

function setPlaying(on) {
  state.playing = on;
  state.lastFrame = 0;
  el.playIcon.setAttribute('d', on ? ICON_PAUSE : ICON_PLAY);
  el.play.setAttribute('aria-label', on ? 'Pause' : 'Play');
}

// ---------------------------------------------------------------- boot -----

async function main() {
  const manifest = await loadManifest();
  state.manifest = manifest;
  state.renderer = new PitchRenderer(el.canvas, manifest);
  state.renderer.drawEmpty(); // paper and markings before any bytes arrive

  buildStops(manifest.policies);
  updateReadout();
  setPlaying(true);

  el.play.addEventListener('click', () => setPlaying(!state.playing));
  el.scrub.addEventListener('input', () => {
    if (!state.episode) return;
    setPlaying(false);
    state.tick = Number(el.scrub.value);
    updateTickLabel();
    state.renderer.drawFrame(state.episode, state.tick);
  });
  el.episode.addEventListener('click', () => {
    if (!state.episode) return;
    const p = manifest.policies[state.policyIndex];
    showEpisode((state.episodeIndex + 1) % p.episodes.length);
  });

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      state.renderer.resize();
      if (state.episode) state.renderer.drawFrame(state.episode, state.tick);
      else state.renderer.drawEmpty();
    }, 120);
  });

  requestAnimationFrame(loop);

  // First policy blocks nothing else; the remaining ~4 MB streams in behind it.
  await selectPolicy(0);
  for (let i = 1; i < manifest.policies.length; i++) {
    loadPolicy(manifest.policies[i]).catch(() => {});
  }
}

main().catch((err) => {
  console.error(err);
  el.tag.textContent = 'LOAD FAILED, SERVE OVER HTTP';
});
