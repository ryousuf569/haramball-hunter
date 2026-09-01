// CSV -> table. Every figure on this page is read from the same files the
// analysis scripts wrote, so nothing here is transcribed by hand.

import { loadCSV } from './data.js';

const num = (v) => Number(v);
const f = (v, n) => (Number.isFinite(v) ? v.toFixed(n) : '–');
const signed = (v, n) => {
  if (!Number.isFinite(v)) return '–';
  const s = Math.abs(v).toFixed(n);
  // A residual that rounds to zero is exact, not signed.
  return (Number(s) === 0 ? ' ' : v >= 0 ? '+' : '−') + s;
};

function cell(row, text, cls) {
  const td = document.createElement('td');
  td.textContent = text;
  if (cls) td.className = cls;
  row.appendChild(td);
}

function fill(tbody, rows, build) {
  tbody.textContent = '';
  rows.forEach((r) => {
    const tr = document.createElement('tr');
    build(tr, r);
    tbody.appendChild(tr);
  });
}

function fail(id, err) {
  const tbody = document.getElementById(id);
  if (!tbody) return;
  const tr = document.createElement('tr');
  const td = document.createElement('td');
  td.colSpan = 8;
  td.className = 'dim';
  td.textContent = 'could not load: serve this over HTTP, not file://';
  tr.appendChild(td);
  tbody.appendChild(tr);
  console.error(err);
}

// -------------------------------------------------- frozen-policy sweep ----
// assets/speed_probe.csv: 150,000 attacker-ticks per policy, weights frozen.

async function probeTable() {
  const rows = (await loadCSV('assets/speed_probe.csv'))
    .filter((r) => r.requested && r.requested !== 'nan');

  fill(document.getElementById('probe-rows'), rows, (tr, r) => {
    const req = num(r.requested);
    const ach = num(r.c_slow);
    const err = ach - req;
    const feasible = ach <= req + 1e-9;
    if (!feasible) tr.className = 'infeasible';
    cell(tr, r.run);
    cell(tr, r.seed, 'num');
    cell(tr, f(req, 2), 'num');
    cell(tr, f(ach, 4), 'num');
    cell(tr, signed(err, 4), 'num');
    cell(tr, f(num(r.mean_speed), 2), 'num');
    cell(tr, feasible ? 'yes' : 'no', 'num ' + (feasible ? 'yes' : 'no'));
  });
}

// ------------------------------------------------- per-target aggregates ----
// assets/constraint_targets.csv, written by model/constrained_analysis.py.

async function targetTable() {
  const rows = await loadCSV('assets/constraint_targets.csv');

  fill(document.getElementById('target-rows'), rows, (tr, r) => {
    cell(tr, f(num(r.requested), 2));
    cell(tr, r.n, 'num');
    cell(tr, f(num(r.achieved), 4), 'num');
    cell(tr, f(num(r.achieved_sd), 4), 'num');
    cell(tr, f(num(r.success), 3), 'num');
    cell(tr, f(num(r.success_sd), 3), 'num');
    cell(tr, f(num(r.speed), 2), 'num');
    cell(tr, f(num(r.ep_len), 1), 'num');
  });
}

// ----------------------------------------------- per-seed reproducibility ---
// Spread across seeds at each target, from the frozen-policy probe.

async function seedTable() {
  const rows = (await loadCSV('assets/speed_probe.csv'))
    .filter((r) => r.requested && r.requested !== 'nan');

  const byTarget = new Map();
  rows.forEach((r) => {
    const k = num(r.requested);
    if (!byTarget.has(k)) byTarget.set(k, []);
    byTarget.get(k).push(num(r.c_slow));
  });

  const out = [...byTarget.entries()].sort((a, b) => a[0] - b[0]).map(([k, v]) => {
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1));
    return { k, v, mean, sd };
  });

  fill(document.getElementById('seed-rows'), out, (tr, r) => {
    cell(tr, f(r.k, 2));
    r.v.forEach((x) => cell(tr, f(x, 3), 'num'));
    cell(tr, f(r.mean, 3), 'num');
    cell(tr, f(r.sd, 3), 'num');
    cell(tr, f(Math.max(...r.v) - Math.min(...r.v), 3), 'num');
  });
}

// --------------------------------------------- crowd_disc constraint sweep --
// assets/constraint_targets_crowd{,_5m}.csv: slow held at d=0.75, crowd_disc
// swept 0.05/0.10/0.15, training-log tail-mean aggregates per target.

async function crowdTargetTable(path, id) {
  const rows = await loadCSV(path);

  fill(document.getElementById(id), rows, (tr, r) => {
    cell(tr, f(num(r.requested), 2));
    cell(tr, r.n, 'num');
    cell(tr, f(num(r.crowd), 4), 'num');
    cell(tr, f(num(r.crowd_sd), 4), 'num');
    cell(tr, f(num(r.slow), 3), 'num');
    cell(tr, f(num(r.success), 3), 'num');
    cell(tr, f(num(r.ep_len), 1), 'num');
  });
}

// assets/constraint_runs_crowd{,_5m}.csv: one row per seed x budget.

async function crowdRunTable() {
  const [r25, r5m] = await Promise.all([
    loadCSV('assets/constraint_runs_crowd.csv'),
    loadCSV('assets/constraint_runs_crowd_5m.csv'),
  ]);
  const rows = r25.map((r) => ({ ...r, budget: '2.5M' }))
    .concat(r5m.map((r) => ({ ...r, budget: '5M' })));

  fill(document.getElementById('crowd-run-rows'), rows, (tr, r) => {
    const ok = num(r.ok_all) === 1;
    if (!ok) tr.className = 'infeasible';
    cell(tr, r.run);
    cell(tr, r.budget, 'num');
    cell(tr, f(num(r.requested), 2), 'num');
    cell(tr, f(num(r.crowd), 4), 'num');
    cell(tr, f(num(r.slow), 3), 'num');
    cell(tr, f(num(r.success), 3), 'num');
    cell(tr, f(num(r.ep_len), 1), 'num');
    cell(tr, ok ? 'yes' : 'no', 'num ' + (ok ? 'yes' : 'no'));
  });
}

Promise.all([
  probeTable().catch((e) => fail('probe-rows', e)),
  targetTable().catch((e) => fail('target-rows', e)),
  seedTable().catch((e) => fail('seed-rows', e)),
  crowdTargetTable('assets/constraint_targets_crowd.csv', 'crowd-target-rows')
    .catch((e) => fail('crowd-target-rows', e)),
  crowdTargetTable('assets/constraint_targets_crowd_5m.csv', 'crowd-target-5m-rows')
    .catch((e) => fail('crowd-target-5m-rows', e)),
  crowdRunTable().catch((e) => fail('crowd-run-rows', e)),
]);
