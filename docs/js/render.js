// Canvas drawing: paper, heatmap, pitch markings, players, ball.
//
// One world-to-canvas transform, set up once per resize. The heatmap is the
// only place the warm orange appears; everything else is the hairline palette.

import { playersAt, ballAt, heatAt } from './data.js';

export const PITCH_X = 105;
export const PITCH_Y = 68;
const PAD = 3; // metres of paper around the touchlines

// Palette, mirroring docs/css/site.css. Canvas cannot read custom properties
// cheaply per frame, so the handful used here are inlined.
const C = {
  paper: '#f9f9f7',
  line: '#c5c6cb',
  zone: '#75777b',
  attacker: '#2d5ea6',
  defender: '#c5c6cb',
  keeper: '#75777b',
  ink: '#1a1c1b',
  heat: [196, 112, 63], // #c4703f
};

export class PitchRenderer {
  constructor(canvas, manifest) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.manifest = manifest;
    this.grid = manifest.grid;
    this.zone = manifest.zone || { x: 86, y: 34, r: 8 };

    // Offscreen image; drawImage does the smoothing for us. One transparent
    // cell of padding on every side so the grid fades out instead of ending in
    // a hard rectangle. The field only exists over the attacking half, and a
    // visible seam at x = 43 looks like a bug even though it isn't.
    this.hw = this.grid.nx + 2;
    this.hh = this.grid.ny + 2;
    this.heatCanvas = document.createElement('canvas');
    this.heatCanvas.width = this.hw;
    this.heatCanvas.height = this.hh;
    this.heatCtx = this.heatCanvas.getContext('2d');
    this.heatImage = this.heatCtx.createImageData(this.hw, this.hh);

    this.resize();
  }

  // CSS pixels are set by the stylesheet; the backing store follows DPR.
  resize() {
    const cssW = this.canvas.clientWidth || 720;
    const cssH = Math.round((cssW * (PITCH_Y + 2 * PAD)) / (PITCH_X + 2 * PAD));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.canvas.style.height = cssH + 'px';
    this.scale = (cssW * dpr) / (PITCH_X + 2 * PAD);
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.setTransform(this.scale, 0, 0, this.scale, PAD * this.scale, PAD * this.scale);
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
    this.hair = 1 / this.scale; // one device pixel, in metres
  }

  clear() {
    const ctx = this.ctx;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = C.paper;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.restore();
  }

  // heat is uint8[nx*ny] indexed i*ny + j, i along x.
  drawHeat(heat) {
    const { nx, ny, cell, x0, y0 } = this.grid;
    const px = this.heatImage.data;
    const [r, g, b] = C.heat;
    px.fill(0);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const v = heat[i * ny + j] / 255;
        // ImageData is row-major in y, and the padding shifts both axes by 1.
        const o = ((j + 1) * this.hw + (i + 1)) * 4;
        px[o] = r;
        px[o + 1] = g;
        px[o + 2] = b;
        // The field is a backdrop: it has to read as a wash the players sit
        // on top of, so the peak is capped well below opaque. The gamma keeps
        // low-control grass faintly visible instead of clipping it to paper.
        px[o + 3] = Math.round(255 * Math.pow(v, 0.9) * 0.58);
      }
    }
    this.heatCtx.putImageData(this.heatImage, 0, 0);
    // Destination is the grid's extent grown by the one padding cell, so
    // every real cell centre still lands on its own cell centre and the
    // browser's bilinear filter does all the interpolation.
    this.ctx.drawImage(this.heatCanvas,
      x0 - cell, y0 - cell, this.hw * cell, this.hh * cell);
  }

  drawPitch() {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = C.line;
    ctx.lineWidth = this.hair;

    ctx.strokeRect(0, 0, PITCH_X, PITCH_Y);

    ctx.beginPath();
    ctx.moveTo(PITCH_X / 2, 0);
    ctx.lineTo(PITCH_X / 2, PITCH_Y);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(PITCH_X / 2, PITCH_Y / 2, 9.15, 0, Math.PI * 2);
    ctx.stroke();

    // Penalty areas (40.32 x 16.5) and six-yard boxes (18.32 x 5.5).
    const boxes = [
      [0, (PITCH_Y - 40.32) / 2, 16.5, 40.32],
      [PITCH_X - 16.5, (PITCH_Y - 40.32) / 2, 16.5, 40.32],
      [0, (PITCH_Y - 18.32) / 2, 5.5, 18.32],
      [PITCH_X - 5.5, (PITCH_Y - 18.32) / 2, 5.5, 18.32],
    ];
    for (const [x, y, w, h] of boxes) ctx.strokeRect(x, y, w, h);
    ctx.restore();
  }

  // The task: get the ball inside this disc while holding pitch control.
  drawZone() {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = C.zone;
    ctx.lineWidth = this.hair * 1.5;
    ctx.setLineDash([1.2, 1.2]);
    ctx.beginPath();
    ctx.arc(this.zone.x, this.zone.y, this.zone.r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  drawPlayers(pos, holder) {
    const ctx = this.ctx;
    const nAtt = this.manifest.n_attackers || 10;

    // Defenders first so attackers sit on top where they overlap.
    for (let p = this.manifest.n_players - 1; p >= 0; p--) {
      const x = pos[p * 2];
      const y = pos[p * 2 + 1];
      const attacker = p < nAtt;
      const keeper = p === nAtt; // row 10 is the goalkeeper
      const r = attacker ? 0.95 : 0.85;

      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      if (keeper) {
        ctx.fillStyle = C.paper;
        ctx.fill();
        ctx.lineWidth = this.hair * 1.5;
        ctx.strokeStyle = C.keeper;
        ctx.stroke();
      } else {
        ctx.fillStyle = attacker ? C.attacker : C.defender;
        ctx.fill();
      }

      // The carrier gets a hairline ring rather than a different colour.
      if (p === holder) {
        ctx.beginPath();
        ctx.arc(x, y, r + 1.1, 0, Math.PI * 2);
        ctx.lineWidth = this.hair * 1.5;
        ctx.strokeStyle = attacker ? C.attacker : C.zone;
        ctx.stroke();
      }
    }
  }

  drawBall(ball, state) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(ball[0], ball[1], 0.55, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.lineWidth = this.hair * 1.5;
    // In flight reads darker; held or loose sits on the hairline grey.
    ctx.strokeStyle = state === 1 ? C.ink : C.zone;
    ctx.stroke();
  }

  drawFrame(ep, t) {
    this.clear();
    this.drawHeat(heatAt(ep, t));
    this.drawPitch();
    this.drawZone();
    this.drawPlayers(playersAt(ep, t), ep.holder[t]);
    this.drawBall(ballAt(ep, t), ep.state[t]);
  }

  // First paint before any recording has arrived.
  drawEmpty() {
    this.clear();
    this.drawPitch();
    this.drawZone();
  }
}
