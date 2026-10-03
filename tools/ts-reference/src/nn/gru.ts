import { Rng } from "./rng.ts";
import { decodeParams, encodeParams, type ParamsJSON } from "./params.ts";

export type GruShape = {

  inputs: number;
  hidden: number;
  head: number[];
  outputs: number;
};

function headSizes(shape: GruShape): number[] {
  return [shape.hidden, ...shape.head, shape.outputs];
}

export function gruParamCount(shape: GruShape): number {
  const inWidth = shape.inputs + shape.outputs;
  const h = shape.hidden;

  let n = 3 * h * inWidth + 3 * h * h + 3 * h;
  const sizes = headSizes(shape);
  for (let i = 0; i + 1 < sizes.length; i++) n += sizes[i] * sizes[i + 1] + sizes[i + 1];
  return n;
}

export function initGruParams(shape: GruShape, rng: Rng): Float64Array {
  const p = new Float64Array(gruParamCount(shape));
  const inWidth = shape.inputs + shape.outputs;
  const h = shape.hidden;
  let o = 0;
  const fill = (count: number, fanIn: number): void => {
    const scale = 1 / Math.sqrt(fanIn);
    for (let i = 0; i < count; i++) p[o++] = rng.nextGaussian() * scale;
  };
  fill(3 * h * inWidth, inWidth);
  fill(3 * h * h, h);
  o += 3 * h;
  const sizes = headSizes(shape);
  for (let i = 0; i + 1 < sizes.length; i++) {
    fill(sizes[i] * sizes[i + 1], sizes[i]);
    o += sizes[i + 1];
  }
  return p;
}

function sigmoid(x: number): number {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

function matvecAcc(p: Float64Array, base: number, acc: Float64Array, rows: number, cols: number, v: Float64Array): void {
  let j = 0;
  for (; j + 8 <= rows; j += 8) {
    let a0 = acc[j];
    let a1 = acc[j + 1];
    let a2 = acc[j + 2];
    let a3 = acc[j + 3];
    let a4 = acc[j + 4];
    let a5 = acc[j + 5];
    let a6 = acc[j + 6];
    let a7 = acc[j + 7];
    const r0 = base + j * cols;
    const r1 = r0 + cols;
    const r2 = r1 + cols;
    const r3 = r2 + cols;
    const r4 = r3 + cols;
    const r5 = r4 + cols;
    const r6 = r5 + cols;
    const r7 = r6 + cols;
    for (let k = 0; k < cols; k++) {
      const x = v[k];
      a0 += p[r0 + k] * x;
      a1 += p[r1 + k] * x;
      a2 += p[r2 + k] * x;
      a3 += p[r3 + k] * x;
      a4 += p[r4 + k] * x;
      a5 += p[r5 + k] * x;
      a6 += p[r6 + k] * x;
      a7 += p[r7 + k] * x;
    }
    acc[j] = a0;
    acc[j + 1] = a1;
    acc[j + 2] = a2;
    acc[j + 3] = a3;
    acc[j + 4] = a4;
    acc[j + 5] = a5;
    acc[j + 6] = a6;
    acc[j + 7] = a7;
  }
  for (; j < rows; j++) {
    let a = acc[j];
    const r0 = base + j * cols;
    for (let k = 0; k < cols; k++) a += p[r0 + k] * v[k];
    acc[j] = a;
  }
}

function matvecAccSparse(p: Float64Array, base: number, acc: Float64Array, rows: number, cols: number, v: Float64Array, nz: Int32Array, nzCount: number): void {
  let j = 0;
  for (; j + 8 <= rows; j += 8) {
    let a0 = acc[j];
    let a1 = acc[j + 1];
    let a2 = acc[j + 2];
    let a3 = acc[j + 3];
    let a4 = acc[j + 4];
    let a5 = acc[j + 5];
    let a6 = acc[j + 6];
    let a7 = acc[j + 7];
    const r0 = base + j * cols;
    const r1 = r0 + cols;
    const r2 = r1 + cols;
    const r3 = r2 + cols;
    const r4 = r3 + cols;
    const r5 = r4 + cols;
    const r6 = r5 + cols;
    const r7 = r6 + cols;
    for (let q = 0; q < nzCount; q++) {
      const k = nz[q];
      const x = v[k];
      a0 += p[r0 + k] * x;
      a1 += p[r1 + k] * x;
      a2 += p[r2 + k] * x;
      a3 += p[r3 + k] * x;
      a4 += p[r4 + k] * x;
      a5 += p[r5 + k] * x;
      a6 += p[r6 + k] * x;
      a7 += p[r7 + k] * x;
    }
    acc[j] = a0;
    acc[j + 1] = a1;
    acc[j + 2] = a2;
    acc[j + 3] = a3;
    acc[j + 4] = a4;
    acc[j + 5] = a5;
    acc[j + 6] = a6;
    acc[j + 7] = a7;
  }
  for (; j < rows; j++) {
    let a = acc[j];
    const r0 = base + j * cols;
    for (let q = 0; q < nzCount; q++) {
      const k = nz[q];
      a += p[r0 + k] * v[k];
    }
    acc[j] = a;
  }
}

export class RecurrentPolicy {
  readonly shape: GruShape;
  readonly params: Float64Array;

  private readonly inWidth: number;
  private readonly sizes: number[];
  private readonly x: Float64Array;
  private readonly h: Float64Array;
  private readonly z: Float64Array;
  private readonly r: Float64Array;
  private readonly n: Float64Array;
  private readonly rh: Float64Array;
  private readonly nz: Int32Array;
  private readonly scratch: Float64Array[];
  private readonly out: Float64Array;

  private readonly oW: number;
  private readonly oU: number;
  private readonly oB: number;
  private readonly oHead: number;

  constructor(shape: GruShape, params?: Float64Array) {
    this.shape = shape;
    const expected = gruParamCount(shape);
    if (params !== undefined && params.length !== expected) {
      throw new Error(`RecurrentPolicy: expected ${expected} params, got ${params.length}`);
    }
    this.params = params ?? new Float64Array(expected);
    this.inWidth = shape.inputs + shape.outputs;
    this.sizes = headSizes(shape);
    this.x = new Float64Array(this.inWidth);
    this.h = new Float64Array(shape.hidden);
    this.z = new Float64Array(shape.hidden);
    this.r = new Float64Array(shape.hidden);
    this.n = new Float64Array(shape.hidden);
    this.rh = new Float64Array(shape.hidden);
    this.nz = new Int32Array(this.inWidth);
    this.out = new Float64Array(shape.outputs);
    this.scratch = this.sizes.map((s) => new Float64Array(s));
    this.oW = 0;
    this.oU = 3 * shape.hidden * this.inWidth;
    this.oB = this.oU + 3 * shape.hidden * shape.hidden;
    this.oHead = this.oB + 3 * shape.hidden;
  }

  reset(): void {
    this.h.fill(0);
    this.out.fill(0);
  }

  saveState(): Float64Array {
    const out = new Float64Array(this.h.length + this.out.length);
    out.set(this.h, 0);
    out.set(this.out, this.h.length);
    return out;
  }

  restoreState(state: Float64Array): void {
    const expected = this.h.length + this.out.length;
    if (state.length !== expected) {
      throw new Error(`RecurrentPolicy.restoreState: expected ${expected} values, got ${state.length}`);
    }
    this.h.set(state.subarray(0, this.h.length));
    this.out.set(state.subarray(this.h.length));
  }

  setParams(p: Float64Array): void {
    if (p.length !== this.params.length) {
      throw new Error(`RecurrentPolicy: expected ${this.params.length} params, got ${p.length}`);
    }
    this.params.set(p);
  }

  act(obs: Float64Array): Float64Array {
    const { inputs, hidden, outputs } = this.shape;
    if (obs.length !== inputs) {
      throw new Error(`RecurrentPolicy: expected ${inputs} observation values, got ${obs.length}`);
    }
    const p = this.params;
    const x = this.x;
    x.set(obs, 0);

    for (let i = 0; i < outputs; i++) x[inputs + i] = Math.tanh(this.out[i]);

    const h = this.h;
    const z = this.z;
    const r = this.r;
    const n = this.n;
    const rh = this.rh;
    const nz = this.nz;
    const inW = this.inWidth;
    const oW = this.oW;
    const oU = this.oU;
    const oB = this.oB;

    let nzCount = 0;
    for (let k = 0; k < inW; k++) if (x[k] !== 0) nz[nzCount++] = k;

    for (let j = 0; j < hidden; j++) {
      z[j] = p[oB + j];
      r[j] = p[oB + hidden + j];
      n[j] = p[oB + 2 * hidden + j];
    }
    matvecAccSparse(p, oW, z, hidden, inW, x, nz, nzCount);
    matvecAccSparse(p, oW + hidden * inW, r, hidden, inW, x, nz, nzCount);
    matvecAccSparse(p, oW + 2 * hidden * inW, n, hidden, inW, x, nz, nzCount);
    matvecAcc(p, oU, z, hidden, hidden, h);
    matvecAcc(p, oU + hidden * hidden, r, hidden, hidden, h);
    for (let j = 0; j < hidden; j++) {
      z[j] = sigmoid(z[j]);
      r[j] = sigmoid(r[j]);
      rh[j] = r[j] * h[j];
    }

    matvecAcc(p, oU + 2 * hidden * hidden, n, hidden, hidden, rh);
    for (let j = 0; j < hidden; j++) {
      n[j] = Math.tanh(n[j]);
      h[j] = (1 - z[j]) * n[j] + z[j] * h[j];
    }

    const sizes = this.sizes;
    this.scratch[0].set(h);
    let o = this.oHead;
    for (let l = 0; l + 1 < sizes.length; l++) {
      const src = this.scratch[l];
      const dst = this.scratch[l + 1];
      const inN = sizes[l];
      const outN = sizes[l + 1];
      const isLast = l + 2 === sizes.length;
      dst.fill(0);
      matvecAcc(p, o, dst, outN, inN, src);
      const bias = o + outN * inN;
      for (let j = 0; j < outN; j++) {
        const a = dst[j] + p[bias + j];
        dst[j] = isLast ? a : Math.tanh(a);
      }
      o += outN * inN + outN;
    }
    this.out.set(this.scratch[sizes.length - 1]);
    return this.out;
  }

  toJSON(): { kind: string; shape: GruShape; params: ParamsJSON } {
    return { kind: "gru", shape: this.shape, params: encodeParams(this.params) };
  }

  static fromJSON(o: unknown): RecurrentPolicy {
    const obj = o as { shape?: GruShape; params?: unknown };
    if (obj?.shape === undefined || obj.params === undefined) {
      throw new Error("RecurrentPolicy.fromJSON: missing shape or params");
    }
    return new RecurrentPolicy(obj.shape, decodeParams(obj.params));
  }
}
