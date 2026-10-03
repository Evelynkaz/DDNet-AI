import { closeSync, fstatSync, openSync, readSync, writeFileSync } from "node:fs";
import type { RecurrentPolicy } from "../nn/gru.ts";
import { ACTION_SIZE } from "../env/action.ts";
import { OBS_SIZE } from "../env/obs.ts";
import { WEAPON_HAMMER } from "../core/types.ts";
import type { PlayerInput } from "../core/types.ts";

const MAGIC = "DDAIHUM1";

export type HumanSequence = { player: string; slot: number; start: number; length: number; tick: number };

export type HumanPlayerInfo = { key: string; name: string; slot: number; samples: number; sequences: number };

export type HumanDatasetHeader = {
  version: 1;
  obsSize: number;
  actionSize: number;
  samples: number;
  sequences: HumanSequence[];
  players: HumanPlayerInfo[];
  source: { demo: string; map: string; timestamp: string; seqLength: number };
  built: string;
};

export type HumanDataset = {
  header: HumanDatasetHeader;
  obsSize: number;
  actionSize: number;
  samples: number;
  sequences: HumanSequence[];
  obs: Float32Array;
  targets: Float32Array;
};

export const W_DIRECTION = 1;
export const W_BINARY = 0.5;
export const W_WEAPON = 0.5;
export const W_AIM = 2;
export const W_TOTAL = W_DIRECTION + 3 * W_BINARY + W_WEAPON + W_AIM;

export function encodeHumanTarget(input: PlayerInput, fired: boolean, activeWeapon: number, target: Float32Array | Float64Array, offset = 0): void {
  target[offset + 0] = input.direction === -1 ? 1 : -1;
  target[offset + 1] = input.direction === 0 ? 1 : -1;
  target[offset + 2] = input.direction === 1 ? 1 : -1;
  target[offset + 3] = input.jump ? 1 : -1;
  target[offset + 4] = input.hook ? 1 : -1;
  target[offset + 5] = fired ? 1 : -1;
  target[offset + 6] = activeWeapon === WEAPON_HAMMER ? 1 : -1;
  target[offset + 7] = activeWeapon === WEAPON_HAMMER ? -1 : 1;
  const len = Math.sqrt(input.targetX * input.targetX + input.targetY * input.targetY);
  if (len > 1e-6) {
    target[offset + 8] = input.targetX / len;
    target[offset + 9] = input.targetY / len;
  } else {
    target[offset + 8] = 1;
    target[offset + 9] = 0;
  }
}

export function agreement(action: ArrayLike<number>, target: ArrayLike<number>, targetOffset = 0): number {
  let dirIdx = 0;
  if (action[1] > action[dirIdx]) dirIdx = 1;
  if (action[2] > action[dirIdx]) dirIdx = 2;
  let tDir = 0;
  if (target[targetOffset + 1] > target[targetOffset + tDir]) tDir = 1;
  if (target[targetOffset + 2] > target[targetOffset + tDir]) tDir = 2;
  let agree = dirIdx === tDir ? W_DIRECTION : 0;

  if (action[3] > 0 === target[targetOffset + 3] > 0) agree += W_BINARY;
  if (action[4] > 0 === target[targetOffset + 4] > 0) agree += W_BINARY;
  if (action[5] > 0 === target[targetOffset + 5] > 0) agree += W_BINARY;

  const wIdx = action[7] > action[6] ? 1 : 0;
  const tW = target[targetOffset + 7] > target[targetOffset + 6] ? 1 : 0;
  if (wIdx === tW) agree += W_WEAPON;

  const ax = action[8];
  const ay = action[9];
  const alen = Math.sqrt(ax * ax + ay * ay);
  if (alen > 1e-9) {
    const cos = (ax * target[targetOffset + 8] + ay * target[targetOffset + 9]) / alen;
    agree += W_AIM * ((cos + 1) / 2);
  }
  return agree / W_TOTAL;
}

export function writeHumanDataset(path: string, header: Omit<HumanDatasetHeader, "version">, obs: Float32Array, targets: Float32Array): void {
  if (obs.length !== header.samples * header.obsSize) throw new Error(`human dataset: obs block has ${obs.length} floats, expected ${header.samples * header.obsSize}`);
  if (targets.length !== header.samples * header.actionSize) throw new Error(`human dataset: target block has ${targets.length} floats, expected ${header.samples * header.actionSize}`);
  let json = JSON.stringify({ version: 1, ...header });
  while ((8 + 4 + Buffer.byteLength(json)) % 4 !== 0) json += " ";
  const jsonBuf = Buffer.from(json, "utf8");
  const head = Buffer.alloc(12);
  head.write(MAGIC, 0, "latin1");
  head.writeUInt32LE(jsonBuf.length, 8);
  writeFileSync(path, Buffer.concat([head, jsonBuf, Buffer.from(obs.buffer, obs.byteOffset, obs.byteLength), Buffer.from(targets.buffer, targets.byteOffset, targets.byteLength)]));
}

function readExact(fd: number, length: number, position: number): Buffer {
  const buf = Buffer.alloc(length);
  let got = 0;
  while (got < length) {
    const r = readSync(fd, buf, got, length - got, position + got);
    if (r <= 0) throw new Error(`human dataset: truncated file (wanted ${length} bytes at ${position})`);
    got += r;
  }
  return buf;
}

export function readHumanDatasetHeader(path: string): HumanDatasetHeader {
  const fd = openSync(path, "r");
  try {
    return parseHeader(fd).header;
  } finally {
    closeSync(fd);
  }
}

function parseHeader(fd: number): { header: HumanDatasetHeader; dataOffset: number } {
  const size = fstatSync(fd).size;
  if (size < 12) throw new Error("human dataset: file too short");
  const head = readExact(fd, 12, 0);
  if (head.toString("latin1", 0, 8) !== MAGIC) throw new Error(`human dataset: bad magic ${JSON.stringify(head.toString("latin1", 0, 8))}`);
  const jsonLength = head.readUInt32LE(8);
  const header = JSON.parse(readExact(fd, jsonLength, 12).toString("utf8")) as HumanDatasetHeader;
  if (header.version !== 1) throw new Error(`human dataset: unsupported version ${header.version}`);
  return { header, dataOffset: 12 + jsonLength };
}

export function loadHumanDataset(path: string): HumanDataset {
  const fd = openSync(path, "r");
  try {
    const { header, dataOffset } = parseHeader(fd);
    const obsBytes = header.samples * header.obsSize * 4;
    const targetBytes = header.samples * header.actionSize * 4;
    const size = fstatSync(fd).size;
    if (size < dataOffset + obsBytes + targetBytes) {
      throw new Error(`human dataset: file has ${size} bytes, header needs ${dataOffset + obsBytes + targetBytes}`);
    }
    const obsBuf = readExact(fd, obsBytes, dataOffset);
    const targetBuf = readExact(fd, targetBytes, dataOffset + obsBytes);
    const obs = new Float32Array(obsBuf.buffer, obsBuf.byteOffset, header.samples * header.obsSize);
    const targets = new Float32Array(targetBuf.buffer, targetBuf.byteOffset, header.samples * header.actionSize);
    for (const s of header.sequences) {
      if (s.start < 0 || s.length <= 0 || s.start + s.length > header.samples) throw new Error(`human dataset: sequence out of range (${s.start}+${s.length} of ${header.samples})`);
    }
    return { header, obsSize: header.obsSize, actionSize: header.actionSize, samples: header.samples, sequences: header.sequences, obs, targets };
  } finally {
    closeSync(fd);
  }
}

export type ActingPolicy = Pick<RecurrentPolicy, "reset" | "act">;

export function scoreHuman(policy: ActingPolicy, data: HumanDataset, seqIds: number[]): number {
  if (data.obsSize !== OBS_SIZE || data.actionSize !== ACTION_SIZE) {
    throw new Error(`human dataset: obs/action size ${data.obsSize}/${data.actionSize} does not match the policy (${OBS_SIZE}/${ACTION_SIZE})`);
  }
  const obs = new Float64Array(data.obsSize);
  let sum = 0;
  let frames = 0;
  for (const id of seqIds) {
    const seq = data.sequences[id];
    if (!seq) throw new RangeError(`human dataset: no sequence ${id} (have ${data.sequences.length})`);
    policy.reset();
    for (let i = 0; i < seq.length; i++) {
      const s = seq.start + i;
      const o = s * data.obsSize;
      for (let k = 0; k < data.obsSize; k++) obs[k] = data.obs[o + k];
      const action = policy.act(obs);
      sum += agreement(action, data.targets, s * data.actionSize);
      frames++;
    }
  }
  return frames === 0 ? 0 : sum / frames;
}

export function humanFrames(data: HumanDataset, seqIds: number[]): number {
  let n = 0;
  for (const id of seqIds) n += data.sequences[id]?.length ?? 0;
  return n;
}
