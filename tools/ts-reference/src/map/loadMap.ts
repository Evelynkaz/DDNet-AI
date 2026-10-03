import { basename, extname } from "node:path";
import { Collision } from "../core/collision.ts";
import { DataFileReader } from "./datafile.ts";

export const MAPITEMTYPE_INFO = 1;
export const MAPITEMTYPE_LAYER = 5;
export const LAYERTYPE_TILES = 2;
export const TILESLAYERFLAG_GAME = 1 << 0;
export const TILESLAYERFLAG_TELE = 1 << 1;
export const TILESLAYERFLAG_SPEEDUP = 1 << 2;
export const TILESLAYERFLAG_FRONT = 1 << 3;
export const TILESLAYERFLAG_SWITCH = 1 << 4;
export const TILESLAYERFLAG_TUNE = 1 << 5;
export const TILE_AIR = 0;
export const TILE_SOLID = 1;
export const TILE_DEATH = 2;
export const TILE_NOHOOK = 3;
export const TILE_NOLASER = 4;
export const TILE_FREEZE = 9;
export const TILE_UNFREEZE = 11;
export const TILE_DFREEZE = 12;
export const TILE_DUNFREEZE = 13;

const VERSION_TEEWORLDS_TILESKIP = 4;
const TILE_SIZE = 4;

const OFF_LAYER_TYPE = 4;
const OFF_TILEMAP_VERSION = 12;
const OFF_WIDTH = 16;
const OFF_HEIGHT = 20;
const OFF_FLAGS = 24;
const OFF_DATA = 56;

const OFF_TELE = 72;
const OFF_SPEEDUP = 76;

const OFF_TELE_V2 = 60;
const OFF_SPEEDUP_V2 = 64;
const OFF_FRONT = 80;

const OFF_FRONT_V2 = 17 * 4;
const MIN_DDRACE_TILEMAP_SIZE = OFF_TELE + 4;
const MIN_SPEEDUP_TILEMAP_SIZE = OFF_SPEEDUP + 4;
const MIN_FRONT_TILEMAP_SIZE = OFF_FRONT + 4;

const OFF_INFO_SETTINGS = 20;
const MIN_INFO_SETTINGS_SIZE = OFF_INFO_SETTINGS + 4;

const SPEEDUP_TILE_SIZE = 6;

const TELE_TILE_SIZE = 2;
const MIN_TILEMAP_SIZE = OFF_DATA + 4;

export type TeleLayer = {

  numbers: Uint8Array;
  types: Uint8Array;
};

export type SpeedupLayer = {
  force: Uint8Array;
  maxSpeed: Uint8Array;
  angle: Int16Array;
};

export type LoadedMap = {
  collision: Collision;
  width: number;
  height: number;
  name: string;
  tele?: TeleLayer;
  speedup?: SpeedupLayer;

  settings: MapSettings;

  layers: { tele: boolean; speedup: boolean; front: boolean; switch: boolean; tune: boolean };
};

export type MapSettings = { noWeakHook: boolean };

export function readMapSettings(df: DataFileReader): MapSettings {
  const settings: MapSettings = { noWeakHook: false };
  for (const item of df.findItems(MAPITEMTYPE_INFO)) {
    if (item.id !== 0) continue;
    if (item.sizeBytes < MIN_INFO_SETTINGS_SIZE) break;
    const index = item.data.getInt32(OFF_INFO_SETTINGS, true);
    if (!(index > -1)) break;
    let raw: Buffer;
    try {
      raw = df.getData(index);
    } catch {
      break;
    }
    for (const line of raw.toString("latin1").split("\0")) {
      for (const command of line.split(";")) {
        const words = command.trim().split(/\s+/);
        if (words[0] !== "sv_no_weak_hook" || words.length < 2) continue;

        const value = Number.parseInt(words[1], 10);
        settings.noWeakHook = Number.isFinite(value) && Math.min(1, Math.max(0, value)) === 1;
      }
    }
    break;
  }
  return settings;
}

export function loadMapCollision(path: string): LoadedMap {
  const df = DataFileReader.open(path);
  const name = basename(path, extname(path));

  let game: { width: number; height: number; version: number; dataIndex: number } | undefined;
  let teleDataIndex = -1;
  let speedupDataIndex = -1;
  let frontDataIndex = -1;
  const layers = { tele: false, speedup: false, front: false, switch: false, tune: false };
  for (const layer of df.findItems(MAPITEMTYPE_LAYER)) {
    if (layer.sizeBytes < MIN_TILEMAP_SIZE) continue;
    if (layer.data.getInt32(OFF_LAYER_TYPE, true) !== LAYERTYPE_TILES) continue;
    const flags = layer.data.getInt32(OFF_FLAGS, true);
    if ((flags & TILESLAYERFLAG_TELE) !== 0) {
      layers.tele = true;

      const off = layer.data.getInt32(OFF_TILEMAP_VERSION, true) <= 2 ? OFF_TELE_V2 : OFF_TELE;
      if (layer.sizeBytes >= (off === OFF_TELE_V2 ? off + 4 : MIN_DDRACE_TILEMAP_SIZE)) teleDataIndex = layer.data.getInt32(off, true);
    }
    if ((flags & TILESLAYERFLAG_SPEEDUP) !== 0) {
      layers.speedup = true;
      const off = layer.data.getInt32(OFF_TILEMAP_VERSION, true) <= 2 ? OFF_SPEEDUP_V2 : OFF_SPEEDUP;
      if (layer.sizeBytes >= (off === OFF_SPEEDUP_V2 ? off + 4 : MIN_SPEEDUP_TILEMAP_SIZE)) speedupDataIndex = layer.data.getInt32(off, true);
    }
    if ((flags & TILESLAYERFLAG_FRONT) !== 0) {
      layers.front = true;
      const version = layer.data.getInt32(OFF_TILEMAP_VERSION, true);
      const off = version <= 2 ? OFF_FRONT_V2 : OFF_FRONT;
      if (layer.sizeBytes >= (version <= 2 ? OFF_FRONT_V2 + 4 : MIN_FRONT_TILEMAP_SIZE)) frontDataIndex = layer.data.getInt32(off, true);
    }
    if ((flags & TILESLAYERFLAG_SWITCH) !== 0) layers.switch = true;
    if ((flags & TILESLAYERFLAG_TUNE) !== 0) layers.tune = true;
    if ((flags & TILESLAYERFLAG_GAME) === 0) continue;
    if (game !== undefined) continue;
    game = {
      width: layer.data.getInt32(OFF_WIDTH, true),
      height: layer.data.getInt32(OFF_HEIGHT, true),
      version: layer.data.getInt32(OFF_TILEMAP_VERSION, true),
      dataIndex: layer.data.getInt32(OFF_DATA, true),
    };
  }
  if (game === undefined) {
    throw new Error(`map '${name}': no tiles layer with TILESLAYERFLAG_GAME found`);
  }

  const { width, height } = game;
  if (width <= 0 || height <= 0) {
    throw new Error(`map '${name}': invalid game layer size ${width}x${height}`);
  }
  if (width > 100000 || height > 100000 || width * height > 50_000_000) {
    throw new Error(`map '${name}': game layer size ${width}x${height} exceeds the maximum allowed`);
  }
  const count = width * height;
  const blob = df.getData(game.dataIndex);
  const tiles = new Uint8Array(count);

  const tileFlags = new Uint8Array(count);

  if (game.version >= VERSION_TEEWORLDS_TILESKIP) {

    const srcCount = Math.floor(blob.length / TILE_SIZE);
    let dst = 0;
    let src = 0;
    while (dst < count && src < srcCount) {
      const index = blob[src * TILE_SIZE];
      const flags = blob[src * TILE_SIZE + 1];
      const skip = blob[src * TILE_SIZE + 2];
      for (let c = 0; c <= skip && dst < count; c++) {
        tileFlags[dst] = flags;
        tiles[dst++] = index;
      }
      src++;
    }
    if (dst < count) {
      throw new Error(`map '${name}': game layer tile stream ended early (${dst} of ${count} tiles)`);
    }
  } else {
    if (blob.length < count * TILE_SIZE) {
      throw new Error(`map '${name}': game layer data too small (${blob.length} bytes for ${width}x${height})`);
    }
    for (let i = 0; i < count; i++) {
      tiles[i] = blob[i * TILE_SIZE];
      tileFlags[i] = blob[i * TILE_SIZE + 1];
    }
  }

  let tele: TeleLayer | undefined;
  if (teleDataIndex >= 0) {
    const raw = df.getData(teleDataIndex);

    if (raw.length >= count * TELE_TILE_SIZE) {
      const numbers = new Uint8Array(count);
      const types = new Uint8Array(count);
      for (let i = 0; i < count; i++) {
        numbers[i] = raw[i * TELE_TILE_SIZE];
        types[i] = raw[i * TELE_TILE_SIZE + 1];
      }
      tele = { numbers, types };
    }
  }

  let speedup: SpeedupLayer | undefined;
  if (speedupDataIndex >= 0) {
    const raw = df.getData(speedupDataIndex);
    if (raw.length >= count * SPEEDUP_TILE_SIZE) {
      const force = new Uint8Array(count);
      const maxSpeed = new Uint8Array(count);
      const angle = new Int16Array(count);
      const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
      for (let i = 0; i < count; i++) {
        force[i] = raw[i * SPEEDUP_TILE_SIZE];
        maxSpeed[i] = raw[i * SPEEDUP_TILE_SIZE + 1];
        angle[i] = view.getInt16(i * SPEEDUP_TILE_SIZE + 4, true);
      }
      speedup = { force, maxSpeed, angle };
    }
  }

  let front: { index: Uint8Array; flags: Uint8Array } | undefined;
  if (frontDataIndex >= 0) {
    const raw = df.getData(frontDataIndex);
    if (raw.length >= count * TILE_SIZE) {
      const index = new Uint8Array(count);
      const flags = new Uint8Array(count);
      for (let i = 0; i < count; i++) {
        index[i] = raw[i * TILE_SIZE];
        flags[i] = raw[i * TILE_SIZE + 1];
      }
      front = { index, flags };
    }
  }

  const settings = readMapSettings(df);
  const collision = new Collision(width, height, tiles, tele, speedup, { tileFlags, front, noWeakHook: settings.noWeakHook });
  return { collision, width, height, name, tele, speedup, settings, layers };
}

export function tileCounts(map: LoadedMap): Map<number, number> {
  const counts = new Map<number, number>();
  for (const t of map.collision.tiles) {
    counts.set(t, (counts.get(t) ?? 0) + 1);
  }
  return counts;
}
