import { createRequire } from "node:module";

const MAX_DECOMPRESSED = 1 << 16;

const EOF_SYMBOL = 256;
const LUTBITS = 10;
const LUTMASK = (1 << LUTBITS) - 1;

type HuffmanNode = { numbits: number; symbol: number; left: number; right: number };
type HuffmanInternals = { nodes: HuffmanNode[]; decode_lut: number[] };

export class NetDecodeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NetDecodeError";
  }
}

let patched = false;

export function patchHuffman(): boolean {
  if (patched) return true;
  try {
    const require = createRequire(import.meta.url);
    const mod = require("teeworlds/lib/huffman.js") as { Huffman?: { prototype: Record<string, unknown> } };
    const proto = mod.Huffman?.prototype;
    if (proto === undefined || typeof proto.decompress !== "function") return false;

    proto.decompress = function bounded(this: HuffmanInternals, inp: Uint8Array, size = 0): Buffer {
      const nodes = this.nodes;
      const lut = this.decode_lut;
      const eof = nodes[EOF_SYMBOL];
      const src = size === 0 ? inp : inp.subarray(0, size);
      const end = src.length;
      const out: number[] = [];
      let bits = 0;
      let bitcount = 0;
      let srcIndex = 0;
      for (;;) {
        let nodeIndex = -1;
        if (bitcount >= LUTBITS) nodeIndex = lut[bits & LUTMASK];
        while (bitcount < 24 && srcIndex !== end) {
          bits |= src[srcIndex] << bitcount;
          bits >>>= 0;
          bitcount += 8;
          srcIndex++;
        }
        if (nodeIndex === -1) nodeIndex = lut[bits & LUTMASK];
        if (nodes[nodeIndex].numbits) {
          bits >>>= nodes[nodeIndex].numbits;
          bitcount -= nodes[nodeIndex].numbits;
        } else {
          bits >>>= LUTBITS;
          bitcount -= LUTBITS;
          for (;;) {
            nodeIndex = (bits & 1) !== 0 ? nodes[nodeIndex].right : nodes[nodeIndex].left;
            bitcount -= 1;
            bits >>>= 1;
            if (nodes[nodeIndex].numbits) break;
            if (bitcount === 0) throw new NetDecodeError("ran out of bits inside a symbol");
          }
        }
        if (nodes[nodeIndex] === eof) break;
        out.push(nodes[nodeIndex].symbol);

        if (out.length > MAX_DECOMPRESSED) {
          throw new NetDecodeError(`huffman produced over ${MAX_DECOMPRESSED} bytes from a ${end}-byte packet`);
        }

        if (srcIndex === end && bitcount <= 0) throw new NetDecodeError("packet ended without an end-of-file symbol");
      }
      return Buffer.from(out);
    };
    patched = true;
    return true;
  } catch {
    return false;
  }
}

const NET_PATTERNS = [/Invalid array length/i, /No more bits, decoding error/i, /huffman produced/i, /Invalid typed array length/i];

export function isNetworkDecodeError(err: unknown): boolean {
  if (err instanceof NetDecodeError) return true;
  const message = err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err);
  if (!NET_PATTERNS.some((p) => p.test(message))) return false;

  return /teeworlds[\\/]lib[\\/]/.test(message) || err instanceof NetDecodeError;
}

export type NetGuard = { dropped: number; install(): void; uninstall(): void };

export function installNetworkGuard(onDrop: (message: string, dropped: number) => void): NetGuard {
  const guard: NetGuard = {
    dropped: 0,
    install(): void {
      process.on("uncaughtException", handler);
    },
    uninstall(): void {
      process.off("uncaughtException", handler);
    },
  };
  function handler(err: unknown): void {
    if (!isNetworkDecodeError(err)) throw err;
    guard.dropped++;
    onDrop(err instanceof Error ? err.message : String(err), guard.dropped);
  }
  return guard;
}

const NETMSG_REDIRECT = 65548;
let redirectPatched = false;

export function patchRedirect(): boolean {
  if (redirectPatched) return true;
  try {
    const require = createRequire(import.meta.url);
    const mod = require("teeworlds/lib/client.js") as { Client?: { prototype: Record<string, unknown> } };
    const unpacker = require("teeworlds/lib/MsgUnpacker.js") as { unpackInt?: (b: Buffer) => { result: number } };
    const proto = mod.Client?.prototype;
    const unpack = proto?.Unpack;
    if (proto === undefined || typeof unpack !== "function" || typeof unpacker.unpackInt !== "function") return false;
    const readInt = unpacker.unpackInt;
    proto.Unpack = function withRedirect(this: { emit: (e: string, ...a: unknown[]) => boolean }, packet: Buffer) {
      const out = (unpack as (p: Buffer) => { chunks?: { sys?: boolean; msgid?: number; raw?: Buffer }[] }).call(this, packet);
      for (const c of out?.chunks ?? []) {
        if (c.sys === true && c.msgid === NETMSG_REDIRECT && c.raw !== undefined && c.raw.length > 0) {
          const port = readInt(c.raw).result;
          if (Number.isInteger(port) && port > 0 && port < 65536) this.emit("redirect", port);
        }
      }
      return out;
    };
    redirectPatched = true;
    return true;
  } catch {
    return false;
  }
}

const ITEM_SIZES = [0, 10, 6, 5, 4, 3, 8, 4, 15, 22, 5, 17, 3, 2, 2, 2, 2, 3, 3, 3, 3];
const EVENT_NAMES: Record<number, string> = {
  13: "common",
  14: "explosion",
  15: "spawn",
  16: "hammerhit",
  17: "death",
  18: "sound_global",
  19: "sound_world",
  20: "damage_indicator",
};
const EVENT_FIRST = 13;
const EVENT_LAST = 20;

type RawDelta = { data: number[]; key: number; id: number; type_id: number; parsed: unknown };
type SnapInternals = {
  deltas: RawDelta[];
  eSnapHolder: unknown;
  held?: Map<number, Map<number, number[]>>;
  crc_errors: number;
  uuid_manager: { LookupType(id: number): unknown; RegisterName(name: string, id: number): void };
  supported_uuids: string[];
  client: { SnapshotUnpacker: { emit(name: string, parsed: unknown): void } };
  parseItem(data: number[], type: number, id: number): unknown;
  crc(): number;
};

function sameInts(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function undiff(base: readonly number[], data: number[]): number[] {
  base.forEach((a, i) => {
    if (a !== undefined && data[i] !== undefined) data[i] += a;
    else data[i] = 0;
  });
  return data;
}

let snapshotPatched = false;
let originalUnpack: ((...a: unknown[]) => unknown) | null = null;

export function originalUnpackSnapshot(): ((...a: unknown[]) => unknown) | null {
  return originalUnpack;
}

export function patchSnapshotDecoder(): boolean {
  if (snapshotPatched) return true;
  try {
    const require = createRequire(import.meta.url);
    const mod = require("teeworlds/lib/snapshot.js") as { Snapshot?: { prototype: Record<string, unknown> } };
    const unpackerMod = require("teeworlds/lib/MsgUnpacker.js") as { MsgUnpacker?: new (b: Buffer) => { unpackInt(): number } };
    const uuidMod = require("teeworlds/lib/UUIDManager.js") as { createTwMD5Hash?: (name: string) => Buffer };
    const proto = mod.Snapshot?.prototype;
    const MsgUnpacker = unpackerMod.MsgUnpacker;
    const md5 = uuidMod.createTwMD5Hash;
    if (proto === undefined || typeof proto.unpackSnapshot !== "function" || MsgUnpacker === undefined || md5 === undefined) return false;
    originalUnpack = proto.unpackSnapshot as (...a: unknown[]) => unknown;
    proto.unpackSnapshot = function linear(this: SnapInternals, snap: Buffer, deltatick: number, recvTick: number, WantedCrc: number) {
      const unpacker = new MsgUnpacker(snap);

      if (this.held === undefined || !Array.isArray(this.eSnapHolder) || (this.eSnapHolder as unknown[]).length > 0) {
        this.held = new Map();
        this.eSnapHolder = [];
      }
      const held = this.held;
      let base: Map<number, number[]> | undefined;
      if (deltatick === -1) {
        held.clear();
        this.deltas = [];
      } else {
        for (const tick of held.keys()) if (tick < deltatick) held.delete(tick);
        base = held.get(deltatick);
        if (base !== undefined && base.size === 0) base = undefined;
      }
      const at = (tick: number): Map<number, number[]> => {
        let m = held.get(tick);
        if (m === undefined) {
          m = new Map();
          held.set(tick, m);
        }
        return m;
      };
      if (snap.length === 0) {

        if (base !== undefined) {
          const into = at(recvTick);
          for (const [k, d] of base) if (!into.has(k)) into.set(k, d);
        }
        return { items: [], recvTick };
      }
      const oldDeltas = this.deltas;
      this.deltas = [];
      const numRemoved = unpacker.unpackInt();
      const numItems = unpacker.unpackInt();
      unpacker.unpackInt();
      const deleted = new Set<number>();
      for (let i = 0; i < numRemoved; i++) deleted.add(unpacker.unpackInt());
      if (base === undefined && deltatick >= 0) return { items: [], recvTick: -1 };
      let oldByKey: Map<number, RawDelta> | null = null;
      const oldOf = (key: number): RawDelta | undefined => {
        if (oldByKey === null) {
          oldByKey = new Map();
          for (const d of oldDeltas) if (!oldByKey.has(d.key)) oldByKey.set(d.key, d);
        }
        return oldByKey.get(key);
      };
      const into = at(recvTick);
      const events: { type_id: number; parsed: unknown }[] = [];
      for (let i = 0; i < numItems; i++) {
        const type_id = unpacker.unpackInt();
        const id = unpacker.unpackInt();
        const key = (type_id << 16) | id;
        const size = type_id > 0 && type_id < ITEM_SIZES.length ? ITEM_SIZES[type_id] : unpacker.unpackInt();
        let data: number[] = [];
        for (let j = 0; j < size; j++) data[j] = unpacker.unpackInt();
        let changed = false;
        if (deltatick >= 0 && base !== undefined) {
          const prev = base.get(key);
          if (prev !== undefined) {
            data = undiff(prev, data);
            changed = true;
          }
        }
        if (!into.has(key)) into.set(key, data);
        if (type_id !== 0) {
          let parsed: unknown;
          if (!changed) {
            const old = oldOf(key);
            parsed = old !== undefined && sameInts(data, old.data) ? old.parsed : this.parseItem(data, type_id, id);
          } else parsed = this.parseItem(data, type_id, id);
          this.deltas.push({ data, key, id, type_id, parsed });
          if (type_id >= EVENT_FIRST && type_id <= EVENT_LAST) events.push({ type_id, parsed });
        } else {
          this.deltas.push({ data, key, id, type_id, parsed: {} });
          if (!this.uuid_manager.LookupType(id)) {
            const bytes: number[] = [];
            for (const v of data) bytes.push((v >> 24) & 0xff, (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff);
            const target = Buffer.from(bytes);
            this.supported_uuids.forEach((name, idx) => {
              if (target.compare(md5(name)) === 0) {
                this.uuid_manager.RegisterName(name, id);
                this.supported_uuids.splice(idx, 1);
              }
            });
          }
        }
      }

      if (base !== undefined) {
        for (const [key, data] of base) {
          if (deleted.has(key) || into.has(key)) continue;
          into.set(key, data);
          const old = oldOf(key);
          if (old !== undefined && sameInts(data, old.data)) this.deltas.push(old);
          else this.deltas.push({ data, key, id: key & 0xffff, type_id: (key >> 16) & 0xffff, parsed: this.parseItem(data, (key >> 16) & 0xffff, key & 0xffff) });
        }
      }
      if (this.crc() !== WantedCrc) {
        this.deltas = oldDeltas;
        this.crc_errors++;
        if (this.crc_errors > 5) {
          recvTick = -1;
          this.crc_errors = 0;
          held.clear();
          this.deltas = [];
        } else recvTick = deltatick;
      } else if (this.crc_errors > 0) this.crc_errors--;
      for (const e of events) this.client.SnapshotUnpacker.emit(EVENT_NAMES[e.type_id], e.parsed);
      return { items: this.deltas, recvTick };
    };
    snapshotPatched = true;
    return true;
  } catch {
    return false;
  }
}
