import { createHash } from "node:crypto";

export const MAX_TYPE = 0x7fff;
export const MAX_ID = 0xffff;
export const MAX_ITEMS = 1024;
export const MAX_SIZE = 64 * 1024;
export const OFFSET_UUID_TYPE = 0x4000;
const MAX_NETOBJSIZES = 64;

export const NETOBJTYPE_EX = 0;
export const NETOBJTYPE_PLAYERINPUT = 1;
export const NETOBJTYPE_PROJECTILE = 2;
export const NETOBJTYPE_LASER = 3;
export const NETOBJTYPE_PICKUP = 4;
export const NETOBJTYPE_FLAG = 5;
export const NETOBJTYPE_GAMEINFO = 6;
export const NETOBJTYPE_GAMEDATA = 7;
export const NETOBJTYPE_CHARACTERCORE = 8;
export const NETOBJTYPE_CHARACTER = 9;
export const NETOBJTYPE_PLAYERINFO = 10;
export const NETOBJTYPE_CLIENTINFO = 11;
export const NETOBJTYPE_SPECTATORINFO = 12;
export const NETEVENTTYPE_COMMON = 13;
export const NETEVENTTYPE_EXPLOSION = 14;
export const NETEVENTTYPE_SPAWN = 15;
export const NETEVENTTYPE_HAMMERHIT = 16;
export const NETEVENTTYPE_DEATH = 17;
export const NETEVENTTYPE_SOUNDGLOBAL = 18;
export const NETEVENTTYPE_SOUNDWORLD = 19;
export const NETEVENTTYPE_DAMAGEIND = 20;
export const NUM_NETOBJTYPES = 21;

export const NETOBJ_INT_SIZES: readonly number[] = [0, 10, 6, 5, 4, 3, 8, 4, 15, 22, 5, 17, 3, 2, 2, 2, 2, 3, 3, 3, 3];

export const OFFSET_UUID = 1 << 16;
const EX_OBJECT_NAMES: readonly string[] = [
  "my-own-object@heinrich5991.de",
  "character@netobj.ddnet.tw",
  "player@netobj.ddnet.tw",
  "gameinfo@netobj.ddnet.tw",
  "projectile@netobj.ddnet.tw",
  "laser@netobj.ddnet.tw",
  "ddnet-projectile@netobj.ddnet.tw",
  "pickup@netobj.ddnet.tw",
  "spectator-info@netobj.ddnet.org",
  "spectator-count@netobj.ddnet.org",
  "birthday@netevent.ddnet.org",
  "finish@netevent.ddnet.org",
  "my-own-event@heinrich5991.de",
  "spec-char@netobj.ddnet.tw",
  "switch-state@netobj.ddnet.tw",
  "entity-ex@netobj.ddnet.tw",
  "map-best-time@netobj.ddnet.org",
  "map-sound-world@netevent.ddnet.org",
];
export const NETOBJTYPE_MYOWNOBJECT = OFFSET_UUID + 0;
export const NETOBJTYPE_DDNETCHARACTER = OFFSET_UUID + 1;
export const NETOBJTYPE_DDNETPLAYER = OFFSET_UUID + 2;
export const NETOBJTYPE_GAMEINFOEX = OFFSET_UUID + 3;
export const NETOBJTYPE_DDRACEPROJECTILE = OFFSET_UUID + 4;
export const NETOBJTYPE_DDNETLASER = OFFSET_UUID + 5;
export const NETOBJTYPE_DDNETPROJECTILE = OFFSET_UUID + 6;
export const NETOBJTYPE_DDNETPICKUP = OFFSET_UUID + 7;
export const NETOBJTYPE_SPECCHAR = OFFSET_UUID + 13;

const TEEWORLDS_NAMESPACE = Buffer.from("e05ddaaac4e64cfbb6425d48e80c0029", "hex");
export function calculateUuid(name: string): string {
  const md5 = createHash("md5");
  md5.update(TEEWORLDS_NAMESPACE);
  md5.update(Buffer.from(name, "utf8"));
  const d = md5.digest();
  d[6] = (d[6] & 0x0f) | 0x30;
  d[8] = (d[8] & 0x3f) | 0x80;
  return d.toString("hex");
}

const uuidToType = new Map<string, number>();
const typeToName = new Map<number, string>();
EX_OBJECT_NAMES.forEach((name, i) => {
  uuidToType.set(calculateUuid(name), OFFSET_UUID + i);
  typeToName.set(OFFSET_UUID + i, name);
});

export function extendedTypeName(type: number): string | undefined {
  return typeToName.get(type);
}

export type RawItem = { key: number; type: number; id: number; fields: Int32Array };

export type RawSnapshot = { items: RawItem[]; byKey: Map<number, number> };

export function parseSnapshot(ints: Int32Array): RawSnapshot {
  const actualSize = ints.length * 4;
  if (ints.length < 2) throw new Error(`snapshot: too small (${actualSize} bytes)`);
  const dataSize = ints[0];
  const numItems = ints[1];
  if (numItems < 0 || numItems > MAX_ITEMS || dataSize < 0 || actualSize > MAX_SIZE) {
    throw new Error(`snapshot: invalid header. num_items=${numItems} data_size=${dataSize} size=${actualSize}`);
  }
  const totalSize = 8 + numItems * 4 + dataSize;
  if (totalSize !== actualSize) throw new Error(`snapshot: size mismatch. expected=${totalSize} actual=${actualSize}`);

  const dataStart = 2 + numItems;
  const items: RawItem[] = [];
  const byKey = new Map<number, number>();
  for (let i = 0; i < numItems; i++) {
    const offset = ints[2 + i];
    if (offset < 0 || offset > dataSize || offset % 4 !== 0) throw new Error(`snapshot: invalid item offset. index=${i} offset=${offset}`);
    const end = i === numItems - 1 ? dataSize : ints[2 + i + 1];
    const itemSize = end - offset - 4;
    if (itemSize < 0 || itemSize % 4 !== 0) throw new Error(`snapshot: invalid item size. index=${i} size=${itemSize}`);
    const at = dataStart + offset / 4;
    const key = ints[at];
    items.push({ key, type: key >> 16, id: key & 0xffff, fields: ints.subarray(at + 1, at + 1 + itemSize / 4) });
    byKey.set(key, i);
  }
  return { items, byKey };
}

export function emptySnapshot(): RawSnapshot {
  return { items: [], byKey: new Map() };
}

export function unpackDelta(from: RawSnapshot, ints: Int32Array): RawSnapshot {
  if (ints.length < 3) throw new Error(`snapshot delta: too small (${ints.length * 4} bytes)`);
  const numDeleted = ints[0];
  const numUpdate = ints[1];
  if (numDeleted < 0) throw new Error(`snapshot delta: negative deleted count (${numDeleted}) [-201]`);
  let p = 3;
  const deletedStart = p;
  p += numDeleted;
  if (p > ints.length) throw new Error("snapshot delta: deleted list runs past the end [-101]");

  const items: RawItem[] = [];
  const byKey = new Map<number, number>();
  for (const it of from.items) {
    let keep = true;
    for (let d = 0; d < numDeleted; d++) {
      if (ints[deletedStart + d] === it.key) {
        keep = false;
        break;
      }
    }
    if (keep) {
      byKey.set(it.key, items.length);
      items.push(it);
    }
  }

  for (let i = 0; i < numUpdate; i++) {
    if (p + 2 > ints.length) throw new Error(`snapshot delta: update ${i} of ${numUpdate} does not fit [-102]`);
    const type = ints[p++];
    if (type < 0 || type > MAX_TYPE) throw new Error(`snapshot delta: type ${type} out of range [-202]`);
    const id = ints[p++];
    if (id < 0 || id > MAX_ID) throw new Error(`snapshot delta: id ${id} out of range [-203]`);

    let size: number;
    if (type < MAX_NETOBJSIZES && NETOBJ_INT_SIZES[type]) {
      size = NETOBJ_INT_SIZES[type];
    } else {
      if (p + 1 > ints.length) throw new Error("snapshot delta: expected item size but hit the end [-103]");
      size = ints[p];
      if (size < 0 || size > 0x1fffffff) throw new Error(`snapshot delta: item size ${size} out of range [-204]`);
      p++;
    }
    if (ints.length - p < size) throw new Error(`snapshot delta: item type=${type} id=${id} size=${size} does not fit [-205]`);

    const key = (type << 16) | id;
    const out = new Int32Array(size);
    const existing = byKey.get(key);
    if (existing !== undefined) {
      if (items[existing].fields.length !== size) throw new Error(`snapshot delta: size mismatch with kept item type=${type} id=${id} [-206]`);
      items[existing] = { key, type, id, fields: out };
    } else {
      byKey.set(key, items.length);
      items.push({ key, type, id, fields: out });
    }

    const fromIndex = from.byKey.get(key);
    if (fromIndex !== undefined) {
      const past = from.items[fromIndex].fields;
      if (past.length !== size) throw new Error(`snapshot delta: size mismatch with previous item type=${type} id=${id} [-207]`);
      for (let j = 0; j < size; j++) out[j] = (past[j] + ints[p + j]) | 0;
    } else {
      for (let j = 0; j < size; j++) out[j] = ints[p + j];
    }
    p += size;
  }

  if (items.length > MAX_ITEMS) throw new Error(`snapshot delta: ${items.length} items exceed MAX_ITEMS`);
  return { items, byKey };
}

export type SnapshotItem = { type: number; id: number; fields: Int32Array; uuid?: string };

export function resolveItems(snap: RawSnapshot): SnapshotItem[] {
  const out: SnapshotItem[] = [];
  const uuidCache = new Map<number, { type: number; uuid: string }>();
  for (const it of snap.items) {
    if (it.type < OFFSET_UUID_TYPE) {
      out.push({ type: it.type, id: it.id, fields: it.fields });
      continue;
    }
    let resolved = uuidCache.get(it.type);
    if (resolved === undefined) {
      const typeItemIndex = snap.byKey.get(it.type);
      if (typeItemIndex === undefined || snap.items[typeItemIndex].fields.length < 4) {
        resolved = { type: -1, uuid: "" };
      } else {
        const f = snap.items[typeItemIndex].fields;
        const bytes = Buffer.alloc(16);
        for (let i = 0; i < 4; i++) bytes.writeUInt32BE(f[i] >>> 0, i * 4);
        const uuid = bytes.toString("hex");
        resolved = { type: uuidToType.get(uuid) ?? -1, uuid };
      }
      uuidCache.set(it.type, resolved);
    }
    out.push({ type: resolved.type, id: it.id, fields: it.fields, uuid: resolved.uuid });
  }
  return out;
}
