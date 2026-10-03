import type { Vec2 } from "./vmath.ts";
import { clamp, roundToInt, vdistance, vlength } from "./vmath.ts";
import {
  CANTMOVE_DOWN,
  CANTMOVE_LEFT,
  CANTMOVE_RIGHT,
  CANTMOVE_UP,
  CFLAG_DEATH,
  CFLAG_NOHOOK,
  CFLAG_SOLID,
  ROTATION_0,
  ROTATION_180,
  ROTATION_270,
  ROTATION_90,
  TILEFLAG_ROTATE,
  TILEFLAG_XFLIP,
  TILEFLAG_YFLIP,
  TILE_AIR,
  TILE_DEATH,
  TILE_FREEZE,
  TILE_LFREEZE,
  TILE_LUNFREEZE,
  TILE_NOHOOK,
  TILE_SOLID,
  TILE_STOP,
  TILE_STOPA,
  TILE_STOPS,
  TILE_TELECHECK,
  TILE_TELECHECKIN,
  TILE_TELECHECKINEVIL,
  TILE_TELECHECKOUT,
  TILE_TELEIN,
  TILE_TELEINEVIL,
  TILE_TELEOUT,
  TILE_TELE_LASER_DISABLE,
  TILE_THROUGH,
  TILE_THROUGH_ALL,
  TILE_THROUGH_CUT,
  TILE_THROUGH_DIR,
  TILE_UNFREEZE,
} from "./tuning.ts";

export function clampVel(moveRestriction: number, vel: Vec2): Vec2 {
  let x = vel.x;
  let y = vel.y;
  if (x > 0 && (moveRestriction & CANTMOVE_RIGHT) !== 0) x = 0;
  if (x < 0 && (moveRestriction & CANTMOVE_LEFT) !== 0) x = 0;
  if (y > 0 && (moveRestriction & CANTMOVE_DOWN) !== 0) y = 0;
  if (y < 0 && (moveRestriction & CANTMOVE_UP) !== 0) y = 0;
  return { x, y };
}

const MR_DIR_HERE = 0;
const MR_DX = [0, 1, 0, -1, 0];
const MR_DY = [0, 0, 1, 0, -1];
const MR_MASK = [0, CANTMOVE_RIGHT, CANTMOVE_DOWN, CANTMOVE_LEFT, CANTMOVE_UP];

function moveRestrictionsRaw(tile: number, flags: number): number {
  flags &= TILEFLAG_XFLIP | TILEFLAG_YFLIP | TILEFLAG_ROTATE;
  if (tile === TILE_STOP) {
    switch (flags) {
      case ROTATION_0: return CANTMOVE_DOWN;
      case ROTATION_90: return CANTMOVE_LEFT;
      case ROTATION_180: return CANTMOVE_UP;
      case ROTATION_270: return CANTMOVE_RIGHT;
      case TILEFLAG_YFLIP ^ ROTATION_0: return CANTMOVE_UP;
      case TILEFLAG_YFLIP ^ ROTATION_90: return CANTMOVE_RIGHT;
      case TILEFLAG_YFLIP ^ ROTATION_180: return CANTMOVE_DOWN;
      case TILEFLAG_YFLIP ^ ROTATION_270: return CANTMOVE_LEFT;
    }
    return 0;
  }
  if (tile === TILE_STOPS) {
    switch (flags) {
      case ROTATION_0:
      case ROTATION_180:
      case TILEFLAG_YFLIP ^ ROTATION_0:
      case TILEFLAG_YFLIP ^ ROTATION_180:
        return CANTMOVE_DOWN | CANTMOVE_UP;
      case ROTATION_90:
      case ROTATION_270:
      case TILEFLAG_YFLIP ^ ROTATION_90:
      case TILEFLAG_YFLIP ^ ROTATION_270:
        return CANTMOVE_LEFT | CANTMOVE_RIGHT;
    }
    return 0;
  }
  if (tile === TILE_STOPA) return CANTMOVE_LEFT | CANTMOVE_RIGHT | CANTMOVE_UP | CANTMOVE_DOWN;
  return 0;
}

function moveRestrictionsFor(direction: number, tile: number, flags: number): number {
  const result = moveRestrictionsRaw(tile, flags);
  if (direction === MR_DIR_HERE && tile === TILE_STOP) return result;
  return result & MR_MASK[direction];
}

function isStopper(tile: number): boolean {
  return tile === TILE_STOP || tile === TILE_STOPS || tile === TILE_STOPA;
}

function isHookThroughTile(tile: number): boolean {
  return tile === TILE_THROUGH_CUT || tile === TILE_THROUGH || tile === TILE_THROUGH_ALL || tile === TILE_THROUGH_DIR;
}

export type CollisionExtras = {

  tileFlags?: Uint8Array;

  front?: { index: Uint8Array; flags: Uint8Array };

  noWeakHook?: boolean;
};

export class Collision {
  readonly width: number;
  readonly height: number;
  readonly tiles: Uint8Array;

  readonly teleType: Uint8Array | undefined;
  readonly teleNumber: Uint8Array | undefined;

  private readonly speedupForce: Uint8Array | undefined;
  private readonly speedupMax: Uint8Array | undefined;
  private readonly speedupAngle: Int16Array | undefined;

  private readonly flags: Uint8Array;

  readonly tileFlags: Uint8Array;

  readonly frontIndex: Uint8Array | undefined;
  readonly frontFlags: Uint8Array | undefined;

  readonly noWeakHook: boolean;

  private hasHookThrough = false;
  private hasStoppers = false;

  constructor(
    width: number,
    height: number,
    tiles: Uint8Array,
    tele?: { types: Uint8Array; numbers: Uint8Array },
    speedup?: { force: Uint8Array; maxSpeed: Uint8Array; angle: Int16Array },
    extras?: CollisionExtras,
  ) {
    this.width = width;
    this.height = height;
    this.tiles = tiles;
    this.flags = new Uint8Array(tiles.length);
    this.tileFlags = extras?.tileFlags !== undefined && extras.tileFlags.length === tiles.length ? extras.tileFlags : new Uint8Array(tiles.length);
    const front = extras?.front;
    if (front !== undefined && front.index.length === tiles.length && front.flags.length === tiles.length) {
      this.frontIndex = front.index;
      this.frontFlags = front.flags;
      for (let i = 0; i < front.index.length; i++) {
        const t = front.index[i];
        if (isHookThroughTile(t)) this.hasHookThrough = true;
        if (isStopper(t)) this.hasStoppers = true;
      }
    }
    this.noWeakHook = extras?.noWeakHook ?? false;
    for (let i = 0; i < tiles.length; i++) this.setTile(i, tiles[i]);
    if (tele !== undefined && tele.types.length === tiles.length) {
      this.teleType = tele.types;
      this.teleNumber = tele.numbers;
    }
    if (speedup !== undefined && speedup.force.length === tiles.length) {
      this.speedupForce = speedup.force;
      this.speedupMax = speedup.maxSpeed;
      this.speedupAngle = speedup.angle;
    }
  }

  speedupAt(index: number): { force: number; maxSpeed: number; dirX: number; dirY: number } | null {
    const f = this.speedupForce;
    if (f === undefined || index < 0 || index >= f.length || f[index] === 0) return null;
    const a = ((this.speedupAngle?.[index] ?? 0) * Math.PI) / 180;
    return { force: f[index], maxSpeed: this.speedupMax?.[index] ?? 0, dirX: Math.cos(a), dirY: Math.sin(a) };
  }

  setTile(index: number, tile: number): void {
    if (index < 0 || index >= this.tiles.length) return;
    this.tiles[index] = tile;
    let f = 0;
    if (tile === TILE_SOLID) f |= 1;
    else if (tile === TILE_NOHOOK) f |= 1 | 2;
    else if (tile === TILE_DEATH) f |= 4;
    if (tile === TILE_FREEZE) f |= 8;
    if (tile === TILE_UNFREEZE) f |= 16;
    this.flags[index] = f;
    if (isHookThroughTile(tile)) this.hasHookThrough = true;
    if (isStopper(tile)) this.hasStoppers = true;
  }

  teleAt(x: number, y: number): { type: number; number: number } {
    const t = this.teleType;
    if (t === undefined) return { type: 0, number: 0 };
    const tx = Math.min(this.width - 1, Math.max(0, Math.trunc(roundToInt(x) / 32)));
    const ty = Math.min(this.height - 1, Math.max(0, Math.trunc(roundToInt(y) / 32)));
    const i = ty * this.width + tx;
    return { type: t[i], number: this.teleNumber === undefined ? 0 : this.teleNumber[i] };
  }

  hasTele(): boolean {
    return this.teleType !== undefined;
  }

  teleTypeAtIndex(index: number): number {
    const t = this.teleType;
    return t === undefined || index < 0 || index >= t.length ? 0 : t[index];
  }

  teleNumberAtIndex(index: number): number {
    const n = this.teleNumber;
    return n === undefined || index < 0 || index >= n.length ? 0 : n[index];
  }

  teleOutsFor(number: number): Vec2[] {
    if (this.teleOuts === undefined) {
      this.teleOuts = this.collectTele(TILE_TELEOUT);
      this.teleCheckOuts = this.collectTele(TILE_TELECHECKOUT);
    }
    return this.teleOuts.get(number) ?? [];
  }

  teleCheckOutsFor(number: number): Vec2[] {
    if (this.teleCheckOuts === undefined) this.teleOutsFor(0);
    return this.teleCheckOuts?.get(number) ?? [];
  }

  private collectTele(type: number): Map<number, Vec2[]> {
    const outs = new Map<number, Vec2[]>();
    const t = this.teleType;
    const n = this.teleNumber;
    if (t !== undefined && n !== undefined) {
      for (let i = 0; i < t.length; i++) {
        if (t[i] !== type || n[i] === 0) continue;
        const list = outs.get(n[i]) ?? [];
        list.push({ x: (i % this.width) * 32 + 16, y: Math.trunc(i / this.width) * 32 + 16 });
        outs.set(n[i], list);
      }
    }
    return outs;
  }

  private teleCheckOuts: Map<number, Vec2[]> | undefined;
  private teleOuts: Map<number, Vec2[]> | undefined;

  getTileIndex(x: number, y: number): number {
    return this.tiles[this.indexAt(x, y)] ?? TILE_AIR;
  }

  private indexAt(x: number, y: number): number {
    const ix = x > 0 ? Math.trunc(x + 0.5) : Math.trunc(x - 0.5);
    const iy = y > 0 ? Math.trunc(y + 0.5) : Math.trunc(y - 0.5);

    let nx = ix >> 5;
    let ny = iy >> 5;
    if (nx < 0) nx = 0;
    else if (nx > this.width - 1) nx = this.width - 1;
    if (ny < 0) ny = 0;
    else if (ny > this.height - 1) ny = this.height - 1;
    return ny * this.width + nx;
  }

  getCollisionAt(x: number, y: number): number {
    const f = this.flags[this.indexAt(x, y)];
    return ((f & 1) === 0 ? 0 : CFLAG_SOLID) | ((f & 2) === 0 ? 0 : CFLAG_NOHOOK) | ((f & 4) === 0 ? 0 : CFLAG_DEATH);
  }

  private checkPoint(x: number, y: number): boolean {
    return (this.flags[this.indexAt(x, y)] & 1) !== 0;
  }

  isSolid(x: number, y: number): boolean {
    return (this.flags[this.indexAt(x, y)] & 1) !== 0;
  }

  isDeath(x: number, y: number): boolean {
    return (this.flags[this.indexAt(x, y)] & 4) !== 0;
  }

  isFreeze(x: number, y: number): boolean {
    return (this.flags[this.indexAt(x, y)] & 8) !== 0;
  }

  isUnFreeze(x: number, y: number): boolean {
    return (this.flags[this.indexAt(x, y)] & 16) !== 0;
  }

  isNoHook(x: number, y: number): boolean {
    return (this.flags[this.indexAt(x, y)] & 2) !== 0;
  }

  testBox(pos: Vec2, size: Vec2): boolean {
    return this.testBoxAt(pos.x, pos.y, size.x * 0.5, size.y * 0.5);
  }

  private testBoxAt(x: number, y: number, halfX: number, halfY: number): boolean {
    if (this.checkPoint(x - halfX, y - halfY)) return true;
    if (this.checkPoint(x + halfX, y - halfY)) return true;
    if (this.checkPoint(x - halfX, y + halfY)) return true;
    if (this.checkPoint(x + halfX, y + halfY)) return true;
    return false;
  }

  moveBox(inoutPos: Vec2, inoutVel: Vec2, size: Vec2, elasticity: Vec2): void {
    let posX = inoutPos.x;
    let posY = inoutPos.y;
    let velX = inoutVel.x;
    let velY = inoutVel.y;
    const halfX = size.x * 0.5;
    const halfY = size.y * 0.5;

    const distance = Math.sqrt(velX * velX + velY * velY);
    const max = Math.trunc(distance);

    if (distance > 0.00001) {
      const fraction = 1.0 / (max + 1);
      const elasticityX = clamp(elasticity.x, -1.0, 1.0);
      const elasticityY = clamp(elasticity.y, -1.0, 1.0);

      for (let i = 0; i <= max; i++) {

        if (velX === 0 && velY === 0) {
          break;
        }

        let newX = posX + velX * fraction;
        let newY = posY + velY * fraction;

        if (newX === posX && newY === posY) {
          break;
        }

        if (this.testBoxAt(newX, newY, halfX, halfY)) {
          let hits = 0;

          if (this.testBoxAt(posX, newY, halfX, halfY)) {
            newY = posY;
            velY *= -elasticityY;
            hits++;
          }

          if (this.testBoxAt(newX, posY, halfX, halfY)) {
            newX = posX;
            velX *= -elasticityX;
            hits++;
          }

          if (hits === 0) {
            newY = posY;
            velY *= -elasticityY;
            newX = posX;
            velX *= -elasticityX;
          }
        }

        posX = newX;
        posY = newY;
      }
    }

    inoutPos.x = posX;
    inoutPos.y = posY;
    inoutVel.x = velX;
    inoutVel.y = velY;
  }

  movePoint(inoutPos: Vec2, inoutVel: Vec2, elasticity: number, bounces: { count: number } | null): void {
    if (bounces) bounces.count = 0;

    const pos: Vec2 = { x: inoutPos.x, y: inoutPos.y };
    const vel: Vec2 = { x: inoutVel.x, y: inoutVel.y };
    if (this.checkPoint(pos.x + vel.x, pos.y + vel.y)) {
      let affected = 0;
      if (this.checkPoint(pos.x + vel.x, pos.y)) {
        inoutVel.x *= -elasticity;
        if (bounces) bounces.count++;
        affected++;
      }

      if (this.checkPoint(pos.x, pos.y + vel.y)) {
        inoutVel.y *= -elasticity;
        if (bounces) bounces.count++;
        affected++;
      }

      if (affected === 0) {
        inoutVel.x *= -elasticity;
        inoutVel.y *= -elasticity;
      }
    } else {
      inoutPos.x = pos.x + vel.x;
      inoutPos.y = pos.y + vel.y;
    }
  }

  tileExists(index: number): boolean {
    if (index < 0) return false;
    const t = this.tiles[index];
    if ((t >= TILE_FREEZE && t <= TILE_TELE_LASER_DISABLE) || (t >= TILE_LFREEZE && t <= TILE_LUNFREEZE)) return true;

    const fi = this.frontIndex;
    if (fi !== undefined) {
      const ft = fi[index];
      if ((ft >= TILE_FREEZE && ft <= TILE_TELE_LASER_DISABLE) || (ft >= TILE_LFREEZE && ft <= TILE_LUNFREEZE)) return true;
    }

    const ty = this.teleTypeAtIndex(index);
    if (ty === TILE_TELEIN || ty === TILE_TELEINEVIL || ty === TILE_TELECHECKINEVIL || ty === TILE_TELECHECK || ty === TILE_TELECHECKIN) return true;

    if (this.speedupForce !== undefined && index < this.speedupForce.length && this.speedupForce[index] !== 0) return true;
    return this.hasStoppers && this.tileExistsNext(index);
  }

  private tileExistsNext(index: number): boolean {
    const n = this.tiles.length;
    const w = this.width;
    const left = index - 1 > 0 ? index - 1 : index;
    const right = index + 1 < n ? index + 1 : index;
    const below = index + w < n ? index + w : index;
    const above = index - w > 0 ? index - w : index;
    const layer = (idx: Uint8Array, fl: Uint8Array): boolean => {
      if ((idx[right] === TILE_STOP && fl[right] === ROTATION_270) || (idx[left] === TILE_STOP && fl[left] === ROTATION_90)) return true;
      if ((idx[below] === TILE_STOP && fl[below] === ROTATION_0) || (idx[above] === TILE_STOP && fl[above] === ROTATION_180)) return true;
      if (idx[right] === TILE_STOPA || idx[left] === TILE_STOPA || idx[right] === TILE_STOPS || idx[left] === TILE_STOPS) return true;
      if (idx[below] === TILE_STOPA || idx[above] === TILE_STOPA || idx[below] === TILE_STOPS || idx[above] === TILE_STOPS) return true;
      return false;
    };
    if (layer(this.tiles, this.tileFlags)) return true;
    return this.frontIndex !== undefined && this.frontFlags !== undefined && layer(this.frontIndex, this.frontFlags);
  }

  getMoveRestrictions(pos: Vec2, distance = 18, overrideCenterIndex = -1): number {
    if (!this.hasStoppers) return 0;
    let restrictions = 0;
    const fi = this.frontIndex;
    const ff = this.frontFlags;
    for (let d = 0; d < 5; d++) {
      let index = this.indexAt(pos.x + MR_DX[d] * distance, pos.y + MR_DY[d] * distance);
      if (d === MR_DIR_HERE && overrideCenterIndex >= 0) index = overrideCenterIndex;
      restrictions |= moveRestrictionsFor(d, this.tiles[index], this.tileFlags[index]);
      if (fi !== undefined && ff !== undefined) restrictions |= moveRestrictionsFor(d, fi[index], ff[index]);
    }
    return restrictions;
  }

  getMapIndex(pos: Vec2): number {
    const nx = clamp(Math.trunc(Math.trunc(pos.x) / 32), 0, this.width - 1);
    const ny = clamp(Math.trunc(Math.trunc(pos.y) / 32), 0, this.height - 1);
    const index = ny * this.width + nx;
    return this.tileExists(index) ? index : -1;
  }

  getMapIndices(prevPos: Vec2, pos: Vec2, out: number[]): number[] {
    out.length = 0;
    const d = vdistance(prevPos, pos);
    const end = Math.trunc(d + 1);
    if (d === 0) {
      const nx = clamp(Math.trunc(Math.trunc(pos.x) / 32), 0, this.width - 1);
      const ny = clamp(Math.trunc(Math.trunc(pos.y) / 32), 0, this.height - 1);
      const index = ny * this.width + nx;
      if (this.tileExists(index)) out.push(index);
      return out;
    }
    let lastIndex = 0;
    for (let i = 0; i < end; i++) {
      const a = i / d;
      const tx = prevPos.x + (pos.x - prevPos.x) * a;
      const ty = prevPos.y + (pos.y - prevPos.y) * a;
      const nx = clamp(Math.trunc(Math.trunc(tx) / 32), 0, this.width - 1);
      const ny = clamp(Math.trunc(Math.trunc(ty) / 32), 0, this.height - 1);
      const index = ny * this.width + nx;
      if (lastIndex !== index && this.tileExists(index)) {
        out.push(index);
        lastIndex = index;
      }
    }
    return out;
  }

  intersectLineHook(pos0: Vec2, pos1: Vec2): { collision: number; outPos: Vec2; outBeforePos: Vec2 } {
    if (!this.hasHookThrough) return this.intersectLine(pos0, pos1);

    const tx = pos0.x - pos1.x;
    const ty = pos0.y - pos1.y;
    let dx = 0;
    let dy = 0;
    if (Math.abs(tx) > Math.abs(ty)) dx = tx < 0 ? -32 : 32;
    else dy = ty < 0 ? -32 : 32;
    const distance = vdistance(pos0, pos1);
    const end = Math.trunc(distance + 1);

    const x0 = pos0.x;
    const y0 = pos0.y;
    const rx = pos1.x - x0;
    const ry = pos1.y - y0;
    let lastX = x0;
    let lastY = y0;
    for (let i = 0; i <= end; i++) {
      const a = i / end;
      const px = x0 + rx * a;
      const py = y0 + ry * a;
      const ix = roundToInt(px);
      const iy = roundToInt(py);

      let hit = 0;
      if (this.checkPoint(ix, iy)) {
        if (!this.isThrough(ix, iy, dx, dy, pos0, pos1)) hit = this.getCollisionAt(ix, iy);
      } else if (this.isHookBlocker(ix, iy, pos0, pos1)) {
        hit = CFLAG_NOHOOK;
      }
      if (hit !== 0) return { collision: hit, outPos: { x: px, y: py }, outBeforePos: { x: lastX, y: lastY } };

      lastX = px;
      lastY = py;
    }
    return {
      collision: 0,
      outPos: { x: pos1.x, y: pos1.y },
      outBeforePos: { x: pos1.x, y: pos1.y },
    };
  }

  private isThrough(x: number, y: number, offsetX: number, offsetY: number, pos0: Vec2, pos1: Vec2): boolean {
    const index = this.indexAt(x, y);
    const fi = this.frontIndex;
    const ff = this.frontFlags;
    if (fi !== undefined && ff !== undefined) {
      const t = fi[index];
      if (t === TILE_THROUGH_ALL || t === TILE_THROUGH_CUT) return true;
      if (t === TILE_THROUGH_DIR) {
        const f = ff[index];
        if ((f === ROTATION_0 && pos0.y > pos1.y) || (f === ROTATION_90 && pos0.x < pos1.x) || (f === ROTATION_180 && pos0.y < pos1.y) || (f === ROTATION_270 && pos0.x > pos1.x)) return true;
      }
    }
    const offsetIndex = this.indexAt(x + offsetX, y + offsetY);
    return this.tiles[offsetIndex] === TILE_THROUGH || (fi !== undefined && fi[offsetIndex] === TILE_THROUGH);
  }

  private isHookBlocker(x: number, y: number, pos0: Vec2, pos1: Vec2): boolean {
    const index = this.indexAt(x, y);
    const blocks = (t: number, f: number): boolean => {
      if (t === TILE_THROUGH_ALL) return true;
      return t === TILE_THROUGH_DIR && ((f === ROTATION_0 && pos0.y < pos1.y) || (f === ROTATION_90 && pos0.x > pos1.x) || (f === ROTATION_180 && pos0.y > pos1.y) || (f === ROTATION_270 && pos0.x < pos1.x));
    };
    if (blocks(this.tiles[index], this.tileFlags[index])) return true;
    return this.frontIndex !== undefined && this.frontFlags !== undefined && blocks(this.frontIndex[index], this.frontFlags[index]);
  }

  intersectLine(pos0: Vec2, pos1: Vec2): { collision: number; outPos: Vec2; outBeforePos: Vec2 } {
    const distance = vdistance(pos0, pos1);
    const end = Math.trunc(distance + 1);
    const x0 = pos0.x;
    const y0 = pos0.y;
    const dx = pos1.x - x0;
    const dy = pos1.y - y0;
    let lastX = x0;
    let lastY = y0;
    for (let i = 0; i <= end; i++) {
      const a = i / end;
      const px = x0 + dx * a;
      const py = y0 + dy * a;
      const ix = roundToInt(px);
      const iy = roundToInt(py);

      if (this.checkPoint(ix, iy)) {
        return { collision: this.getCollisionAt(ix, iy), outPos: { x: px, y: py }, outBeforePos: { x: lastX, y: lastY } };
      }

      lastX = px;
      lastY = py;
    }
    return {
      collision: 0,
      outPos: { x: pos1.x, y: pos1.y },
      outBeforePos: { x: pos1.x, y: pos1.y },
    };
  }
}
