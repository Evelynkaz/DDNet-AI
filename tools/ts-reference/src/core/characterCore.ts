import type { Vec2 } from "./vmath.ts";
import { closestPointOnLineOrNull, roundToInt, vadd, vdistance, vdot, vlength, vmix, vmul, vnormalize, vsub } from "./vmath.ts";
import { PHYSICAL_SIZE, SERVER_TICK_SPEED, TUNING } from "./tuning.ts";
import type { Collision } from "./collision.ts";
import { clampVel } from "./collision.ts";
import { CFLAG_NOHOOK } from "./tuning.ts";
import type { PlayerInput } from "./types.ts";
import { emptyInput } from "./types.ts";

export const HOOK_RETRACTED = -1;
export const HOOK_IDLE = 0;
export const HOOK_RETRACT_START = 1;
export const HOOK_RETRACT_END = 3;
export const HOOK_FLYING = 4;
export const HOOK_GRABBED = 5;

export const COREEVENT_GROUND_JUMP = 0x01;
export const COREEVENT_AIR_JUMP = 0x02;
export const COREEVENT_HOOK_LAUNCH = 0x04;
export const COREEVENT_HOOK_ATTACH_PLAYER = 0x08;
export const COREEVENT_HOOK_ATTACH_GROUND = 0x10;
export const COREEVENT_HOOK_HIT_NOHOOK = 0x20;
export const COREEVENT_HOOK_RETRACT = 0x40;

export interface CoreWorld {
  allCores(): Iterable<CharacterCore>;
  coreById(id: number): CharacterCore | undefined;
}

function saturatedAdd(min: number, max: number, current: number, modifier: number): number {
  if (modifier < 0) {
    if (current < min) return current;
    current += modifier;
    if (current < min) current = min;
    return current;
  }
  if (current > max) return current;
  current += modifier;
  if (current > max) current = max;
  return current;
}

function velocityRamp(value: number, start: number, range: number, curvature: number): number {
  if (value < start) return 1.0;
  return 1.0 / Math.pow(curvature, (value - start) / range);
}

const MOVE_SIZE: Vec2 = { x: PHYSICAL_SIZE, y: PHYSICAL_SIZE };
const MOVE_ELASTICITY: Vec2 = { x: 0, y: 0 };

export class CharacterCore {
  readonly id: number;
  private readonly collision: Collision;
  private readonly world: CoreWorld;

  pos: Vec2 = { x: 0, y: 0 };
  vel: Vec2 = { x: 0, y: 0 };

  hookPos: Vec2 = { x: 0, y: 0 };
  hookDir: Vec2 = { x: 0, y: 0 };
  hookTeleBase: Vec2 = { x: 0, y: 0 };
  hookTick = 0;
  hookState = HOOK_IDLE;
  hookedPlayer = -1;
  attachedPlayers: Set<number> = new Set();

  activeWeapon = 0;

  newHook = false;

  moveRestrictions = 0;

  jumped = 0;
  jumpedTotal = 0;
  jumps = 2;

  direction = 0;
  angle = 0;
  input: PlayerInput = emptyInput();

  triggeredEvents = 0;

  colliding = 0;
  leftWall = false;

  freezeStart = 0;
  freezeEnd = 0;
  isInFreeze = false;
  collisionDisabled = false;
  hookHitDisabled = false;
  endlessHook = false;
  solo = false;

  constructor(id: number, collision: Collision, world: CoreWorld) {
    this.id = id;
    this.collision = collision;
    this.world = world;
  }

  reset(): void {
    this.pos = { x: 0, y: 0 };
    this.vel = { x: 0, y: 0 };
    this.newHook = false;
    this.moveRestrictions = 0;
    this.hookPos = { x: 0, y: 0 };
    this.hookDir = { x: 0, y: 0 };
    this.hookTeleBase = { x: 0, y: 0 };
    this.hookTick = 0;
    this.hookState = HOOK_IDLE;
    this.setHookedPlayer(-1);
    this.attachedPlayers.clear();
    this.jumped = 0;
    this.jumpedTotal = 0;
    this.jumps = 2;
    this.triggeredEvents = 0;

    this.solo = false;
    this.collisionDisabled = false;
    this.endlessHook = false;
    this.hookHitDisabled = false;
    this.freezeStart = 0;
    this.freezeEnd = 0;
    this.isInFreeze = false;

    this.input = emptyInput();
  }

  tick(useInput: boolean, doDeferred = true): void {

    this.moveRestrictions = this.collision.getMoveRestrictions(this.pos);
    this.triggeredEvents = 0;

    const grounded = this.collision.isSolid(this.pos.x + PHYSICAL_SIZE / 2, this.pos.y + PHYSICAL_SIZE / 2 + 5) ||
      this.collision.isSolid(this.pos.x - PHYSICAL_SIZE / 2, this.pos.y + PHYSICAL_SIZE / 2 + 5);

    this.vel.y += TUNING.gravity;

    const maxSpeed = grounded ? TUNING.groundControlSpeed : TUNING.airControlSpeed;
    const accel = grounded ? TUNING.groundControlAccel : TUNING.airControlAccel;
    const friction = grounded ? TUNING.groundFriction : TUNING.airFriction;

    if (useInput) {
      this.direction = this.input.direction;

      const pi = Math.PI;
      const tmpAngle = Math.atan2(this.input.targetY, this.input.targetX);
      if (tmpAngle < -(pi / 2.0)) {
        this.angle = Math.trunc((tmpAngle + 2.0 * pi) * 256.0);
      } else {
        this.angle = Math.trunc(tmpAngle * 256.0);
      }

      if (this.input.jump) {
        if (!(this.jumped & 1)) {
          if (grounded && (!(this.jumped & 2) || this.jumps !== 0)) {
            this.triggeredEvents |= COREEVENT_GROUND_JUMP;
            this.vel.y = -TUNING.groundJumpImpulse;
            if (this.jumps > 1) {
              this.jumped |= 1;
            } else {
              this.jumped |= 3;
            }
            this.jumpedTotal = 0;
          } else if (!(this.jumped & 2)) {
            this.triggeredEvents |= COREEVENT_AIR_JUMP;
            this.vel.y = -TUNING.airJumpImpulse;
            this.jumped |= 3;
            this.jumpedTotal++;
          }
        }
      } else {
        this.jumped &= ~1;
      }

      if (this.input.hook) {
        if (this.hookState === HOOK_IDLE) {

          const targetDirection = vnormalize({ x: this.input.targetX, y: this.input.targetY });
          this.hookState = HOOK_FLYING;
          this.hookPos = vadd(this.pos, vmul(targetDirection, PHYSICAL_SIZE * 1.5));
          this.hookDir = targetDirection;
          this.setHookedPlayer(-1);
          this.hookTick = Math.trunc(SERVER_TICK_SPEED * (1.25 - TUNING.hookDuration));
          this.triggeredEvents |= COREEVENT_HOOK_LAUNCH;
        }
      } else {
        this.setHookedPlayer(-1);
        this.hookState = HOOK_IDLE;
        this.hookPos = { ...this.pos };
      }
    }

    if (grounded) {
      this.jumped &= ~2;
      this.jumpedTotal = 0;
    }

    if (this.direction < 0) {
      this.vel.x = saturatedAdd(-maxSpeed, maxSpeed, this.vel.x, -accel);
    }
    if (this.direction > 0) {
      this.vel.x = saturatedAdd(-maxSpeed, maxSpeed, this.vel.x, accel);
    }
    if (this.direction === 0) {
      this.vel.x *= friction;
    }

    if (this.hookState === HOOK_IDLE) {
      this.setHookedPlayer(-1);
      this.hookPos = { ...this.pos };
    } else if (this.hookState >= HOOK_RETRACT_START && this.hookState < HOOK_RETRACT_END) {
      this.hookState++;
    } else if (this.hookState === HOOK_RETRACT_END) {
      this.triggeredEvents |= COREEVENT_HOOK_RETRACT;
      this.hookState = HOOK_RETRACTED;
    } else if (this.hookState === HOOK_FLYING) {
      const hookBase = this.pos;
      let newPos = vadd(this.hookPos, vmul(this.hookDir, TUNING.hookFireSpeed));
      if (vdistance(hookBase, newPos) > TUNING.hookLength) {
        this.hookState = HOOK_RETRACT_START;
        newPos = vadd(hookBase, vmul(vnormalize(vsub(newPos, hookBase)), TUNING.hookLength));
      }

      let goingToHitGround = false;
      let goingToRetract = false;

      const hit = this.collision.intersectLineHook(this.hookPos, newPos);
      if (hit.collision !== 0) {
        if ((hit.collision & CFLAG_NOHOOK) !== 0) {
          goingToRetract = true;
        } else {
          goingToHitGround = true;
        }
        newPos = hit.outPos;
      }

      if (!this.hookHitDisabled && TUNING.playerHooking && (this.hookState === HOOK_FLYING || !this.newHook)) {
        let bestDistance = 0;
        for (const other of this.world.allCores()) {
          if (other === this || other.solo || this.solo) continue;

          const closest = closestPointOnLineOrNull(this.hookPos, newPos, other.pos);
          if (closest !== null && vdistance(other.pos, closest) < PHYSICAL_SIZE + 2.0) {
            const d = vdistance(this.hookPos, other.pos);
            if (this.hookedPlayer === -1 || d < bestDistance) {
              this.triggeredEvents |= COREEVENT_HOOK_ATTACH_PLAYER;
              this.hookState = HOOK_GRABBED;
              this.setHookedPlayer(other.id);
              bestDistance = d;
            }
          }
        }
      }

      if (this.hookState === HOOK_FLYING) {
        if (goingToHitGround) {
          this.triggeredEvents |= COREEVENT_HOOK_ATTACH_GROUND;
          this.hookState = HOOK_GRABBED;
        } else if (goingToRetract) {
          this.triggeredEvents |= COREEVENT_HOOK_HIT_NOHOOK;
          this.hookState = HOOK_RETRACT_START;
        }
        this.hookPos = newPos;
      }
    }

    if (this.hookState === HOOK_GRABBED) {
      if (this.hookedPlayer !== -1) {
        const other = this.world.coreById(this.hookedPlayer);
        if (other) {
          this.hookPos = { ...other.pos };
        } else {
          this.setHookedPlayer(-1);
          this.hookState = HOOK_RETRACTED;
          this.hookPos = { ...this.pos };
        }
      }

      if (this.hookedPlayer === -1 && vdistance(this.hookPos, this.pos) > 46.0) {
        let hookVel = vmul(vnormalize(vsub(this.hookPos, this.pos)), TUNING.hookDragAccel);

        if (hookVel.y > 0) hookVel = { x: hookVel.x, y: hookVel.y * 0.3 };

        if ((hookVel.x < 0 && this.direction < 0) || (hookVel.x > 0 && this.direction > 0)) {
          hookVel = { x: hookVel.x * 0.95, y: hookVel.y };
        } else {
          hookVel = { x: hookVel.x * 0.75, y: hookVel.y };
        }

        const newVel = vadd(this.vel, hookVel);
        const newVelLength = vlength(newVel);
        if (newVelLength < TUNING.hookDragSpeed || newVelLength < vlength(this.vel)) {
          this.vel = newVel;
        }
      }

      this.hookTick++;
      const hookedGone = this.hookedPlayer !== -1 && !this.world.coreById(this.hookedPlayer);
      if (this.hookedPlayer !== -1 && (this.hookTick > SERVER_TICK_SPEED + SERVER_TICK_SPEED / 5 || hookedGone)) {
        this.setHookedPlayer(-1);
        this.hookState = HOOK_RETRACTED;
        this.hookPos = { ...this.pos };
      }
    }

    if (doDeferred) this.tickDeferred();
  }

  tickDeferred(): void {
    for (const other of this.world.allCores()) {
      if (other === this) continue;
      if (this.solo || other.solo) continue;

      const dist = vdistance(this.pos, other.pos);
      if (dist > 0) {

        const canCollide = !this.collisionDisabled && !other.collisionDisabled && TUNING.playerCollision;
        if (canCollide && dist < PHYSICAL_SIZE * 1.25) {
          const dir = vnormalize(vsub(this.pos, other.pos));
          const a = PHYSICAL_SIZE * 1.45 - dist;
          let velocity = 0.5;
          if (vlength(this.vel) > 0.0001) {
            velocity = 1 - (vdot(vnormalize(this.vel), dir) + 1) / 2;
          }
          this.vel = vadd(this.vel, vmul(dir, a * (velocity * 0.75)));
          this.vel = vmul(this.vel, 0.85);
        }

        if (!this.hookHitDisabled && this.hookedPlayer === other.id && TUNING.playerHooking) {
          if (dist > PHYSICAL_SIZE * 1.5) {
            const dir = vnormalize(vsub(this.pos, other.pos));
            const hookAccel = TUNING.hookDragAccel * (dist / TUNING.hookLength);
            const dragSpeed = TUNING.hookDragSpeed;

            other.vel = clampVel(other.moveRestrictions, {
              x: saturatedAdd(-dragSpeed, dragSpeed, other.vel.x, hookAccel * dir.x * 1.5),
              y: saturatedAdd(-dragSpeed, dragSpeed, other.vel.y, hookAccel * dir.y * 1.5),
            });
            this.vel = clampVel(this.moveRestrictions, {
              x: saturatedAdd(-dragSpeed, dragSpeed, this.vel.x, -hookAccel * dir.x * 0.25),
              y: saturatedAdd(-dragSpeed, dragSpeed, this.vel.y, -hookAccel * dir.y * 0.25),
            });
          }
        }
      }
    }

    if (this.hookState !== HOOK_FLYING) {
      this.newHook = false;
    }

    if (vlength(this.vel) > 6000) {
      this.vel = vmul(vnormalize(this.vel), 6000);
    }
  }

  move(): void {
    const rampValue = velocityRamp(vlength(this.vel) * 50, TUNING.velrampStart, TUNING.velrampRange, TUNING.velrampCurvature);
    this.vel.x = this.vel.x * rampValue;

    const newPos: Vec2 = { ...this.pos };
    const newVel: Vec2 = { ...this.vel };

    const oldVelX = this.vel.x;
    MOVE_ELASTICITY.x = TUNING.groundElasticityX;
    MOVE_ELASTICITY.y = TUNING.groundElasticityY;
    this.collision.moveBox(newPos, newVel, MOVE_SIZE, MOVE_ELASTICITY);
    this.vel = newVel;

    this.colliding = 0;
    if (this.vel.x < 0.001 && this.vel.x > -0.001) {
      if (oldVelX > 0) this.colliding = 1;
      else if (oldVelX < 0) this.colliding = 2;
    } else {
      this.leftWall = true;
    }

    this.vel.x = this.vel.x * (1.0 / rampValue);

    if (TUNING.playerCollision && !this.collisionDisabled && !this.solo) {

      const dist = vdistance(this.pos, newPos);
      if (dist > 0) {
        const end = Math.trunc(dist + 1);

        const fromX = this.pos.x;
        const fromY = this.pos.y;
        let lastX = fromX;
        let lastY = fromY;

        const cores = this.world.allCores();
        for (let i = 0; i < end; i++) {
          const a = i / dist;
          const px = fromX + (newPos.x - fromX) * a;
          const py = fromY + (newPos.y - fromY) * a;
          for (const other of cores) {
            if (other === this) continue;
            if (this.solo || other.solo || other.collisionDisabled) continue;
            const dx = px - other.pos.x;
            const dy = py - other.pos.y;
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d < PHYSICAL_SIZE) {
              if (a > 0.0) {
                this.pos = { x: lastX, y: lastY };
              } else if (vdistance(newPos, other.pos) > d) {
                this.pos = newPos;
              }
              return;
            }
          }
          lastX = px;
          lastY = py;
        }
      }
    }

    this.pos = newPos;
  }

  quantize(): void {
    const x = roundToInt(this.pos.x);
    const y = roundToInt(this.pos.y);
    const velX = roundToInt(this.vel.x * 256.0);
    const velY = roundToInt(this.vel.y * 256.0);
    const hookX = roundToInt(this.hookPos.x);
    const hookY = roundToInt(this.hookPos.y);
    const hookDx = roundToInt(this.hookDir.x * 256.0);
    const hookDy = roundToInt(this.hookDir.y * 256.0);

    this.pos = { x, y };
    this.vel = { x: velX / 256.0, y: velY / 256.0 };
    this.hookPos = { x: hookX, y: hookY };
    this.hookDir = { x: hookDx / 256.0, y: hookDy / 256.0 };
  }

  setHookedPlayer(hookedPlayer: number): void {
    if (hookedPlayer !== this.hookedPlayer) {
      if (this.hookedPlayer !== -1) {
        const prev = this.world.coreById(this.hookedPlayer);
        if (prev) prev.attachedPlayers.delete(this.id);
      }
      if (hookedPlayer !== -1) {
        const next = this.world.coreById(hookedPlayer);
        if (next) next.attachedPlayers.add(this.id);
      }
      this.hookedPlayer = hookedPlayer;
    }
  }
}
