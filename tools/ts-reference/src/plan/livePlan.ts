import type { SimWorld } from "../core/world.ts";
import type { PlayerInput, TeeState, WorldEvent } from "../core/types.ts";
import { emptyInput, wireAngleRad } from "../core/types.ts";

export function enemyInputFromSnapshot(target: TeeState): PlayerInput {
  const enemyInput = emptyInput();
  enemyInput.direction = target.direction;
  enemyInput.hook = target.hookState > 0 ? 1 : 0;

  const enemyAngle = wireAngleRad(target.angle);
  enemyInput.targetX = Math.round(Math.cos(enemyAngle) * 300);
  enemyInput.targetY = Math.round(Math.sin(enemyAngle) * 300);
  return enemyInput;
}

export function syncPlanningWorld(
  sim: SimWorld,
  ownId: number,
  self: TeeState,
  targetId: number,
  target: TeeState,
  prevInput: PlayerInput,
  enemyInput: PlayerInput,
  lag: number,
  inFlight?: readonly PlayerInput[],
  heldAtSnapshot?: PlayerInput,
): WorldEvent[][] {
  sim.applyTeeState(ownId, self);
  sim.applyTeeState(targetId, target);
  sim.setHeldInput(ownId, heldAtSnapshot ?? prevInput);
  sim.setHeldInput(targetId, enemyInput);

  const rolled: WorldEvent[][] = [];
  for (let t = 0; t < lag; t++) {
    sim.setInput(ownId, inFlight?.[t] ?? prevInput);
    sim.setInput(targetId, enemyInput);
    rolled.push(sim.step());
  }
  return rolled;
}

export function syncPlanningWorldLegacy(
  sim: SimWorld,
  ownId: number,
  self: TeeState,
  targetId: number,
  target: TeeState,
  prevInput: PlayerInput,
  enemyInput: PlayerInput,
  lag: number,
): void {

  const old = (t: TeeState): TeeState => ({ ...t, hookTick: undefined, jumpedTotal: undefined, reloadTicks: undefined, frozenFor: undefined, jumps: undefined, ddnetFlags: undefined, sinceAttack: undefined });
  sim.applyTeeState(ownId, old(self));
  sim.applyTeeState(targetId, old(target));

  const recs = (sim as unknown as { tees: Map<number, { prevPos: { x: number; y: number } }> }).tees;
  for (const [id, t] of [[ownId, self], [targetId, target]] as const) {
    const r = recs.get(id);
    if (r !== undefined) r.prevPos = { x: t.pos.x, y: t.pos.y };
  }
  for (let t = 0; t < lag; t++) {
    sim.setInput(ownId, prevInput);
    sim.setInput(targetId, enemyInput);
    sim.step();
  }
}

export function syncOthers(sim: SimWorld, present: Set<number>, others: readonly TeeState[]): void {
  const wanted = new Set(others.map((t) => t.id));
  for (const id of present) {
    if (!wanted.has(id)) {
      sim.removeTee(id);
      present.delete(id);
    }
  }
  for (const t of others) {
    if (!present.has(t.id)) {
      sim.addTee(t.id, t.pos);
      present.add(t.id);
    }
    sim.applyTeeState(t.id, t);
    sim.setHeldInput(t.id, enemyInputFromSnapshot(t));
  }
}
