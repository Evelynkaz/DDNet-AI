export type Vec2 = { x: number; y: number };

export function vec2(x: number, y: number): Vec2 {
  return { x, y };
}

export function vadd(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y };
}

export function vsub(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y };
}

export function vmul(v: Vec2, f: number): Vec2 {
  return { x: v.x * f, y: v.y * f };
}

export function vdiv(v: Vec2, f: number): Vec2 {
  return { x: v.x / f, y: v.y / f };
}

export function vlength(a: Vec2): number {
  return Math.sqrt(vdot(a, a));
}

export function vdistance(a: Vec2, b: Vec2): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export function vdot(a: Vec2, b: Vec2): number {
  return a.x * b.x + a.y * b.y;
}

export function vnormalize(v: Vec2): Vec2 {
  const divisor = vlength(v);
  if (divisor === 0.0) {
    return { x: 0.0, y: 0.0 };
  }
  const l = 1.0 / divisor;
  return { x: v.x * l, y: v.y * l };
}

export function vmix(a: Vec2, b: Vec2, amount: number): Vec2 {
  return { x: a.x + (b.x - a.x) * amount, y: a.y + (b.y - a.y) * amount };
}

export function direction(angle: number): Vec2 {
  return { x: Math.cos(angle), y: Math.sin(angle) };
}

export function getAngle(dir: Vec2): number {
  if (dir.x === 0 && dir.y === 0) return 0.0;
  if (dir.x === 0) return dir.y < 0 ? -Math.PI / 2 : Math.PI / 2;
  let result = Math.atan(dir.y / dir.x);
  if (dir.x < 0) result = result + Math.PI;
  return result;
}

export function closestPointOnLineOrNull(lineA: Vec2, lineB: Vec2, targetPoint: Vec2): Vec2 | null {
  const AB = vsub(lineB, lineA);
  const squaredMagnitudeAB = vdot(AB, AB);
  if (squaredMagnitudeAB > 0) {
    const AP = vsub(targetPoint, lineA);
    const APdotAB = vdot(AP, AB);
    const t = APdotAB / squaredMagnitudeAB;
    return vadd(lineA, vmul(AB, clamp(t, 0, 1)));
  }
  return null;
}

export function closestPointOnLine(lineA: Vec2, lineB: Vec2, targetPoint: Vec2): Vec2 {
  return closestPointOnLineOrNull(lineA, lineB, targetPoint) ?? lineA;
}

export function clamp(val: number, min: number, max: number): number {
  if (val < min) return min;
  if (val > max) return max;
  return val;
}

export function roundToInt(f: number): number {
  return f > 0 ? Math.trunc(f + 0.5) : Math.trunc(f - 0.5);
}

export function sign(f: number): number {
  if (f > 0) return 1;
  if (f < 0) return -1;
  return 0;
}
