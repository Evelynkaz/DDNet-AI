export const SERVER_TICK_SPEED = 50;
export const PHYSICAL_SIZE = 28;
export const TICK_MS = 20;

function tune(v: number): number {
  return Math.trunc(v * 100) / 100;
}

export const TUNING = {
  groundControlSpeed: tune(10.0),
  groundControlAccel: tune(100.0 / SERVER_TICK_SPEED),
  groundFriction: tune(0.5),
  groundJumpImpulse: tune(13.2),
  airJumpImpulse: tune(12.0),
  airControlSpeed: tune(250.0 / SERVER_TICK_SPEED),
  airControlAccel: tune(1.5),
  airFriction: tune(0.95),
  hookLength: tune(380.0),
  hookFireSpeed: tune(80.0),
  hookDragAccel: tune(3.0),
  hookDragSpeed: tune(15.0),
  gravity: tune(0.5),

  velrampStart: tune(550),
  velrampRange: tune(2000),
  velrampCurvature: tune(1.4),

  gunCurvature: tune(1.25),
  gunSpeed: tune(2200.0),
  gunLifetime: tune(2.0),

  shotgunCurvature: tune(1.25),
  shotgunSpeed: tune(2750.0),
  shotgunSpeeddiff: tune(0.8),
  shotgunLifetime: tune(0.2),

  grenadeCurvature: tune(7.0),
  grenadeSpeed: tune(1000.0),
  grenadeLifetime: tune(2.0),

  laserReach: tune(800.0),
  laserBounceDelay: tune(150),
  laserBounceNum: tune(1000),
  laserBounceCost: tune(0),
  laserDamage: tune(5),

  playerCollision: tune(1),
  playerHooking: tune(1),

  jetpackStrength: tune(400.0),
  shotgunStrength: tune(10.0),
  explosionStrength: tune(6.0),
  hammerStrength: tune(1.0),
  hookDuration: tune(1.25),

  hammerFireDelay: tune(125),
  gunFireDelay: tune(125),
  shotgunFireDelay: tune(500),
  grenadeFireDelay: tune(500),
  laserFireDelay: tune(800),
  ninjaFireDelay: tune(800),
  hammerHitFireDelay: tune(320),

  groundElasticityX: tune(0),
  groundElasticityY: tune(0),
} as const;

export const TILE_AIR = 0;
export const TILE_SOLID = 1;
export const TILE_DEATH = 2;
export const TILE_NOHOOK = 3;
export const TILE_FREEZE = 9;
export const TILE_UNFREEZE = 11;

export const TILE_DFREEZE = 12;
export const TILE_DUNFREEZE = 13;

export const TILE_TELEINEVIL = 10;
export const TILE_TELEIN = 26;
export const TILE_TELEOUT = 27;

export const TILE_TELECHECK = 29;
export const TILE_TELECHECKOUT = 30;
export const TILE_TELECHECKIN = 31;
export const TILE_TELECHECKINEVIL = 63;

export const TILE_TELE_LASER_DISABLE = 129;
export const TILE_LFREEZE = 144;
export const TILE_LUNFREEZE = 145;

export const TILE_THROUGH_CUT = 5;
export const TILE_THROUGH = 6;
export const TILE_THROUGH_ALL = 66;
export const TILE_THROUGH_DIR = 67;

export const TILE_STOP = 60;
export const TILE_STOPS = 61;
export const TILE_STOPA = 62;

export const TILEFLAG_XFLIP = 1 << 0;
export const TILEFLAG_YFLIP = 1 << 1;
export const TILEFLAG_ROTATE = 1 << 3;
export const ROTATION_0 = 0;
export const ROTATION_90 = TILEFLAG_ROTATE;
export const ROTATION_180 = TILEFLAG_XFLIP | TILEFLAG_YFLIP;
export const ROTATION_270 = TILEFLAG_XFLIP | TILEFLAG_YFLIP | TILEFLAG_ROTATE;

export const CANTMOVE_LEFT = 1 << 0;
export const CANTMOVE_RIGHT = 1 << 1;
export const CANTMOVE_UP = 1 << 2;
export const CANTMOVE_DOWN = 1 << 3;

export const CFLAG_SOLID = 1 << 0;
export const CFLAG_DEATH = 1 << 1;
export const CFLAG_NOHOOK = 1 << 2;
