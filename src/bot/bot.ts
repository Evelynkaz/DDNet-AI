import { createRequire } from "node:module";
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { Vec2 } from "../core/vmath.ts";
import { vdistance } from "../core/vmath.ts";
import { Collision } from "../core/collision.ts";
import { PHYSICAL_SIZE, TUNING } from "../core/tuning.ts";
import type { PlayerInput, TeeState } from "../core/types.ts";
import { WEAPON_GRENADE, WEAPON_HAMMER, emptyInput, wireAngleRad } from "../core/types.ts";
import { HOOK_FLYING, HOOK_IDLE } from "../core/characterCore.ts";
import type { RecurrentPolicy } from "../nn/gru.ts";
import { PLANNER_DEFAULTS, Planner, ropeCatchAlong } from "../plan/planner.ts";
import type { PlannerConfig } from "../plan/planner.ts";
import { getLang, isLang, setLang, t } from "../i18n.ts";
import type { Lang } from "../i18n.ts";

const TRY_SETTINGS: Record<string, Record<string, unknown>> = {

  trackaim: { trackAim: true },

  launch: { launchExposure: 1.0 },

  both: { trackAim: true, launchExposure: 1.0 },

  careful: { selfFreezeBias: 1.5 },

  careful2: { selfFreezeBias: 2.0 },

  bold: { selfFreezeBias: 1.0 },

  nothaw: { noThaw: true },

  readsyou: { opponentModel: "learned" },

  bothcareful: { trackAim: true, launchExposure: 1.0, selfFreezeBias: 1.5 },

  others2: { planOthers: 2 },
  oldcopy: { liveTransfer: "legacy" },

  smooth: { warmShiftElapsed: true },
};

const LIVE_BUDGET_MS = 18;

export const PLANNER_BOLD = { population: 64, iterations: 3, budgetMs: LIVE_BUDGET_MS } as const;
import { SimWorld } from "../core/world.ts";
import { Rng } from "../nn/rng.ts";
import { ACTION_SIZE, decodeAction } from "../env/action.ts";
import { OBS_SIZE, encodeObs, isGrounded} from "../env/obs.ts";
import { scriptedAction } from "../env/scripted.ts";
import { LiveWorld, mapCollisionFromClient } from "./liveWorld.ts";
import { RingRecorder, snapInput, snapTee } from "../watch/recording.ts";
import type { RecFrame } from "../watch/recording.ts";
import { enemyInputFromSnapshot, syncOthers, syncPlanningWorld, syncPlanningWorldLegacy } from "../plan/livePlan.ts";
import { restsInFreeze, sealedIn } from "../plan/seal.ts";
import { escapeExists, saferInput } from "../plan/shield.ts";
import { deadZone, findRoute, spawnTiles } from "../plan/route.ts";
import { FreezeMemory } from "../plan/memory.ts";
import type { RouteStep } from "../plan/route.ts";
import type { Mlp } from "../nn/mlp.ts";
import type { Recording } from "../watch/recording.ts";
import { findIncidents, mergeOverlapping } from "../watch/incidents.ts";
import { Navigator, teleGoals, tileGoal } from "./navigate.ts";
import { LagWatch, LOW_CPU, STRONG_WB } from "./cpuLoad.ts";
import type { LagSummary } from "./cpuLoad.ts";
import type { NavGoal } from "./navigate.ts";
import type { Crossing } from "./crossing.ts";
import { inAnyBox } from "./crossing.ts";
import { WAYBLOCKS, WbSideChooser, inWbHall, inWbLeash, inWbZone, sideAt, sideDef, wayblockFor, wbWalkAllowed } from "./wayblock.ts";
import type { WbDef, WbSide } from "./wayblock.ts";
import { installNetworkGuard, patchHuffman, patchRedirect, patchSnapshotDecoder } from "./netPatch.ts";
import type { NetGuard } from "./netPatch.ts";
import { AutoChat } from "./autoChat.ts";
import type { AutoChatConfig } from "./autoChat.ts";
import type { MapClientLike, RawSnapItem, SnapshotSource } from "./liveWorld.ts";

const require = createRequire(import.meta.url);

type TwPlayerInput = {
  m_Direction: number;
  m_TargetX: number;
  m_TargetY: number;
  m_Jump: number;
  m_Fire: number;
  m_Hook: number;
  m_PlayerFlags: number;
  m_WantedWeapon: number;
  m_NextWeapon: number;
  m_PrevWeapon: number;
};

type TwMovement = {
  RunLeft(): void;
  RunRight(): void;
  RunStop(): void;
  Jump(state?: boolean): void;
  Fire(): void;
  Hook(state?: boolean): void;
  WantedWeapon(weapon: number): void;
  SetAim(x: number, y: number): void;
  FlagPlaying(toggle?: boolean): void;
  FlagScoreboard?(toggle?: boolean): void;
};

type TwGame = {
  Say(message: string, team?: boolean): void;
  ChangePlayerInfo?(info: {
    name: string;
    clan: string;
    country: number;
    skin: string;
    use_custom_color: boolean;
    color_body: number;
    color_feet: number;
  }): void;
  SetTeam(team: number): void;
  Kill(): void;
  Vote(yes: boolean): void;
  CallVoteOption(value: string, reason: string): void;
  Emote(emoticon: number): void;
};

const EMOTICON_EXCLAMATION = 1;
const EMOTICON_HEARTS = 2;
const EMOTICON_DROP = 3;
const EMOTICON_SORRY = 6;
const EMOTICON_SPLATTEE = 9;
const EMOTICON_ZZZ = 12;
const EMOTICON_QUESTION = 15;

export const PLAN_OTHERS_PX = 500;

const ECHO_MAX_TICKS = 10;
const ECHO_MIN_SAMPLES = 40;

const ECHO_MATCH_MAX = 0.3;

const ECHO_MATCH_GAP = 0.05;
const ECHO_LOG = 8;

const MAX_LAG_TICKS = 6;
const DUEL_ACCEPT_COOLDOWN_MS = 15000;

const DUEL_ACCEPT_WINDOW_MS = 120_000;

const AUTO_ACCEPT_HOLD_MS = 10_000;

const DUEL_ENTER_TICKS = 25;

const DUEL_LEAVE_TICKS = 100;
const EMOTE_COOLDOWN_MS = 3000;

export const COMMAND_PREFIXES = ["!", "?"];

const BRUSH_OFFS = ["не", "лол", "ахах нет", "скилл", "чё", "не бот", "мимо", "ну да ну да"];

const BRUSH_OFFS_EN = ["no", "lol", "haha no", "skill", "what", "not a bot", "missed", "yeah yeah"];

const REPLY_DELAY_MS = 2200;
const REPLY_JITTER_MS = 1800;

const REPLY_COOLDOWN_MS = 60_000;

const ACCUSATION = /(?<![a-zа-яё])(bot|бот|боты|aimbot|аимбот|аимбота|cheat|чит|читер|читак|читы|hack|хак|aim|аим)(?![a-zа-яё])/iu;

export const TARGET_MAX_PX = 1600;
export const AGGRESSOR_RANGE_PX = 500;
export const AGGRESSOR_MEMORY_TICKS = 3 * 50;

const SWING_AT_US_PX = 128;

const BLOCK_CREDIT_TICKS = 50;

const TELEPORT_JUMP_PX = 200;

const WB_WALK_MAX_FAILS = 4;
const WB_WALK_PAUSE_MS = 5 * 60_000;
const WB_WALK_PAUSE_MAX_MS = 30 * 60_000;

const DUEL_LOG_KEEP = 2000;

const HAMMER_REACH_AHEAD_PX = 21;
const HAMMER_REACH_PX = 56;

const REFREEZE_TICKS = 6;
const AT_US_MEMORY_TICKS = AGGRESSOR_MEMORY_TICKS;

const AT_FRIEND_SCORE = 450;
const RESCUE_RANGE_PX = 10 * 32;

const RESCUE_HAMMER_PX = 56;

const RESCUE_MIN_FREEZE_TICKS = 25;
const RESCUE_WALK_RETRY_TICKS = 2 * 50;

const RESCUE_GIVE_UP_TICKS = 3 * 50;
const RESCUE_PAUSE_TICKS = 10 * 50;

const PULL_ROLL_TICKS = 40;
const PULL_HOLDS = [8, 16, 24, 32] as const;

const PULL_GIVE_UP_TICKS = 5 * 50;

const PULL_NONE_TICKS = 10;

const GO_HOME_AFTER_TICKS = 4 * 50;

const WB_RETURN_TICKS = 50;

const WB_FINISH = process.env.WB_FINISH !== "0";

const WB_THAW_URGENCY = Number(process.env.WB_URGENCY ?? "0");

export const WB_PLAN_OVERRIDES: Partial<PlannerConfig> = process.env.WB_PLAN
  ? (JSON.parse(process.env.WB_PLAN) as Partial<PlannerConfig>)
  : { noThawRope: true, frozenThrow: 3, airJumpCost: 0.3, launchExactReach: 100 };

const WB_PLAN_STRONG: Partial<PlannerConfig> = { ...WB_PLAN_OVERRIDES, ...STRONG_WB };

const WB_NO_CLIMB_TILES = 3;

function onWbSpot(here: { tx: number; ty: number }, p: { tx: number; ty: number }): boolean {
  return Math.abs(here.tx - p.tx) <= 2 && Math.abs(here.ty - p.ty) <= 2;
}

export const WB_ZONE_SCORE = 300;

const COUNTER_REACH_PX = 64;

const PARTNER_ROPED_SCORE = Number.isFinite(Number(process.env.TEAM_HELP)) && (process.env.TEAM_HELP ?? "").trim() !== "" ? Number(process.env.TEAM_HELP) : 700;
const WAYBLOCK_NAMES = WAYBLOCKS.map((d) => `'${d.name}'`).join(" and ");

const TRAVEL_RETRY_TICKS = 5 * 50;

const CROWD_RADIUS_PX = 600;
const ACTION_MEMORY_TICKS = 2 * 50;

const SEEK_MARGIN = 3;
const SEEK_PATIENCE_TICKS = 4 * 50;
const SEEK_ARRIVED_PX = 500;

const TREK_REACHED_PX = 56;

export const PATH_NEAR_PX = 420;
export const PATH_REACHED_PX = 56;

export const PATH_REFRESH_TICKS = 25;
export const PATH_MIN_REFRESH_TICKS = 6;
export const PATH_MOVED_PX = 96;

export const PATH_PROGRESS_WINDOW = 10;
const TREK_STALL_TICKS = 150;

export const ENGAGED_PX = 420;

export const AFK_TICKS = 10 * 50;

const INPUT_SETTLE_TICKS = 50;

const CROWD_FROZEN_TICKS = 5 * 50;

const FOLLOW_ARRIVED_PX = 64;
const FOLLOW_LOST_TICKS = 5 * 50;
const FOLLOW_MAX_FAILS = 3;
const FOLLOW_RETRY_TICKS = 50;
const FOLLOW_STALL_TICKS = 30 * 50;
const FOLLOW_MAX_TICKS = 120 * 50;
const FOLLOW_MAX_DEATHS = 3;
const FOLLOW_GOAL_TILES = 3;

const FOLLOW_JUMP_PX = 8 * 32;
const GOTO_USAGE = "?goto tele | ?goto <x> <y> in tiles | ?goto <nick> (or @nick) | ?goto for progress | ?stop to call it off";

function listed(list: Map<string, string>, nameKey: string, partners?: ReadonlySet<string>, byId = false): boolean {
  if (nameKey === "" || list.size === 0) return false;
  const bare = nameKey.replace(DUPLICATE_PREFIX, "");
  const partnerish = (key: string): boolean => partners?.has(key.replace(DUPLICATE_PREFIX, "")) === true;
  if (list.has(nameKey) && !(byId && partnerish(nameKey))) return true;
  for (const key of list.keys()) {
    if (key === "" || partnerish(key)) continue;
    if (nameKey.includes(key) || key.replace(DUPLICATE_PREFIX, "") === bare) return true;
  }
  return false;
}

const DUPLICATE_PREFIX = /^\(\d+\)/;

function partnerNick(partners: ReadonlySet<string>, nameKey: string): boolean {
  if (partners.has(nameKey)) return true;
  if (!DUPLICATE_PREFIX.test(nameKey)) return false;
  const bare = nameKey.replace(DUPLICATE_PREFIX, "");
  if (bare === "") return false;
  for (const k of partners) if (k.startsWith(bare)) return true;
  return false;
}

function foldName(name: string): string {
  return name.trim().replace(/\s+/g, " ").toLowerCase();
}

export function matchPlayer(
  players: readonly { id: number; name: string }[],
  text: string,
  selfId: number,
): { id: number; name: string } | { none: true } | { self: string } | { many: string[] } {
  const want = foldName(text);
  if (want === "") return { none: true };
  const named = players.filter((p) => foldName(p.name) !== "");
  const exact = named.find((p) => foldName(p.name) === want);
  if (exact !== undefined) return exact.id === selfId ? { self: exact.name } : { id: exact.id, name: exact.name };
  const others = named.filter((p) => p.id !== selfId);
  for (const hits of [others.filter((p) => foldName(p.name).startsWith(want)), others.filter((p) => foldName(p.name).includes(want))]) {
    if (hits.length === 1) return { id: hits[0].id, name: hits[0].name };
    if (hits.length > 1) return { many: hits.map((p) => p.name.trim()) };
  }
  return { none: true };
}

export function inputKeysOf(t: TeeState): number {
  return (t.direction + 1) | ((t.jumped & 1) << 2) | ((t.hookState !== HOOK_IDLE ? 1 : 0) << 3);
}

const RELATIONS_FILE = "runs/relations.json";

const HALF_TEE = PHYSICAL_SIZE / 2;
export const BLOCKING_RANGE_PX = 320;

const GRENADE_WATCH_PX = 600;

export const FINISH_BLOCK_TICKS = 150;
export const FINISH_BLOCK_SCORE = 600;

export const SEAL_NEAR_TILES = 2;

export const SEAL_ANSWER_TICKS = 6;

export const BYSTANDER_PX = 160;

export const REACH_ANSWER_TICKS = 25;

export const OUT_OF_REACH_SCORE = 700;
export const REACH_MAX_NODES = 20000;

export const TARGET_HOLD_SCORE = 400;

export const HOLD_FADE_PX = 400;
export const TARGET_DIST_WEIGHT = 0.25;

const EMOTICON_BY_NAME: Record<string, number> = {
  exclamation: EMOTICON_EXCLAMATION,
  hearts: EMOTICON_HEARTS,
  drop: EMOTICON_DROP,
  sorry: EMOTICON_SORRY,
  splat: EMOTICON_SPLATTEE,
  zzz: EMOTICON_ZZZ,
  question: EMOTICON_QUESTION,
};

export function cursorOf(input: PlayerInput): { x: number; y: number } | undefined {
  const len = Math.hypot(input.targetX, input.targetY);
  if (!(len >= 1)) return undefined;
  const k = Math.min(1, 400 / len);
  return { x: Math.round(input.targetX * k), y: Math.round(input.targetY * k) };
}

export type BotLine = { kind: "log" | "chat" | "event" | "whisper"; text: string; from?: string; sys?: boolean };

const CHAT_ALL = 0;
const CHAT_TEAM = 1;
const CHAT_WHISPER_SENT = 2;
const CHAT_WHISPER_RECV = 3;

export type LiveMap = {
  name: string;
  width: number;
  height: number;
  kinds: string;

  traps?: string;
};
export type LiveTee = {
  id: number;
  name: string;
  x: number;
  y: number;
  frozen: boolean;
  hook: number;
  hx: number;
  hy: number;
  hooked: number;

  clan?: string;
  skin?: string;
  cc?: boolean;
  cb?: number;
  cf?: number;
  aim?: number;
  wp?: number;
  emote?: number;
  vx?: number;
  vy?: number;
  dir?: number;
  jumped?: number;
  atk?: number;
  fz?: number;
  pf?: number;
  jl?: number;
  fzf?: number;
  deep?: boolean;
  xf?: number;
  jt?: number;
};

export type LivePlayer = {
  id: number;
  name: string;
  clan: string;
  score: number;
  ping: number;
  team: number;
  skin?: string;
  cc?: boolean;
  cb?: number;
  cf?: number;
};
export type LiveFrame = {
  tick: number;
  selfId: number;
  target: number;
  map: string;
  tees: LiveTee[];

  doing: string;
  goal: { x: number; y: number } | null;
  route: { x: number; y: number; kind: string }[];

  cursor?: { x: number; y: number };
  players?: LivePlayer[];

  roundStart?: number;

  timeScore?: boolean;

  mapKey?: string;
};

export type BotStatus = {
  phase: "offline" | "connecting" | "online";
  acting: boolean;
  brain: "planner" | "net" | "scripted";
  mode: "fight" | "passive" | "hold" | "goto";
  server: string;
  name: string;
  targetName: string | null;
  targetDist: number | null;

  walk: string | null;

  offlineReason: string;

  wb?: string | null;

  panel?: { wbMode: "auto" | "left" | "right" | "off" | null; duelMode: "auto" | "on" | "off"; inDuel: boolean; spectating: boolean; pinnedTarget: string | null; home: boolean; tryName: string; duelScore?: { name: string; ours: number; theirs: number } | null };
  frozen: boolean;
  tick: number;
  stats: BotStats;

  lowCpu?: boolean;

  strong?: boolean;
  lag?: LagSummary;
};

type TwPlayerInfo = { local: number; client_id: number; team: number; score: number; latency: number };

type TwClientInfo = {
  name: string;
  clan: string;
  id: number;
  skin?: string;
  use_custom_color?: number;
  color_body?: number;
  color_feet?: number;
};

interface TwSnapshotUnpacker extends SnapshotSource {
  readonly OwnID: number | undefined;
  getObjPlayerInfo(id: number): TwPlayerInfo | undefined;
  readonly AllObjClientInfo: TwClientInfo[];
}

type TwKill = { killer_id: number; victim_id: number; weapon: number };

type TwMessage = { team: number; client_id: number; message: string };

interface TwClient extends MapClientLike {
  on(event: "connected", listener: () => void): this;
  on(event: "disconnect", listener: (reason: string, fromServer: boolean) => void): this;
  on(event: "snapshot", listener: (items: unknown[]) => void): this;
  on(event: "kill", listener: (kill: TwKill) => void): this;
  on(event: "message", listener: (message: TwMessage) => void): this;
  on(event: "map_change", listener: (change: { map_name: string }) => void): this;
  connect(): Promise<void>;
  Disconnect(): Promise<unknown>;
  readonly State: number;
  readonly SnapshotUnpacker: TwSnapshotUnpacker;
  readonly movement: TwMovement;
  readonly game: TwGame;
  sendInput(): void;
  readonly input: TwPlayerInput;

  readonly currentSnapshotGameTick?: number;

  readonly rawSnapUnpacker?: { readonly deltas: readonly RawSnapItem[] };
}

type TwIdentity = {
  name: string;
  clan: string;
  country: number;
  skin: string;
  use_custom_color: number;
  color_body: number;
  color_feet: number;
};

type TwOptions = {
  identity?: TwIdentity;
  password?: string;
  ddnet_version?: { version: number; release_version: string };
  timeout?: number;
  lightweight?: boolean;
  downloadMap?: boolean;
};

type TeeworldsModule = {
  Client: new (host: string, port: number, nickname: string, options?: TwOptions) => TwClient;
  Protocol: { States: { STATE_OFFLINE: number; STATE_CONNECTING: number; STATE_LOADING: number; STATE_ONLINE: number } };
};

const teeworlds = require("teeworlds") as TeeworldsModule;

const DEFAULT_PROTOCOL_VERSION = 19000;
const CLIENT_VERSION = { version: DEFAULT_PROTOCOL_VERSION, release_version: "19.0" };

const CHAT_MIN_INTERVAL_MS = 2000;
const JOIN_RETRY_MS = 3000;
const COLLISION_RETRY_MS = 2000;
const STATUS_INTERVAL_MS = 5000;
const RECONNECT_MIN_MS = 1000;
const RECONNECT_MAX_MS = 30000;

const KILL_COOLDOWN_TICKS = 10 * 50;

const WB_LYING_TICKS = 25;
const WB_KILL_COOLDOWN_TICKS = 2 * 50;

const HELPED_LIMIT_TICKS = 30 * 50;

const HELPER_RANGE_PX = 140;

const SNAPSHOTS_PER_SECOND = 25;
const CLIP_SECONDS = 30;

const CLIP_SCAN_EVERY_FRAMES = 50;

const CLIP_COOLDOWN_TICKS = 45 * 50;

const CLIP_SEVERITY = 250;

const CLIP_SEVERITY_BY_KIND: Record<string, number> = { "self-freeze": 180, "chased-into-freeze": 180, "goto-into-freeze": 180 };
const DEFAULT_CLIP_DIR = "runs/clips";

const CROSS_CLIP_COOLDOWN_TICKS = 60 * 50;

const CLIP_KEEP = 24;
const CLIP_KEEP_PER_KIND = 16;

export const LIVE_PLANNER_CFG: Record<string, unknown> = { thirdTeeExposure: 0, memoryTrust: 0.9, frozenThrow: 3 };

export type DuelRow = { at: string; seconds: number; opponent: string; ours: number; theirs: number; by?: string };

export function readDuelFile(file: string): DuelRow[] {
  try {
    const raw = JSON.parse(readFileSync(file, "utf8")) as unknown;
    if (!Array.isArray(raw)) return [];
    return raw
      .filter((d): d is DuelRow => d !== null && typeof d === "object" && typeof (d as { opponent?: unknown }).opponent === "string" && typeof (d as { at?: unknown }).at === "string")
      .reverse();
  } catch {
    return [];
  }
}

export function bothBotsDuels(first: DuelRow[], firstName: string, second: DuelRow[], secondName: string): DuelRow[] {
  return [...first.map((d) => ({ ...d, by: firstName })), ...second.map((d) => ({ ...d, by: secondName }))].sort((a, b) => (b.at < a.at ? -1 : b.at > a.at ? 1 : 0));
}

export type BotConfig = {
  host: string;
  port: number;
  name: string;
  clan?: string;

  clipDir?: string;

  acceptDuels?: boolean;

  relationsFile?: string;

  settingsFile?: string;

  lowCpu?: boolean;

  strong?: boolean;

  autoServer?: boolean;

  autoAvoidFile?: string;

  lagCompensation?: boolean;

  password?: string;
  skin?: string;
  country?: number;
  colorBody?: number;
  colorFeet?: number;
  policy?: RecurrentPolicy;

  planner?: boolean;

  plannerCfg?: ConstructorParameters<typeof Planner>[0];

  memoryDir?: string;

  opponentDirNet?: Mlp | null;
  scripted?: boolean;
  mapDir: string;
  chat?: boolean;

  emotes?: boolean;
  targetName?: string;

  brushOff?: boolean;

  protocolVersion?: number;

  goto?: string;
  reconnect?: boolean;
  verbose?: boolean;
};

export type BotStats = {
  ticks: number;
  kills: number;
  deaths: number;
  selfKills: number;
  clips: number;
  hammerFires: number;
  hooksFired: number;
  disconnects: number;
  errors: number;

  blocks: number;
  blockedBy: number;
};

export type Travel = { start: Vec2 | null; end: Vec2 | null; distance: number };

export function wanderHazardBelow(col: { isSolid(x: number, y: number): boolean; isFreeze(x: number, y: number): boolean; isDeath(x: number, y: number): boolean }, x: number, y: number): boolean {
  for (let k = 1; k <= 6; k++) {
    const yy = y + k * 32;
    if (col.isSolid(x, yy)) return false;
    if (col.isFreeze(x, yy) || col.isDeath(x, yy)) return true;
  }
  return false;
}

export function firePressed(prev: PlayerInput, cur: PlayerInput): boolean {
  return cur.fire !== prev.fire && (cur.fire & 1) !== 0;
}

export class DdnetBot {
  readonly stats: BotStats = { ticks: 0, kills: 0, deaths: 0, selfKills: 0, clips: 0, hammerFires: 0, hooksFired: 0, disconnects: 0, errors: 0, blocks: 0, blockedBy: 0 };

  private readonly lastTouch = new Map<number, { by: number; tick: number }>();
  private readonly lastPosById = new Map<number, { x: number; y: number; tick: number }>();
  private readonly thawTickById = new Map<number, number>();
  readonly travel: Travel = { start: null, end: null, distance: 0 };

  private readonly cfg: BotConfig;
  private client: TwClient | undefined;
  private readonly world: LiveWorld;
  private collisionReady = false;
  private lastCollisionTryMs = 0;
  private phase: "offline" | "connecting" | "online" = "offline";
  private stopping = false;
  private acting = true;

  private mode: "fight" | "passive" | "hold" | "goto" = "fight";
  private nav: Navigator | null = null;

  private pendingGoto: string | null = null;

  private navReturnMode: "fight" | "passive" | "hold" = "fight";

  private navPending = false;
  private ownId = -1;
  private targetId = -1;
  private prevInput: PlayerInput = emptyInput();

  private sent: { tick: number; input: PlayerInput }[] = [];
  private readonly obs = new Float64Array(OBS_SIZE);
  private readonly raw = new Float64Array(ACTION_SIZE);
  private readonly rng = new Rng((Date.now() ^ process.pid) >>> 0);
  private reconnectDelayMs = RECONNECT_MIN_MS;
  private reconnectTimer: ReturnType<typeof setTimeout> | undefined;
  private lastChatMs = 0;
  private lastJoinMs = 0;

  private wantSpectate = false;
  private lastStatusMs = 0;
  private lastPos: Vec2 | null = null;

  private stuckAnchor: Vec2 | null = null;
  private stuckAnchorTick = 0;
  private stuckAnchorMs = 0;
  private stuckAnchorFrozen = false;

  private frozenSince = -1;

  private lastDisconnect = "";

  private readonly frozenSinceById = new Map<number, number>();

  private readonly atUsById = new Map<number, number>();

  private readonly atFriendById = new Map<number, number>();
  private rescueWalkTick = -Infinity;
  private rescueSaidTick = -Infinity;

  private rescueId = -1;
  private rescueHammerSince = -1;
  private rescueHammerId = -1;
  private readonly rescuePausedUntil = new Map<number, number>();

  private pullSim: SimWorld | null = null;
  private rescuePullId = -1;
  private rescuePullSince = -1;
  private rescuePullAt = -Infinity;
  private readonly pullNone = new Map<number, { key: string; tick: number }>();

  private readonly lastInputById = new Map<
    number,
    { name: string; angle: number; attack: number; keys: number; at: number; firstSeen: number; changed: number; settleUntil: number }
  >();

  private follow: {
    id: number;
    name: string;
    tx: number;
    ty: number;
    routedTick: number;
    seenTick: number;
    startTick: number;
    best: number;
    bestTick: number;
    fails: number;
    blockedAt: number;
    deaths: number;
    selfDeaths: number;
    last: Vec2 | null;
    waiting: boolean;
    navOpts: { throughFreeze?: boolean; crossings?: readonly Crossing[] };
  } | null = null;

  private readonly relations = {
    war: new Map<string, string>(),
    friend: new Map<string, string>(),
    clanWar: new Map<string, string>(),
    clanFriend: new Map<string, string>(),

    ignore: new Map<string, string>(),
  };

  private home: { tx: number; ty: number } | null = null;

  private homeMap = "?";

  private wbDef: WbDef | null = null;
  private wbMode: "auto" | "left" | "right" | "off" = "auto";
  private readonly wbChooser = new WbSideChooser();
  private wbCounts = { left: 0, right: 0 };
  private wbWalk = false;

  private wbWalkFails = 0;
  private wbPauses = 0;
  private wbPausedUntilMs = 0;

  private routeKillTick = -1_000_000;

  private duelMode: "auto" | "on" | "off" = "auto";
  private duelSeen = false;
  private duelForSince = -1;
  private duelAgainstSince = -1;
  private idleSinceTick = -1;

  private travelSinceTick = -1000;
  private dullSinceTick = -1;
  private seekingGame = false;

  private trek: { steps: RouteStep[]; at: number; since: number; best: number; bestTick: number } | null = null;

  private trekAvoid = new Set<number>();
  private trekAvoidMap: unknown = null;

  private deadCells: { width: number; cells: Uint8Array } | null = null;

  private memory: FreezeMemory | null = null;

  private memoryPath: string | null = null;
  private memoryDirty = false;
  private lastTileIndex = -1;

  private path: { steps: RouteStep[]; at: number; target: number; to: Vec2; done: number } | null = null;

  private readonly clipRing = new RingRecorder(CLIP_SECONDS * SNAPSHOTS_PER_SECOND);
  private clipMapSet = false;
  private lastClipTick = -Infinity;

  private lastCrossClipTick = -Infinity;
  private framesSinceScan = 0;
  private lastKillTick = -Infinity;
  private lastEmoteMs = 0;
  private lastAcceptMs = 0;

  private duelAnsweredMs = 0;

  private armAcceptLine = "";
  private armAcceptUntil = 0;

  private duelChallenger = "";
  private readonly lastSeenDist = new Map<number, number>();
  private readonly lastReplyMs = new Map<number, number>();
  private readonly replyTimers = new Set<NodeJS.Timeout>();
  private wanderDir = 0;
  private wanderUntilTick = 0;
  private wanderAim = 0;

  private wanderLook = 0;
  private wanderJumpUntilTick = 0;
  private wanderHookUntilTick = 0;
  private readonly wanderRng = new Rng(0x5eed ^ Date.now());
  private planner: Planner | null = null;

  private lowCpu = false;

  private strong = false;

  private readonly lag = new LagWatch();

  private snapQueued = false;
  private snapArrived = 0;
  private snapTick: number | undefined = undefined;
  private snapFirstRead = 0;

  private reachLeft = -1;
  private planSim: SimWorld | null = null;

  private sealSim: SimWorld | null = null;

  private shieldSim: SimWorld | null = null;
  private sealAnswers = new Map<number, { tick: number; sealed: boolean }>();

  private reachAnswers = new Map<number, { tick: number; from: number; to: number; ok: boolean }>();

  private readonly reachWanted = new Set<number>();
  private reachWantTick = -1;

  private readonly planOthersIn = new Set<number>();
  private planTargetId = -1;
  private planSelfId = -1;
  private planCollision: Collision | null = null;
  private wasFrozen = false;
  private wasAlive = false;
  private startWaiter: { resolve: () => void; reject: (err: Error) => void } | undefined;

  constructor(cfg: BotConfig) {
    if (!cfg.scripted && !cfg.policy && !cfg.planner) {
      throw new Error("DdnetBot: one of policy, scripted or planner is required");
    }

    Object.assign(this.baseCfg, LIVE_PLANNER_CFG, cfg.plannerCfg ?? {});
    Object.assign(this.startCfg, this.baseCfg);
    if (cfg.policy) {
      const { inputs, outputs } = cfg.policy.shape;
      if (inputs !== OBS_SIZE || outputs !== ACTION_SIZE) {
        throw new Error(`DdnetBot: policy shape ${inputs}->${outputs} does not match OBS_SIZE=${OBS_SIZE}/ACTION_SIZE=${ACTION_SIZE}`);
      }
    }
    this.cfg = { ...cfg, plannerCfg: { ...this.baseCfg } };
    this.lowCpu = cfg.lowCpu === true;
    this.strong = cfg.strong === true && !this.lowCpu;

    this.world = new LiveWorld(new Collision(1, 1, new Uint8Array(1)));
    this.loadRelations();

    this.autoChat = new AutoChat(join(dirname(this.cfg.relationsFile ?? RELATIONS_FILE), "autochat.json"));
  }

  private readonly autoChat: AutoChat;
  autoChatInfo(): AutoChatConfig {
    return this.autoChat.config();
  }
  setAutoChat(raw: unknown): AutoChatConfig {
    const cfg = this.autoChat.set(raw);
    this.log(`auto chat: every ${cfg.periodic.on ? `${cfg.periodic.everySec}s` : "off"}, name ${cfg.mention.on ? "on" : "off"}, ${cfg.keywords.filter((k) => k.on).length} keyword rule(s)`);
    return cfg;
  }

  private autoSay(text: string, line: string | null = null): void {
    const ok = this.say(text);
    this.autoChat.sent(text, ok);
    if (ok) this.emit("event", `auto chat: ${text}`);
    if (line === null || !/^\/(?:accept|duel)\b/.test(text.trim().toLowerCase())) return;
    if (ok) {
      this.answeredInvitation(line);
      this.armAcceptUntil = 0;
    } else {
      this.armAcceptLine = line;
      this.armAcceptUntil = Date.now() + AUTO_ACCEPT_HOLD_MS;
    }
  }

  private answeredInvitation(line: string): void {
    if (!/\/(?:accept|duel)\b/i.test(line)) return;
    this.duelAnsweredMs = Date.now();

    const m = /^\s*'(.{1,40}?)'(?=\s)/u.exec(line) ?? /^\s*(.+?)\s+(?:challenged|invited|invites|challenges|вызвал|вызывает|пригласил|приглашает)(?![\p{L}])/iu.exec(line) ?? /(?<![\p{L}\p{N}_])(?:from|от)\s+(\S+)/iu.exec(line);
    this.duelChallenger = m === null ? "" : m[1].trim().toLowerCase().replace(/[\s,.!:;]+$/u, "");
  }

  start(): Promise<void> {

    if (!patchHuffman()) this.log("could not bound the huffman decoder; a malformed packet may still be fatal");
    if (!patchRedirect()) this.log("could not teach the network library server redirects");
    if (!patchSnapshotDecoder()) this.log("could not replace the network library's snapshot decoder; a slow PC may fall behind for good");
    this.netGuard ??= installNetworkGuard((message, dropped) => {
      this.stats.errors++;
      this.emit("event", `dropped an undecodable packet (${dropped} so far): ${message}`);

      if (dropped === 1 || dropped % 50 === 0) {
        if (dropped >= 50) this.log(`${dropped} undecodable packets from this server -- the connection may be unusable`);
      }
    });
    this.netGuard.install();
    if (this.client) return Promise.reject(new Error("DdnetBot.start: already started"));
    this.watchServer();
    const client = new teeworlds.Client(this.cfg.host, this.cfg.port, this.cfg.name, {
      identity: {
        name: this.cfg.name,
        clan: this.cfg.clan ?? "",
        country: this.cfg.country ?? -1,
        skin: this.cfg.skin ?? "default",
        use_custom_color: this.cfg.colorBody !== undefined || this.cfg.colorFeet !== undefined ? 1 : 0,
        color_body: this.cfg.colorBody ?? 0,
        color_feet: this.cfg.colorFeet ?? 0,
      },
      ...(this.cfg.password !== undefined && this.cfg.password.length > 0 ? { password: this.cfg.password } : {}),
      ddnet_version:
        this.cfg.protocolVersion === undefined
          ? CLIENT_VERSION
          : { version: this.cfg.protocolVersion, release_version: `${(this.cfg.protocolVersion / 1000).toFixed(1)}` },
      downloadMap: true,
    });
    this.client = client;
    client.on("connected", () => this.onConnected());
    client.on("disconnect", (reason, fromServer) => this.onDisconnect(reason, fromServer));

    (client as unknown as { on(e: "redirect", fn: (port: number) => void): void }).on("redirect", (port) => {
      if (port === this.cfg.port) return;
      this.emit("event", `the server redirects us to port ${port}`);
      this.cfg.port = port;
      (client as unknown as { port: number }).port = port;
      this.reconnectDelayMs = RECONNECT_MIN_MS;

      void client.Disconnect().catch(() => {});
    });
    client.on("snapshot", () => this.queueSnapshot());
    client.on("kill", (kill) => this.onKill(kill));
    client.on("message", (msg) => this.onMessage(msg));
    client.on("map_change", (change) => {
      this.collisionReady = false;
      this.lastCollisionTryMs = 0;

      this.onTickReset(0);
      this.targetId = -1;
      this.lastSeenDist.clear();

      if (this.nav !== null) {
        this.emit("event", `goto: dropped, the map is now '${change.map_name}'`);
        this.endNav();
      }

      if (this.home !== null) {
        if (this.homeMap === "?") this.homeMap = change.map_name;
        else if (change.map_name !== this.homeMap) {
          this.emit("event", `home (${this.home.tx},${this.home.ty}) forgotten: it was on '${this.homeMap}', the map is now '${change.map_name}'`);
          this.home = null;
        }
      }

      this.wbDef = null;
      this.wbChooser.reset();

      this.navPending = this.cfg.goto !== undefined && this.cfg.goto.trim().length > 0;
      this.log(`map change: ${change.map_name}`);
    });
    return new Promise((resolve, reject) => {
      this.startWaiter = { resolve, reject };
      this.connect();
    });
  }

  async stop(): Promise<void> {

    this.endDuelScore();
    this.stopping = true;
    if (this.serverWatch !== null) clearInterval(this.serverWatch);
    this.serverWatch = null;
    this.netGuard?.uninstall();

    for (const t of this.replyTimers) clearTimeout(t);
    this.replyTimers.clear();

    this.saveMemory();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = undefined;
    }
    const client = this.client;
    if (!client) return;

    if (client.State !== teeworlds.Protocol.States.STATE_OFFLINE) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const timeout = new Promise<void>((resolve) => {
        timer = setTimeout(resolve, 2500);
      });
      await Promise.race([client.Disconnect(), timeout]);
      if (timer) clearTimeout(timer);
    }
    this.phase = "offline";
  }

  statsLine(): string {
    const s = this.stats;

    const held = this.world.getTee(this.ownId ?? -1)?.activeWeapon;
    const weapon = held === undefined ? "?" : held === WEAPON_HAMMER ? "hammer" : `weapon${held}`;

    return `ticks=${s.ticks} brain=${this.brainName()} weapon=${weapon} try=${this.tryName} kills=${s.kills} deaths=${s.deaths} selfKills=${s.selfKills} clips=${s.clips} hammerFires=${s.hammerFires} hooksFired=${s.hooksFired} blocks=${s.blocks} blockedBy=${s.blockedBy} disconnects=${s.disconnects} errors=${s.errors}`;
  }

  private sink: ((line: BotLine) => void) | null = null;

  onOutput(sink: ((line: BotLine) => void) | null): void {
    this.sink = sink;
  }

  private quitRequested: (() => void) | null = null;
  private netGuard: NetGuard | null = null;

  onQuitRequested(fn: () => void): void {
    this.quitRequested = fn;
  }

  private switchRequested: (() => void) | null = null;
  private serverWatch: ReturnType<typeof setInterval> | null = null;
  private emptySince = 0;

  onSwitchRequested(fn: () => void): void {
    this.switchRequested = fn;
  }

  private watchServer(): void {
    if (this.cfg.autoServer !== true || this.serverWatch !== null) return;
    this.serverWatch = setInterval(() => void this.checkServer(), 30_000);
    this.serverWatch.unref?.();
  }

  private offlineSince = 0;

  private async checkServer(): Promise<void> {
    if (this.stopping || this.switchRequested === null) return;
    const now = Date.now();
    const here = `${this.cfg.host}:${this.cfg.port}`;

    if (this.phase !== "online") {
      if (this.offlineSince === 0) this.offlineSince = now;
      if (now - this.offlineSince < 120_000) return;
      this.offlineSince = 0;
      try {
        if (this.cfg.autoAvoidFile !== undefined) {
          const { addAvoid } = await import("./serverPick.ts");
          addAvoid(this.cfg.autoAvoidFile, here, 30 * 60_000);
        }
      } catch {

      }
      this.emit("event", t("на {addr} не пускает, ищу другой сервер", { addr: here }));
      this.switchRequested?.();
      return;
    }
    this.offlineSince = 0;
    const others = Math.max(0, (this.client?.SnapshotUnpacker?.AllObjClientInfo?.length ?? 1) - 1);
    if (others >= 2) {
      this.emptySince = 0;
      return;
    }
    if (this.emptySince === 0) this.emptySince = now;
    if (now - this.emptySince < 180_000) return;

    this.emptySince = now;
    try {
      const { fetchMaster, pickBlockServer, readAvoid } = await import("./serverPick.ts");
      const avoid = [here, ...(this.cfg.autoAvoidFile !== undefined ? readAvoid(this.cfg.autoAvoidFile) : [])];
      const pick = pickBlockServer(await fetchMaster(), { avoid, lang: getLang() });
      if (pick === null || pick.players < others + 3) return;

      try {
        if (this.cfg.autoAvoidFile !== undefined) {
          const { addAvoid } = await import("./serverPick.ts");
          addAvoid(this.cfg.autoAvoidFile, here, 10 * 60_000);
        }
      } catch {

      }
      this.emit("event", t("здесь почти никого, перехожу на {name} ({n} игроков)", { name: pick.name, n: pick.players }));
      this.switchRequested?.();
    } catch (err) {
      this.log(`auto server: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  private emit(kind: BotLine["kind"], text: string, from?: string, sys?: boolean): void {
    if (this.sink !== null) {
      this.sink(sys === true ? { kind, text, from, sys } : { kind, text, from });
      return;
    }
    if (kind === "log" && !this.cfg.verbose) return;
    console.log(kind === "chat" ? `<${from ?? "?"}> ${text}` : `[bot ${new Date().toISOString()}] ${text}`);
  }

  private log(msg: string): void {
    this.emit("log", msg);
  }

  status(): BotStatus {
    const self = this.ownId >= 0 ? this.world.getTee(this.ownId) : undefined;
    const target = this.targetId >= 0 ? this.world.getTee(this.targetId) : undefined;
    return {
      phase: this.phase,
      acting: this.acting,
      brain: this.cfg.planner === true ? "planner" : this.cfg.scripted === true ? "scripted" : "net",
      mode: this.mode,
      server: `${this.cfg.host}:${this.cfg.port}`,
      name: this.cfg.name,
      targetName: this.targetId >= 0 ? this.nameOfLive(this.targetId) : null,
      targetDist: self && target ? Math.round(vdistance(self.pos, target.pos)) : null,
      walk: this.nav === null ? null : this.follow !== null && this.follow.waiting ? `goto ${this.follow.name.slice(0, 12)} …` : this.nav.brief(self ?? null, this.follow?.name.slice(0, 12)),
      offlineReason: this.phase === "online" ? "" : this.lastDisconnect,
      wb: this.wbHolding() !== null && this.wbChooser.side !== null ? `WB ${this.wbChooser.side}` : null,
      panel: {
        wbMode: this.wbDef === null ? null : this.wbMode,
        duelMode: this.duelMode,
        inDuel: this.duelNow(),
        spectating: this.wantSpectate,
        pinnedTarget: this.cfg.targetName ?? null,
        home: this.home !== null,
        tryName: this.tryName,
        duelScore: this.duelScore === null ? null : { name: this.duelScore.name, ours: this.duelScore.ours, theirs: this.duelScore.theirs },
      },
      frozen: self?.frozen ?? false,
      tick: this.world.tick,
      stats: this.stats,
      lowCpu: this.lowCpu,
      strong: this.strong,
      lag: this.lag.summary(),
    };
  }

  private tryName = "off";

  private readonly baseCfg: Record<string, unknown> = {};

  private readonly startCfg: Record<string, unknown> = {};

  private resetControllers(): void {
    this.planner?.reset();
    this.cfg.policy?.reset();
  }

  private maybeAcceptDuel(msg: TwMessage, who: string): void {
    if (this.cfg.acceptDuels === false) return;
    const text = msg.message.toLowerCase();
    const invited = /\/accept|accept|challeng|duel|дуэл|вызов|вызвал/.test(text);

    const already = /(?<![\p{L}\p{N}_)/-])(?:accepted|started|ended|won|lost|declined|declines|declining)(?![\p{L}\p{N}_])|отклонил|начал|закончил|проиграл|победил|выиграл|принял|окончен|завершен(?:а|о)?(?!\p{L})/u.test(text);
    if (!invited || already) return;
    const now = Date.now();
    if (now - this.lastAcceptMs < DUEL_ACCEPT_COOLDOWN_MS) return;
    this.lastAcceptMs = now;

    this.answeredInvitation(msg.message);
    if (this.say("/accept")) this.emit("event", `duel invitation from ${who}: answered /accept`);
    else this.emit("event", `duel invitation from ${who}: could not answer, chat cooldown`);
  }

  private maybeAnswerAccusation(msg: TwMessage): void {
    if (this.cfg.brushOff !== true) return;
    if (!ACCUSATION.test(msg.message)) return;
    const now = Date.now();
    const last = this.lastReplyMs.get(msg.client_id) ?? 0;
    if (now - last < REPLY_COOLDOWN_MS) return;
    this.lastReplyMs.set(msg.client_id, now);
    const pool = /[а-яё]/i.test(msg.message) || !/[a-z]/i.test(msg.message) ? BRUSH_OFFS : BRUSH_OFFS_EN;
    const text = pool[Math.floor(this.wanderRng.nextFloat() * pool.length)];
    const delay = REPLY_DELAY_MS + Math.floor(this.wanderRng.nextFloat() * REPLY_JITTER_MS);
    this.emit("event", `'${this.nameOfLive(msg.client_id)}' asked; answering '${text}' in ${(delay / 1000).toFixed(1)}s`);
    const timer = setTimeout(() => {
      this.replyTimers.delete(timer);
      if (!this.stopping) this.say(text);
    }, delay);
    this.replyTimers.add(timer);
  }

  private onTickReset(newTick: number): void {

    if (this.world.tick > 0) this.log(`server game tick restarted (${this.world.tick} -> ${newTick}); clearing tick-keyed state`);
    this.lastKillTick = -Infinity;
    this.routeKillTick = -1_000_000;
    this.sent = [];
    this.wanderUntilTick = 0;
    this.wanderJumpUntilTick = 0;
    this.wanderHookUntilTick = 0;
    this.stuckAnchor = null;
    this.stuckAnchorTick = newTick;
    this.frozenSince = -1;
    this.planSim = null;

    this.travelSinceTick = -1000;
    this.dullSinceTick = -1;
    this.idleSinceTick = -1;
    this.lastClipTick = -Infinity;
    this.lastCrossClipTick = -Infinity;
    this.path = null;
    if (this.trek !== null) this.endTrek();
    this.frozenSinceById.clear();
    this.lastInputById.clear();
    this.atUsById.clear();
    this.atFriendById.clear();
    this.lastTouch.clear();
    this.thawTickById.clear();
    this.lastPosById.clear();
    this.rescueWalkTick = -Infinity;
    this.rescueSaidTick = -Infinity;
    this.rescueHammerSince = -1;
    this.rescuePausedUntil.clear();
    this.rescuePullSince = -1;
    this.rescuePullAt = -Infinity;
    this.pullNone.clear();
  }

  private nameOfLive(id: number): string {

    const info = this.client?.SnapshotUnpacker?.AllObjClientInfo?.find((c) => c.id === id);
    return info?.name ?? `#${id}`;
  }

  voteOptions(): string[] {
    try {
      const list = (this.client as unknown as { VoteOptionList?: unknown } | undefined)?.VoteOptionList;
      return Array.isArray(list) ? list.filter((v): v is string => typeof v === "string" && v !== "") : [];
    } catch {
      return [];
    }
  }

  handleConsole(line: string): string {
    const trimmed = line.trim();
    if (trimmed.length === 0) return "";

    if (!COMMAND_PREFIXES.includes(trimmed[0])) {
      if (!this.say(trimmed)) return "not sent: chat cooldown, try again in a moment";

      if (/^\/accept\b/i.test(trimmed)) {
        this.duelAnsweredMs = Date.now();
        this.duelChallenger = "";
      }
      return "";
    }
    const [cmd, ...rest] = trimmed.slice(1).split(/\s+/);
    const arg = rest.join(" ");
    switch (cmd.toLowerCase()) {
      case "help":
        return [
          "  type anything          say it in the game chat, as the bot",
          "",
          "  !stop / !go            stop playing and stand still / resume",
          "  !war [name|off]        fight them on sight  ·  !friend [name|off]  never touch them, defend them, hammer them out of the freeze",
          "  !ignore [name|off]     never touch them, never answer them",
          "  !clanwar [clan|off]    the same by clan tag  ·  !clanfriend [clan|off]",
          "  !home [x y|off]        mark a spot to return to when there is nobody to fight",
          "  !wb [off|left|right|auto]  hold the wayblock (Copy Love Box): auto takes the emptier side and keeps it",
          "  !duel [on|off|auto]    1 on 1: no WB, no walks; auto turns it on when a duel it accepted starts",
          "  !style default|wb|duel the window's three ways to play",
          "  !clip [note]           save the last 30s to a file for review",
          "  !log on|off            show every debug line, or just the status bar",
          "  !mode <name>           fight (default) | passive (never engage) | hold",
          "  !try <name>|off        switch a candidate planner setting on mid-game",
          "  !target <nick>         fight only this player, '!target -' to clear",
          "  !brain <name>          planner | net | scripted, swapped live",
          "",
          "  !goto tele             walk to the nearest teleporter",
          "  !goto <x> <y>          walk to that tile; '!goto' alone reports progress",
          "  !goto <nick>           walk to that player, following them ('!goto @nick' if it looks like a command)",
          "",
          "  !stats / !where        counters / position, target and freeze state",
          "  !emote <name>          " + Object.keys(EMOTICON_BY_NAME).join(", "),
          "  !reset, !kill          kill and respawn",
          "  !yes, !no              vote on the running vote, like F3 / F4",
          "  !votes / !vote <text>  list the server's votes / call the one whose name matches",
          "  !spec / !join          go to the spectators / back into the game",
          "  !lang ru|en            the language of the console and the bot's window",
          "  !low on|off            the mode for a weak PC: a shorter search, a new plan every 2 snapshots",
          "  !strong on|off         the strong mode: on the WB a search three times the size (more CPU)",
          "  !quit                  disconnect and exit",
          "",
          "  '?' works too. Neither prefix ever reaches the server.",
        ].join("\n");
      case "mode": {
        const want = arg.toLowerCase();
        if (want === "" ) return `mode: ${this.mode} (fight | passive | hold)`;
        if (want !== "fight" && want !== "passive" && want !== "hold") return "!mode fight | passive | hold";
        const gave = this.dropNav("?mode");
        this.mode = want;
        this.acting = want !== "hold";
        if (want !== "fight") this.targetId = -1;
        return `${gave}mode: ${want}`;
      }
      case "goto":
        return this.gotoCommand(arg);
      case "stop":

        if (this.nav !== null) return this.cancelNav("cancelled");
        this.mode = "hold";
        this.acting = false;
        return "stopped";
      case "go": {
        const gave = this.dropNav("?go");
        this.mode = "fight";
        this.acting = true;
        return `${gave}playing`;
      }

      case "yes":
      case "f3":
      case "no":
      case "f4": {
        const yes = cmd.toLowerCase() === "yes" || cmd.toLowerCase() === "f3";
        try {
          this.client?.game.Vote(yes);
        } catch {
          return "not connected";
        }
        return yes ? "voted yes (F3)" : "voted no (F4)";
      }
      case "votes": {
        const list = this.voteOptions();
        return list.length === 0 ? "the server offers no votes" : list.map((v, i) => `${i + 1}. ${v}`).join("\n");
      }
      case "vote": {
        const list = this.voteOptions();
        const want = arg.trim();
        if (want === "") return "!vote <name>; !votes lists them";
        const n = Number.parseInt(want, 10);
        const low = want.toLowerCase();
        const pick =
          (String(n) === want && n >= 1 && n <= list.length ? list[n - 1] : undefined) ??
          list.find((v) => v.toLowerCase() === low) ??
          list.find((v) => v.toLowerCase().includes(low));
        if (pick === undefined) return `no vote matches "${want}"; !votes lists them`;
        try {
          this.client?.game.CallVoteOption(pick, "");
        } catch {
          return "not connected";
        }
        return `called the vote: ${pick}`;
      }
      case "spec":
      case "join": {
        const spec = cmd.toLowerCase() === "spec";
        try {
          this.client?.game.SetTeam(spec ? -1 : 0);
        } catch {
          return "not connected";
        }

        this.wantSpectate = spec;
        return spec ? "went to the spectators" : "joined the game";
      }
      case "kill":
      case "reset": {
        if (this.world.tick - this.lastKillTick < KILL_COOLDOWN_TICKS) return "reset is on cooldown";
        this.lastKillTick = this.world.tick;
        this.stuckAnchor = null;
        this.stats.selfKills++;
        try {
          this.client?.game.Kill();
        } catch {
          return "the server refused /kill";
        }
        return "killed, respawning";
      }
      case "target": {
        if (arg === "" || arg === "-") {
          this.cfg.targetName = undefined;
          return "target cleared, back to picking automatically";
        }

        const names = (this.client?.SnapshotUnpacker?.AllObjClientInfo ?? []).map((c) => (c.name ?? "").trim()).filter((n) => n !== "");
        const exact = names.find((n) => foldName(n) === foldName(arg));
        const hits = exact !== undefined ? [exact] : this.playersMatching(arg);
        if (hits.length > 1) return `target: '${arg}' matches ${hits.length} players: ${hits.join(", ")} -- be more specific`;
        const name = hits.length === 1 ? hits[0] : arg;
        this.cfg.targetName = name;
        return hits.length === 1 ? `target set to '${name}'` : `target set to '${name}' (nobody by that name is on the server now)`;
      }
      case "brain": {
        const want = arg.toLowerCase();
        if (want === "planner") {
          if (this.planner === undefined) return "no planner in this build";
          this.cfg.planner = true;
          this.cfg.scripted = false;
        } else if (want === "net") {
          if (this.cfg.policy === undefined) return "no policy was loaded (start with --policy)";
          this.cfg.planner = false;
          this.cfg.scripted = false;
        } else if (want === "scripted") {
          this.cfg.planner = false;
          this.cfg.scripted = true;
        } else {
          return "!brain planner | net | scripted";
        }
        this.resetControllers();
        return `brain: ${want}`;
      }

      case "try": {
        const want = arg.trim().toLowerCase();
        if (want === "" || want === "?") {
          return `!try ${Object.keys(TRY_SETTINGS).join(" | ")} | off   (now: ${this.tryName})`;
        }
        if (want === "off" || want === "-") {
          this.tryName = "off";
          this.cfg.plannerCfg = { ...this.baseCfg };
          this.planner = null;
          return "try: off, back to the defaults";
        }
        const found = TRY_SETTINGS[want];
        if (found === undefined) return `!try ${Object.keys(TRY_SETTINGS).join(" | ")} | off`;
        this.tryName = want;
        this.cfg.plannerCfg = { ...this.baseCfg, ...found };

        this.planner = null;
        return `try: ${want} -> ${JSON.stringify(found)}`;
      }
      case "stats":
        return this.statsLine();
      case "lang": {
        const want = arg.trim().toLowerCase();
        if (want === "") return `lang: ${getLang()} (ru | en)`;
        if (!isLang(want)) return "!lang ru | en";
        setLang(want);
        this.saveLang(want);
        return want === "ru" ? "язык: русский" : "language: English";
      }
      case "low": {
        const want = arg.trim().toLowerCase();
        const state = (on: boolean): string =>
          on ? `mode for a weak PC: on (search ${LOW_CPU.budgetMs} ms, a new plan every ${LOW_CPU.commit} snapshots, fewer route checks)` : "mode for a weak PC: off (the full search)";
        if (want === "") return `${state(this.lowCpu)}; !low on | off`;
        if (want !== "on" && want !== "off") return "!low on | off";

        this.setLowCpu(want === "on");
        this.saveLowCpu(this.lowCpu);
        if (this.lowCpu) this.saveStrong(false);
        return state(this.lowCpu);
      }
      case "strong": {
        const want = arg.trim().toLowerCase();
        const state = (on: boolean): string =>
          on
            ? (this.cfg.plannerCfg?.population ?? PLANNER_DEFAULTS.population) >= STRONG_WB.population
              ? "strong mode: on -- this brain's own search is already as big: nothing changes"
              : `strong mode: on (on the WB a search of ${STRONG_WB.population} x ${STRONG_WB.iterations}, up to ${STRONG_WB.budgetMs} ms a decision: more CPU)`
            : "strong mode: off (the plain search)";
        if (want === "") return `${state(this.strong)}; !strong on | off`;
        if (want !== "on" && want !== "off") return "!strong on | off";
        if (want === "on" && this.lowCpu) {
          this.setLowCpu(false);
          this.saveLowCpu(false);
        }
        this.setStrong(want === "on");
        this.saveStrong(this.strong);
        return state(this.strong);
      }
      case "log": {

        const want = arg.toLowerCase();
        if (want !== "on" && want !== "off") return "!log on | off";
        this.emit("event", `log lines: ${want}`);
        return `verbose logging ${want}`;
      }
      case "war":
        return this.relationCommand("war", arg, "war");
      case "friend":
        return this.relationCommand("friend", arg, "friends");
      case "ignore":
        return this.relationCommand("ignore", arg, "ignored");
      case "clanwar":
        return this.relationCommand("clanWar", arg, "clan war");
      case "clanfriend":
        return this.relationCommand("clanFriend", arg, "friendly clans");
      case "home": {
        const self = this.ownId >= 0 ? this.world.getTee(this.ownId) : undefined;
        if (arg.toLowerCase() === "off") {
          this.home = null;
          if (this.wbDef !== null && this.wbMode !== "off") return "home cleared: on this map it holds the WB when there is nobody to fight (!wb off for neither)";
          return "home cleared: it will stay wherever the fight is";
        }
        const parts = arg.split(/[\s,]+/).filter((x) => x !== "");
        if (parts.length === 2 && Number.isFinite(Number(parts[0])) && Number.isFinite(Number(parts[1]))) {
          this.home = { tx: Math.trunc(Number(parts[0])), ty: Math.trunc(Number(parts[1])) };
          this.homeMap = this.mapName();
        } else if (parts.length === 0) {
          if (self === undefined) return "no tee yet -- stand somewhere first, or !home <x> <y>";
          this.home = { tx: Math.trunc(self.pos.x / 32), ty: Math.trunc(self.pos.y / 32) };
          this.homeMap = this.mapName();
        } else {
          return "!home            mark where you are standing\n!home <x> <y>    mark a tile\n!home off        forget it";
        }
        return `home set to tile (${this.home.tx},${this.home.ty}); it walks back there after ${GO_HOME_AFTER_TICKS / 50}s with nobody to fight${this.wbDef !== null && this.wbMode !== "off" ? " (and does not hold the WB while it is set)" : ""}`;
      }
      case "wb":
        return this.wbCommand(arg);
      case "duel": {
        const want = arg.trim().toLowerCase();
        if (want === "") return `duel: ${this.duelMode}${this.duelNow() ? ", in one now" : this.duelSeen ? ", one is on screen (ignored: off)" : ""}`;
        if (want !== "on" && want !== "off" && want !== "auto") return "!duel on | off | auto";
        this.duelMode = want;
        const gave = want === "on" && this.nav !== null ? `${this.cancelNav("duel on")}; ` : "";
        if (want === "on" && this.trek !== null) this.endTrek();
        return want === "on" ? `${gave}duel: on -- fights whoever is there, no WB, no walks` : want === "off" ? "duel: off -- plays as usual even in a duel" : "duel: auto -- on by itself once a duel it accepted starts";
      }

      case "style": {
        const want = arg.trim().toLowerCase();
        if (want === "") return `style: ${this.duelNow() ? "duel" : this.wbDef !== null && this.wbMode !== "off" ? "wb" : "default"}`;
        if (want === "duel") return this.handleConsole("!duel on");
        if (want === "wb") {
          this.duelMode = "auto";
          return this.wbCommand(this.wbMode === "left" || this.wbMode === "right" ? this.wbMode : "auto");
        }
        if (want === "default") {
          this.duelMode = "auto";
          return this.wbCommand("off");
        }
        return "!style default | wb | duel";
      }
      case "clip": {

        if (this.ownId < 0) return "no tee yet";
        const rec = this.clipRing.toRecording({ map: this.mapName(), controller: this.brainName(), selfId: this.ownId, label: arg === "" ? undefined : arg });
        if (rec === null) return "nothing recorded yet";
        const file = this.writeClip(rec, `manual-${this.world.tick}${arg === "" ? "" : `-${arg}`}`);
        return file === null ? "could not write the clip" : `saved ${(rec.frames.length / 25).toFixed(0)}s to ${file}`;
      }
      case "where": {
        const st = this.status();
        const self = this.ownId >= 0 ? this.world.getTee(this.ownId) : undefined;
        const at = self === undefined ? "" : `tile (${Math.trunc(self.pos.x / 32)},${Math.trunc(self.pos.y / 32)}), `;
        const walk = this.nav === null ? "" : `, goto: ${this.nav.progress(self ?? null)}`;
        return `${st.phase}, ${at}${st.acting ? "playing" : "stopped"}, ${st.frozen ? "frozen" : "free"}, target ${st.targetName ?? "none"}${st.targetDist === null ? "" : ` at ${st.targetDist}px`}${st.wb ? `, ${st.wb}` : ""}, tick ${st.tick}${walk}`;
      }
      case "emote": {
        const id = EMOTICON_BY_NAME[arg.toLowerCase()];
        if (id === undefined) return `unknown emote; try ${Object.keys(EMOTICON_BY_NAME).join(", ")}`;
        this.lastEmoteMs = 0;
        this.emote(id);
        return "";
      }
      case "quit":

        if (this.quitRequested !== null) this.quitRequested();
        else void this.stop();
        return "disconnecting";
      default:
        return `unknown command '${cmd}' -- try !help`;
    }
  }

  private wbCommand(arg: string): string {
    const want = arg.trim().toLowerCase();
    if (want === "") {
      if (this.wbDef === null) return `WB: '${this.mapName()}' has none -- ${WAYBLOCK_NAMES} do`;
      const held = this.wbHolding();
      const side = this.wbChooser.side;
      const pausedMin = Math.ceil((this.wbPausedUntilMs - Date.now()) / 60_000);
      const why =
        this.wbMode === "off" ? "off" : this.home !== null ? "not held while !home is set (!home off)" : pausedMin > 0 ? `${this.wbMode}, left alone for ${pausedMin} more min after dying on the way in (!wb ${this.wbMode} to go now)` : held === null ? `${this.wbMode}, not held in mode ${this.mode}` : `${this.wbMode}, holding the ${side ?? "?"}`;
      return `WB: ${why}; playing: ${this.wbCounts.left} on the left, ${this.wbCounts.right} on the right`;
    }
    const mode = want === "on" ? "auto" : want;
    if (mode !== "off" && mode !== "left" && mode !== "right" && mode !== "auto") return "!wb off | left | right | auto";
    this.wbMode = mode;

    if (mode !== "off") {
      this.wbPausedUntilMs = 0;
      this.wbWalkFails = 0;
    }
    let gave = "";

    if (mode === "off" && this.wbWalk && this.nav !== null) gave = `${this.cancelNav("the WB is off")}; `;
    if (this.wbDef === null) return `${gave}WB: ${mode} (this map has none; it applies on ${WAYBLOCK_NAMES})`;
    if (mode === "off") return `${gave}WB: off -- it stays wherever the fight is`;
    const home = this.home !== null ? " (not held while !home is set: !home off)" : "";
    return mode === "auto" ? `WB: auto -- the side with fewer players playing, the one it stands on in a tie, then held${home}` : `WB: the ${mode} side${home}`;
  }

  private gotoCommand(arg: string, navOpts: { throughFreeze?: boolean; crossings?: readonly Crossing[] } = {}): string {
    const text = arg.trim();

    const named = text.startsWith("@");
    const word = text.toLowerCase();
    if (!named && (word === "stop" || word === "-" || word === "off")) {
      const wasQueued = this.pendingGoto !== null;
      this.pendingGoto = null;
      if (this.nav === null) return wasQueued ? "queued walk dropped" : "not going anywhere";
      return this.cancelNav("cancelled");
    }
    const col = this.world.collision;
    const self = this.ownId >= 0 ? this.world.getTee(this.ownId) : undefined;

    if (text !== "" && (!this.collisionReady || self === undefined || !self.alive)) {
      this.pendingGoto = text;
      const why = !this.collisionReady ? "the map is still loading" : "the tee has not spawned";
      return `queued: ${why}, starting as soon as it is ready`;
    }
    if (!this.collisionReady) return "no map yet: without a collision grid there is nowhere to walk to";

    if (text === "") {
      if (this.nav !== null) return this.follow !== null ? this.followProgress(self ?? null) : this.nav.progress(self ?? null);
      if (self === undefined || !self.alive) return "nothing set, and the tee is not alive yet";
      if (!col.hasTele()) return `nothing set. This map has no teleport layer, so '?goto <x> <y>' (tiles, 0..${col.width - 1} by 0..${col.height - 1}) or '?goto <nick>' are the forms`;
      const goals = teleGoals(col, self.pos.x, self.pos.y);
      if (goals.length === 0) return "nothing set, and no teleporter on this map can be reached from here by walking";
      return `nothing set. '?goto tele' would walk to the ${goals[0].label}`;
    }

    if (self === undefined || !self.alive) return "the tee is not alive; try again once it has spawned";
    if (named) return this.gotoPlayer(text.slice(1), self, navOpts);

    if (word === "tele" || word === "teleport" || word === "tp") {
      if (!col.hasTele()) return "this map has no teleport layer";
      const goals = teleGoals(col, self.pos.x, self.pos.y);
      if (goals.length === 0) return "no teleporter on this map can be reached from here by walking";
      return this.startNav(goals, navOpts);
    }

    const parts = text.split(/[\s,]+/);
    const tx = parts.length === 2 && /^-?\d+$/.test(parts[0]) ? Number.parseInt(parts[0], 10) : Number.NaN;
    const ty = parts.length === 2 && /^-?\d+$/.test(parts[1]) ? Number.parseInt(parts[1], 10) : Number.NaN;

    if (!Number.isInteger(tx) || !Number.isInteger(ty)) return this.gotoPlayer(text, self, navOpts);
    if (tx < 0 || ty < 0 || tx >= col.width || ty >= col.height) {
      return `(${tx},${ty}) is off the map -- it is ${col.width}x${col.height} tiles`;
    }

    const px = tx * 32 + 16;
    const py = ty * 32 + 16;
    if (col.isSolid(px, py)) return `(${tx},${ty}) is a wall`;
    if (col.isDeath(px, py)) return `(${tx},${ty}) is a death tile`;
    if (col.isFreeze(px, py)) return `(${tx},${ty}) is freeze -- walking into it on purpose is not a plan`;
    return this.startNav([tileGoal(col, tx, ty)], navOpts);
  }

  private gotoPlayer(text: string, self: TeeState, navOpts: { throughFreeze?: boolean; crossings?: readonly Crossing[] }): string {
    const want = text.trim();
    if (want === "") return `whose name? ${GOTO_USAGE}`;
    const cards = (this.client?.SnapshotUnpacker?.AllObjClientInfo ?? []).map((c) => ({ id: c.id, name: c.name ?? "" }));
    const hit = matchPlayer(cards, want, this.ownId);
    if ("none" in hit) return `nobody named '${want}' is on the server. ${GOTO_USAGE}`;
    if ("self" in hit) return `'${hit.self}' is the bot itself`;
    if ("many" in hit) return t('{label}: "{what}" -- это {n}: {list}. Уточни.', { label: "goto", what: want, n: hit.many.length, list: hit.many.join(", ") });
    const tee = this.world.getTee(hit.id);
    if (this.world.notPlaying(hit.id)) return `${hit.name} is not in the game (spectating or paused)`;
    if (tee === undefined || !tee.alive) return `${hit.name} has no tee on the map right now`;
    if (vdistance(self.pos, tee.pos) <= FOLLOW_ARRIVED_PX) return `already next to ${hit.name}`;
    const goal = this.followTile(tee.pos) ?? { tx: Math.trunc(tee.pos.x / 32), ty: Math.trunc(tee.pos.y / 32) };
    const reply = this.startNav([this.followGoal(goal, hit.name)], navOpts);
    const tick = this.world.tick;
    this.follow = {
      id: hit.id,
      name: hit.name,
      tx: goal.tx,
      ty: goal.ty,
      routedTick: tick,
      seenTick: tick,
      startTick: tick,
      best: vdistance(self.pos, tee.pos),
      bestTick: tick,
      fails: 0,
      blockedAt: -1,
      deaths: 0,
      selfDeaths: 0,
      last: { x: tee.pos.x, y: tee.pos.y },
      waiting: false,
      navOpts,
    };
    return `${reply}, following them as they move; ${this.afterArrival(hit.id, hit.name)}`;
  }

  private afterArrival(id: number, name: string): string {
    const back = this.navReturnMode;
    if (back !== "fight") return `nobody is touched on the way, and it goes back to ${back} there, which fights nobody`;
    const card = this.client?.SnapshotUnpacker?.AllObjClientInfo?.find((c) => c.id === id);
    const nameKey = (card?.name ?? name).trim().toLowerCase();
    const clanKey = (card?.clan ?? "").trim().toLowerCase();
    if (this.onList("friend", nameKey, id) || this.clanFriend(clanKey, id)) return `${name} is a friend, never touched on the way or after`;
    if (this.onList("ignore", nameKey, id)) return `${name} is ignored, never touched on the way or after`;
    if (this.onList("war", nameKey, id) || listed(this.relations.clanWar, clanKey)) return `not touched on the way; after arriving ${name} is fought, being on the war list`;
    const tee = this.world.getTee(id);
    if (tee !== undefined && this.afk(tee)) return `not touched on the way, nor after while ${name} is away from the keyboard`;
    return `not touched on the way; after arriving it is back to fight, and ${name} is a target there like anybody else who is playing`;
  }

  private followTile(pos: Vec2): { tx: number; ty: number } | null {
    const col = this.world.collision;
    const cx = Math.trunc(pos.x / 32);
    const cy = Math.trunc(pos.y / 32);
    const free = (tx: number, ty: number): boolean => {
      if (tx < 0 || ty < 0 || tx >= col.width || ty >= col.height) return false;
      const px = tx * 32 + 16;
      const py = ty * 32 + 16;
      return !col.isSolid(px, py) && !col.isFreeze(px, py) && !col.isDeath(px, py);
    };
    let best: { tx: number; ty: number } | null = null;
    let bestD = Infinity;
    for (let oy = -FOLLOW_GOAL_TILES; oy <= FOLLOW_GOAL_TILES; oy++) {
      for (let ox = -FOLLOW_GOAL_TILES; ox <= FOLLOW_GOAL_TILES; ox++) {
        const d = ox * ox + oy * oy + (col.isSolid((cx + ox) * 32 + 16, (cy + oy + 1) * 32 + 16) ? 0 : 5);
        if (d < bestD && free(cx + ox, cy + oy)) {
          bestD = d;
          best = { tx: cx + ox, ty: cy + oy };
        }
      }
    }
    return best;
  }

  private followGoal(tile: { tx: number; ty: number }, name: string): NavGoal {
    return { tx: tile.tx, ty: tile.ty, label: `${name} at (${tile.tx},${tile.ty})`, tele: null };
  }

  private followProgress(self: TeeState | null): string {
    const f = this.follow;
    if (f === null || this.nav === null) return "not going anywhere";
    if (f.waiting) return `waiting for ${f.name}: no tee on the map for ${((this.world.tick - f.seenTick) / 50).toFixed(1)}s`;
    return `following ${f.name}: ${this.nav.progress(self)}`;
  }

  private steerFollow(self: TeeState): "go" | "wait" | "over" {
    const f = this.follow;
    const nav = this.nav;
    if (f === null || nav === null) return "go";
    const tick = this.world.tick;
    const end = (why: string): "over" => {
      const back = this.navReturnMode;
      this.endNav();
      this.emit("event", `goto: ${why} -- back to ${back}`);
      return "over";
    };

    if (tick < f.startTick) {
      f.startTick = f.routedTick = f.seenTick = f.bestTick = tick;
      if (f.blockedAt >= 0) f.blockedAt = tick;
    }
    if (tick - f.startTick > FOLLOW_MAX_TICKS) return end(`gave up on ${f.name} after ${FOLLOW_MAX_TICKS / 50}s of walking`);
    if (f.selfDeaths >= FOLLOW_MAX_DEATHS) return end(`died ${f.selfDeaths} times on the way to ${f.name}; giving up`);
    if (f.deaths >= FOLLOW_MAX_DEATHS) return end(`${f.name} died ${f.deaths} times before it got to them; giving up`);

    const card = this.client?.SnapshotUnpacker?.AllObjClientInfo?.find((c) => c.id === f.id);
    if (card === undefined || foldName(card.name ?? "") !== foldName(f.name)) return end(`${f.name} left the server`);
    const tee = this.world.getTee(f.id);
    const away = this.world.notPlaying(f.id);
    if (tee === undefined || !tee.alive || away) {
      if (tick - f.seenTick > FOLLOW_LOST_TICKS) return end(`${f.name} is not in the game (${away ? "spectating or paused" : "no tee on the map"})`);
      f.waiting = true;
      return "wait";
    }
    f.waiting = false;
    const jumped = f.last !== null && vdistance(f.last, tee.pos) > FOLLOW_JUMP_PX;
    f.seenTick = tick;
    f.last = { x: tee.pos.x, y: tee.pos.y };
    const d = vdistance(self.pos, tee.pos);
    if (d <= FOLLOW_ARRIVED_PX && !self.frozen) {
      const said = this.afterArrival(f.id, f.name);
      return end(`arrived at ${f.name}; ${said}`);
    }
    if (d < f.best - 32) {
      f.best = d;
      f.bestTick = tick;
      f.fails = 0;
    } else if (tick - f.bestTick > FOLLOW_STALL_TICKS) {
      return end(`no closer to ${f.name} in ${FOLLOW_STALL_TICKS / 50}s; giving up`);
    }
    const goal = this.followTile(tee.pos);
    const moved = goal !== null && Math.hypot(goal.tx - f.tx, goal.ty - f.ty) * 32 >= PATH_MOVED_PX;
    let reroute = false;
    if (nav.phase === "blocked") {

      if (f.blockedAt < 0) {
        f.blockedAt = tick;
        f.fails++;
        if (f.fails >= FOLLOW_MAX_FAILS) return end(`no way to ${f.name} from here: ${nav.outcome}`);
        this.emit("event", `goto: no way to ${f.name} yet (${nav.outcome}); trying again as they move`);
      }
      if (!(goal !== null && (goal.tx !== f.tx || goal.ty !== f.ty)) && tick - f.blockedAt < FOLLOW_RETRY_TICKS) return "wait";
      reroute = true;
    } else if (nav.phase === "arrived" && !jumped && goal !== null && goal.tx === f.tx && goal.ty === f.ty) {

      return end(`as near ${f.name} as it gets without the freeze; ${this.afterArrival(f.id, f.name)}`);
    } else if (nav.phase === "arrived" || jumped) {

      reroute = true;
    } else if (moved && tick - f.routedTick >= PATH_REFRESH_TICKS) {
      reroute = true;
    }
    if (reroute && goal !== null) {
      this.nav = new Navigator(this.world.collision, [this.followGoal(goal, f.name)], this.navOptsFor(f.navOpts));
      f.tx = goal.tx;
      f.ty = goal.ty;
      f.routedTick = tick;
      f.blockedAt = -1;
    }

    return this.nav === null || this.nav.done ? "wait" : "go";
  }

  private startNav(goals: NavGoal[], navOpts: { throughFreeze?: boolean; crossings?: readonly Crossing[] } = {}): string {
    if (this.nav !== null) this.emit("event", "goto: replaced by a new destination");

    if (this.mode !== "goto") this.navReturnMode = this.mode;

    this.seekingGame = false;

    this.follow = null;
    this.wbWalk = false;
    this.nav = new Navigator(this.world.collision, goals, this.navOptsFor(navOpts));
    this.mode = "goto";
    this.acting = true;
    this.targetId = -1;
    const rest = goals.length > 1 ? `, then ${goals.length - 1} more candidate${goals.length > 2 ? "s" : ""} if it is not a doorway` : "";
    return `goto: ${goals[0].label}${rest}`;
  }

  private navOptsFor(navOpts: { throughFreeze?: boolean; crossings?: readonly Crossing[] }): { throughFreeze?: boolean; crossings?: readonly Crossing[] } {
    if (navOpts.crossings !== undefined || this.wbDef === null) return navOpts;
    return { ...navOpts, crossings: this.wbDef.crossings };
  }

  private endNav(): void {
    this.nav = null;
    this.follow = null;
    this.rescueId = -1;
    this.seekingGame = false;
    this.wbWalk = false;
    this.mode = this.navReturnMode;
    this.acting = this.mode !== "hold";
    if (this.mode !== "fight") this.targetId = -1;
  }

  private cancelNav(why: string): string {
    this.nav?.cancel(why);
    const back = this.navReturnMode;
    this.endNav();
    return `goto: ${why}, back to ${back}`;
  }

  private dropNav(by: string): string {
    if (this.nav === null) return "";
    this.nav.cancel(`dropped by ${by}`);
    this.nav = null;
    this.follow = null;
    this.rescueId = -1;
    this.wbWalk = false;
    return `goto: dropped by ${by}. `;
  }

  private driveNav(client: TwClient, self: TeeState): void {
    if (this.nav === null) return;

    if (this.world.collision !== this.nav.collision) {
      this.emit("event", "goto: the map changed under it, giving up");
      this.endNav();
      this.idle();
      return;
    }
    const following = this.follow !== null;
    if (following && this.steerFollow(self) !== "go") {
      this.idle();
      return;
    }

    const nav = this.nav;
    if (nav === null) return;
    nav.crossBudgetMs = this.navBudgetMs;
    const want = nav.step(
      self,
      this.world.tick,
      this.world.allTees().filter((t) => t.id !== self.id && t.alive),
      this.lagTicks(),
    );

    const input = nav.crossing || nav.plannedFreeze ? want : this.guard(self, want);

    if (input !== want) nav.vetoed();

    for (const note of nav.takeNotes()) {
      if (following) this.log(`goto: ${note}`);
      else this.emit("event", `goto: ${note}`);
      if (/; trying again from the spawn$|^no way through /.test(note)) this.clipCrossFail(note);
    }

    if (nav.takeKill()) {
      if (this.world.tick - this.lastKillTick >= KILL_COOLDOWN_TICKS) {
        this.lastKillTick = this.world.tick;
        this.routeKillTick = this.world.tick;
        this.stats.selfKills++;
        this.emit("event", "goto: the way there starts with a respawn -> /kill");
        try {
          client.game.Kill();
        } catch {

        }
      } else {
        this.emit("event", "goto: the way there starts with a respawn, but /kill is on cooldown");
      }
    }
    this.applyInput(client, input, self.activeWeapon);

    if (nav.done && !following) {
      const back = this.navReturnMode;
      this.endNav();
      this.emit("event", `goto: ${back === "hold" ? "standing by" : `back to ${back}`}`);
    }
  }

  private connect(): void {
    const client = this.client;
    if (!client || this.stopping) return;
    if (client.State !== teeworlds.Protocol.States.STATE_OFFLINE) return;
    this.phase = "connecting";
    this.log(`connecting to ${this.cfg.host}:${this.cfg.port} as '${this.cfg.name}'`);
    client.connect().catch((err: unknown) => {
      this.stats.errors++;
      this.emit("event", `could not connect to ${this.cfg.host}:${this.cfg.port}: ${String(err)}`);
      this.onDisconnect(`connect failed: ${String(err)}`, false);
    });
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer || this.stopping) return;
    const delay = this.reconnectDelayMs;
    this.reconnectDelayMs = Math.min(this.reconnectDelayMs * 2, RECONNECT_MAX_MS);
    this.log(`reconnecting in ${delay}ms`);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = undefined;
      this.connect();
    }, delay);
  }

  private onConnected(): void {
    const client = this.client;
    if (!client) return;
    this.phase = "online";
    this.reconnectDelayMs = RECONNECT_MIN_MS;
    this.ownId = -1;
    this.targetId = -1;
    this.lastPos = null;
    this.wasAlive = false;
    this.wantSpectate = false;

    this.endDuelScore();
    this.duelSeen = false;
    this.duelForSince = -1;
    this.duelAnsweredMs = 0;
    this.duelChallenger = "";
    this.armAcceptUntil = 0;
    this.navPending = this.cfg.goto !== undefined && this.cfg.goto.trim().length > 0;
    client.movement.FlagPlaying(true);
    client.movement.SetAim(0, -1);
    this.idle();

    client.game.Say("/showall 1");
    this.log(`connected (map '${client.map?.map_name ?? "?"}')`);
    if (!this.collisionReady) this.refreshCollision();
    const waiter = this.startWaiter;
    this.startWaiter = undefined;
    waiter?.resolve();
  }

  private onDisconnect(reason: string, fromServer: boolean): void {
    this.endDuelScore();

    for (const t of this.replyTimers) clearTimeout(t);
    this.replyTimers.clear();
    this.lastReplyMs.clear();
    this.lastSeenDist.clear();
    this.planSim = null;

    this.sent = [];

    if (fromServer && this.cfg.autoServer === true && /\bban/i.test(reason) && this.switchRequested !== null && !this.stopping) {
      const here = `${this.cfg.host}:${this.cfg.port}`;
      void import("./serverPick.ts")
        .then(({ addAvoid }) => {
          if (this.cfg.autoAvoidFile !== undefined) addAvoid(this.cfg.autoAvoidFile, here, 60 * 60_000);
        })
        .catch(() => {})
        .finally(() => {
          this.emit("event", t("на {addr} не пускает, ищу другой сервер", { addr: here }));
          this.switchRequested?.();
        });
    }

    if (this.phase === "offline") return;
    this.phase = "offline";
    this.lastPos = null;

    this.emit("event", `disconnected${fromServer ? " BY THE SERVER" : ""}: ${reason || "(no reason given)"}`);
    this.lastDisconnect = reason || (fromServer ? "kicked, no reason given" : "connection lost");
    if (this.stopping) return;
    this.stats.disconnects++;
    if (this.cfg.reconnect === false) {
      const waiter = this.startWaiter;
      this.startWaiter = undefined;
      waiter?.reject(new Error(`connection to ${this.cfg.host}:${this.cfg.port} lost: ${reason}`));
      return;
    }
    this.scheduleReconnect();
  }

  private onKill(kill: TwKill): void {

    this.lastTouch.delete(kill.victim_id);
    this.thawTickById.delete(kill.victim_id);
    if (this.ownId < 0) return;

    const seen = this.lastInputById.get(kill.victim_id);
    if (seen !== undefined) seen.settleUntil = this.world.tick + INPUT_SETTLE_TICKS;

    if (this.follow !== null) {
      if (kill.victim_id === this.follow.id) this.follow.deaths++;
      if (kill.victim_id === this.ownId && this.world.tick - this.lastKillTick > 50) this.follow.selfDeaths++;
    }
    if (kill.victim_id === this.ownId) {
      this.stats.deaths++;

      if (this.wbWalk && this.world.tick - this.routeKillTick > 50) this.noteWbWalkDeath();

      this.wasAlive = false;
      this.lastPos = null;
      this.wasFrozen = false;
      this.emote(kill.killer_id === this.ownId ? EMOTICON_ZZZ : EMOTICON_SORRY);
    } else if (kill.killer_id === this.ownId) {
      this.stats.kills++;
      this.emote(EMOTICON_SPLATTEE);
    } else if (kill.victim_id === this.targetId) {

      this.emote(EMOTICON_QUESTION);
    }
  }

  private emote(emoticon: number): void {
    if (this.cfg.emotes === false) return;
    const now = Date.now();
    if (now - this.lastEmoteMs < EMOTE_COOLDOWN_MS) return;
    this.lastEmoteMs = now;
    try {
      this.client?.game.Emote(emoticon);
    } catch {

    }
  }

  private onMessage(msg: TwMessage): void {

    if (msg.client_id < 0) {
      this.emit("chat", msg.message, t("сервер"), true);
      const answer = this.autoChat.onChat({ from: "", text: msg.message, server: true, me: this.cfg.name });
      if (answer !== null) this.autoSay(answer, msg.message);
      return;
    }
    if (msg.client_id === this.ownId) {
      this.emit("chat", msg.message, this.cfg.name);
      return;
    }

    if (msg.team === CHAT_WHISPER_SENT) {
      this.emit("chat", `[→ ${this.nameOfLive(msg.client_id)}] ${msg.message}`, this.cfg.name);
      return;
    }
    const who = this.nameOfLive(msg.client_id);
    const whisper = msg.team === CHAT_WHISPER_RECV;

    const mentioned = msg.message.toLowerCase().includes(this.cfg.name.toLowerCase());
    if (whisper) this.emit("whisper", msg.message, who);
    else this.emit("chat", `${msg.team === CHAT_TEAM ? "(team) " : ""}${mentioned ? "*" : ""}${msg.message}`, who);

    if (this.onList("ignore", who.trim().toLowerCase(), msg.client_id)) return;
    const answer = this.autoChat.onChat({ from: who, text: msg.message, server: false, me: this.cfg.name });
    if (answer !== null) this.autoSay(answer, whisper ? msg.message : null);
    if (whisper) this.maybeAcceptDuel(msg, who);
    this.maybeAnswerAccusation(msg);
    if (!this.cfg.chat) return;
    const text = msg.message.trim().toLowerCase();
    const me = this.cfg.name;
    if (text === "!bot") {
      this.say(`${me}: DDNet AI bot (${this.cfg.scripted ? "scripted baseline" : "neural policy"}); commands: !bot !stop !go !stats !reset !hi`);
      this.emote(EMOTICON_HEARTS);
    } else if (text === "!stop") {
      this.acting = false;
      this.say(`${me}: stopped (say !go to resume)`);
    } else if (text === "!go") {
      this.acting = true;
      this.say(`${me}: resumed`);
    } else if (text === "!stats") {
      this.say(`${me}: ${this.statsLine()}`);
    } else if (text.startsWith("!try")) {

      this.say(`${me}: ${this.handleConsole(text).split("\n")[0]}`);
    } else if (text === "!hi" || text === "!hello" || text === "!привет") {

      this.lastEmoteMs = 0;
      this.emote(EMOTICON_HEARTS);
      this.say(`${me}: hi`);
    } else if (text === "!reset") {

      if (this.world.tick - this.lastKillTick < KILL_COOLDOWN_TICKS) {
        this.say(`${me}: reset is on cooldown`);
        return;
      }
      this.lastKillTick = this.world.tick;
      this.stuckAnchor = null;
      this.stats.selfKills++;
      this.say(`${me}: resetting (kill + respawn)`);
      try {
        this.client?.game.Kill();
      } catch {

      }
    }
  }

  private say(text: string): boolean {
    const now = Date.now();
    if (now - this.lastChatMs < CHAT_MIN_INTERVAL_MS) return false;
    if (this.client === undefined || this.client === null) return false;
    this.lastChatMs = now;
    this.client.game.Say(text);
    return true;
  }

  private refreshCollision(): void {
    const client = this.client;
    if (!client) return;
    const now = Date.now();
    if (now - this.lastCollisionTryMs < COLLISION_RETRY_MS) return;
    this.lastCollisionTryMs = now;
    const collision = mapCollisionFromClient(client, this.cfg.mapDir);
    if (!collision) {
      this.log(
        `no collision grid for map '${client.map?.map_name ?? "?"}' (downloaded ${client.map?.mapBuffer?.length ?? 0} bytes, fallback ${this.cfg.mapDir}); idling`,
      );
      return;
    }
    this.world.setCollision(collision);
    this.collisionReady = true;
    const hadWb = this.wbDef;
    this.wbDef = wayblockFor(this.mapName(), collision);

    this.wbWalkFails = 0;
    this.wbPauses = 0;
    this.wbPausedUntilMs = 0;
    if (this.wbDef === null && hadWb === null && wayblockFor(this.mapName()) !== null) {
      this.emit("event", `WB: '${this.mapName()}' here is another version of the map the WB was measured on (its size, a spot or the tube's rope anchors differ); not holding it`);
    } else if (this.wbDef !== null && hadWb !== this.wbDef) {
      this.emit("event", `WB: this map has one; ${this.wbMode === "off" ? "not holding it (!wb off)" : this.home !== null ? "not holding it while !home is set" : "holding it when there is nobody to fight (!wb off to stop)"}`);
    }

    this.clipRing.clear();
    this.clipMapSet = false;

    const spawns = spawnTiles(collision);
    this.deadCells = spawns.length === 0 ? null : { width: collision.width, cells: deadZone(collision, spawns) };
    let deadCount = 0;
    if (this.deadCells !== null) for (const v of this.deadCells.cells) deadCount += v;
    this.planner?.setDeadZone(this.deadCells);
    this.saveMemory();
    this.memoryPath = this.memoryFile();
    this.memory = FreezeMemory.load(this.memoryPath, collision.width, collision.height);
    this.memoryDirty = false;
    this.planner?.setFreezeMemory(this.memory);
    this.log(
      `collision grid ready: ${collision.width}x${collision.height} tiles` +
        (this.deadCells === null ? "" : `, ${deadCount} of them no way back from (${spawns.length} spawns)`),
    );
  }

  private queueSnapshot(): void {
    const tick = this.client?.currentSnapshotGameTick;
    this.snapTick = typeof tick === "number" ? tick : undefined;
    if (this.snapArrived === 0) this.snapFirstRead = performance.now();
    this.snapArrived++;
    if (this.snapQueued) return;
    this.snapQueued = true;
    setImmediate(() => {
      this.snapQueued = false;
      const arrived = this.snapArrived;
      const firstRead = this.snapFirstRead;
      this.snapArrived = 0;
      if (this.stopping || this.client === null) return;
      const t0 = performance.now();
      this.onSnapshot({ tick: this.snapTick });
      let line = this.lag.note(firstRead, arrived, t0, performance.now());

      if (line !== null && this.strong) line += " -- or first the strong mode off (!strong off)";
      if (line !== null) this.log(line);
    });
  }

  private plannerCfgNow(): PlannerConfig {
    const base: PlannerConfig = { budgetMs: LIVE_BUDGET_MS, explain: true, ...this.cfg.plannerCfg };
    if (!this.lowCpu) return base;
    const budget = base.budgetMs ?? LIVE_BUDGET_MS;
    return {
      ...base,
      budgetMs: budget > 0 ? Math.min(budget, LOW_CPU.budgetMs) : LOW_CPU.budgetMs,
      hardMs: base.hardMs !== undefined && base.hardMs > 0 ? Math.min(base.hardMs, LOW_CPU.hardMs) : LOW_CPU.hardMs,
      commitDecisions: Math.max(base.commitDecisions ?? 1, LOW_CPU.commit),
      explain: false,

      shieldCadence: true,
    };
  }

  setLowCpu(on: boolean): void {
    if (on) this.strong = false;
    if (this.lowCpu === on) return;
    this.lowCpu = on;
    this.planner = null;
    if (this.nav !== null) this.nav.crossBudgetMs = this.navBudgetMs;
  }

  private get navBudgetMs(): number {
    return this.lowCpu ? LOW_CPU.navMs : 0;
  }
  private get reachBudget(): number {
    return this.lowCpu ? LOW_CPU.reachChecks : -1;
  }

  get lowCpuOn(): boolean {
    return this.lowCpu;
  }

  setStrong(on: boolean): void {
    this.strong = on && !this.lowCpu;
  }

  get strongOn(): boolean {
    return this.strong;
  }

  private onSnapshot(saved?: { tick: number | undefined }): void {
    const client = this.client;
    if (!client) return;
    try {
      this.stats.ticks++;
      this.reachLeft = this.reachBudget;
      if (!this.collisionReady) this.refreshCollision();

      const snap = client.SnapshotUnpacker;
      const ownId = snap.OwnID;
      if (ownId === undefined) {
        this.idle();
        return;
      }
      this.ownId = ownId;

      const autoLine = this.autoChat.due(this.cfg.name);
      if (autoLine !== null) this.autoSay(autoLine, Date.now() < this.armAcceptUntil ? this.armAcceptLine : null);

      const tick = saved !== undefined ? saved.tick : typeof client.currentSnapshotGameTick === "number" ? client.currentSnapshotGameTick : undefined;
      if (tick !== undefined && tick < this.world.tick) this.onTickReset(tick);
      this.world.updateFromSnapshot(snap, ownId, tick, client.rawSnapUnpacker?.deltas);

      this.refreshInputClock();

      this.updateDuel(ownId);

      const self = this.world.getTee(ownId);
      if (!self || !self.alive) {

        this.lastPos = null;
        this.wasAlive = false;

        this.wasFrozen = false;
        this.maybeJoinGame(snap, ownId);
        this.idle();
        return;
      }
      if (!this.wasAlive) {

        this.wasAlive = true;
        this.cfg.policy?.reset();
        this.prevInput = emptyInput();

        this.sent = [];

        this.wanderUntilTick = 0;
        this.wanderJumpUntilTick = 0;
        this.wanderHookUntilTick = 0;

        if (this.pendingGoto !== null && this.collisionReady) {
          const want = this.pendingGoto;
          this.pendingGoto = null;
          this.emit("event", `starting the queued walk: ${this.gotoCommand(want)}`);
        }

        this.nav?.respawned();
      }
      this.trackTravel(self.pos);

      if (this.memory !== null && self.alive && !self.frozen) {
        const tile = Math.trunc(self.pos.y / 32) * this.world.collision.width + Math.trunc(self.pos.x / 32);
        if (tile !== this.lastTileIndex) {
          this.lastTileIndex = tile;
          this.memory.notePass(self.pos.x, self.pos.y);
          this.memoryDirty = true;
        }
      }
      this.measureEcho(self);
      this.recordFrame(self);
      this.maybeLogStatus(self);
      this.maybeUnstick(client, self);

      if (self.frozen !== this.wasFrozen) {
        this.emote(self.frozen ? EMOTICON_DROP : EMOTICON_EXCLAMATION);

        if (self.frozen && this.memory !== null) {
          this.memory.note(self.pos.x, self.pos.y);
          this.memoryDirty = true;
          if (this.memory.noted % 20 === 0) this.saveMemory();
        }
        this.wasFrozen = self.frozen;
      }

      if (!this.collisionReady || !this.acting) {
        this.idle();
        return;
      }

      this.updateWbSide(ownId, self);

      if (this.wbWalk && this.nav !== null && !self.frozen && this.wbDef !== null && this.wbChooser.side !== null) {

        const goal = this.nav.goal ?? null;
        const tx = Math.trunc(self.pos.x / 32);
        const ty = Math.trunc(self.pos.y / 32);
        if (goal !== null && inWbHall(this.wbDef, this.wbChooser.side, tx, ty) && ty - goal.ty >= WB_NO_CLIMB_TILES) {
          this.log(this.cancelNav(`below the WB spot (${goal.tx},${goal.ty}) inside the hall: no climb up to it`));
          this.idleSinceTick = this.world.tick - WB_RETURN_TICKS - 1;
        }
      }

      if ((this.wbWalkFails > 0 || this.wbPauses > 0) && !self.frozen && this.wbDef !== null) {
        const tx = Math.trunc(self.pos.x / 32);
        const ty = Math.trunc(self.pos.y / 32);
        if (inWbHall(this.wbDef, "left", tx, ty) || inWbHall(this.wbDef, "right", tx, ty)) {
          this.wbWalkFails = 0;
          this.wbPauses = 0;
        }
      }

      if (this.navPending && this.nav === null) {
        this.navPending = false;
        const reply = this.gotoCommand(this.cfg.goto ?? "");
        this.emit("event", `--goto ${this.cfg.goto}: ${reply}`);
      }

      if (this.trek !== null && this.someoneWorthFighting(ownId, self.pos, SEEK_ARRIVED_PX)) {
        this.endTrek();
        this.log("found a game on the way; stopping the walk");
      }

      let picked: number | undefined;

      const walkingTo = this.rescueId >= 0 ? this.world.getTee(this.rescueId) : undefined;
      if (this.nav !== null && this.rescueId >= 0 && (!this.rescuable(self, walkingTo) || this.inFreezeTiles(walkingTo.pos))) {
        this.log("the friend it was walking to needs no rescue now; ending the walk");
        this.endNav();
      }
      if (this.nav !== null) {

        if (
          this.seekingGame &&
          (!this.wbWalk || this.wbWalkCuttable(self)) &&
          this.someoneWorthFighting(ownId, self.pos, SEEK_ARRIVED_PX) &&
          (picked = this.pickTarget(snap, ownId, self.pos)) !== -1
        ) {
          this.seekingGame = false;
          this.endNav();
          this.log("found a game on the way; stopping the walk");
        } else {
          this.driveNav(client, self);
          return;
        }
      }

      const targetId = this.mode === "fight" ? (picked ?? this.pickTarget(snap, ownId, self.pos)) : -1;
      if (targetId !== this.targetId) {
        this.targetId = targetId;

        if (targetId === -1) this.log("no target in reach");
      }

      if (
        this.mode === "fight" &&

        targetId !== -1 &&
        this.seekEnabled() &&

        this.wbHolding() === null &&
        this.nav === null &&
        this.world.tick - this.travelSinceTick > TRAVEL_RETRY_TICKS &&
        !this.engagedNow(ownId, self) &&

        !this.someoneWorthFighting(ownId, self.pos, SEEK_ARRIVED_PX)
      ) {
        const here = this.crowdAt(ownId, self.pos);
        const spot = this.gameSpot(ownId, self.pos);
        if (spot !== null && spot.tees + spot.busy >= here.tees + here.busy + SEEK_MARGIN) {
          if (this.dullSinceTick < 0) this.dullSinceTick = this.world.tick;
          if (this.world.tick - this.dullSinceTick > SEEK_PATIENCE_TICKS) {
            this.travelSinceTick = this.world.tick;
            this.dullSinceTick = -1;
            const tx = Math.trunc(spot.x / 32);
            const ty = Math.trunc(spot.y / 32);
            const reply = this.startTrek(self.pos, spot);
            this.log(
              `here: ${here.tees} tees, ${here.busy} fighting; there: ${spot.tees}/${spot.busy} at (${tx},${ty}), ${Math.round(spot.dist)}px -- ${reply}`,
            );

          }
        } else this.dullSinceTick = -1;
      }

      if (targetId === -1) {

        if (this.trek !== null) this.endTrek();
        if (this.idleSinceTick < 0) this.idleSinceTick = this.world.tick;
        if (!this.duelNow() && this.rescueFriend(client, self)) return;
        const holding = this.wbHolding();

        if (holding === null && this.nav === null && !this.duelNow() && this.world.tick - this.travelSinceTick > TRAVEL_RETRY_TICKS) {
          const spot = this.gameSpot(ownId, self.pos);
          if (spot !== null) {
            this.travelSinceTick = this.world.tick;
            const tx = Math.trunc(spot.x / 32);
            const ty = Math.trunc(spot.y / 32);

            const way = findRoute(this.world.collision, self.pos, spot, { nearTiles: 3, partial: false, allowKill: true, throughFreeze: false, maxNodes: REACH_MAX_NODES });
            if (way === null) {
              this.log(`nobody in reach; the game is at (${tx},${ty}) but there is no way there without freeze -- staying`);
            } else {
              const reply = this.gotoCommand(`${tx} ${ty}`, { throughFreeze: false });

              this.seekingGame = this.nav !== null && this.navReturnMode === "fight";
              this.log(`nobody in reach; walking to where the game is: (${tx},${ty}), ${Math.round(spot.dist)}px, ${spot.tees} tees, ${spot.busy} of them fighting -- ${reply}`);
            }
          }
        }
        if (this.home !== null && this.nav === null && !this.duelNow() && this.world.tick - this.idleSinceTick > GO_HOME_AFTER_TICKS) {
          const here = { tx: Math.trunc(self.pos.x / 32), ty: Math.trunc(self.pos.y / 32) };
          if (Math.abs(here.tx - this.home.tx) > 2 || Math.abs(here.ty - this.home.ty) > 2) {
            const reply = this.gotoCommand(`${this.home.tx} ${this.home.ty}`);
            this.emit("event", `nobody to fight: walking home to (${this.home.tx},${this.home.ty}) -- ${reply}`);
          }
          this.idleSinceTick = this.world.tick;
        } else if (holding !== null && this.nav === null && this.world.tick - this.idleSinceTick > WB_RETURN_TICKS) {
          this.walkToWb(ownId, self);
          this.idleSinceTick = this.world.tick;
        }

        const side = holding === null ? null : this.wbChooser.side;
        const spot = holding === null || side === null ? null : this.wbSpot(ownId, holding, side, { tx: Math.trunc(self.pos.x / 32), ty: Math.trunc(self.pos.y / 32) });
        const near = spot !== null && side !== null && holding !== null && inWbHall(holding, side, Math.trunc(self.pos.x / 32), Math.trunc(self.pos.y / 32));
        const watch = near ? sideDef(holding, side).watch : null;
        this.wander(client, self, near ? spot.tx * 32 + 16 : undefined, watch === null ? undefined : { x: watch.tx * 32 + 16, y: watch.ty * 32 + 16 });
        return;
      }
      this.idleSinceTick = -1;

      if (!this.duelNow() && this.rescueFriend(client, self, true)) return;

      let input: PlayerInput;
      if (this.cfg.planner) {
        input = this.planAction(ownId, targetId, self);
        this.applyInput(client, input, self.activeWeapon);
        return;
      }
      encodeObs(this.world, ownId, targetId, this.obs);
      if (this.cfg.scripted || !this.cfg.policy) {
        input = scriptedAction(this.world, ownId, targetId, this.prevInput, this.rng);
      } else {
        this.raw.set(this.cfg.policy.act(this.obs));
        input = decodeAction(this.raw, this.prevInput, undefined, !isGrounded(this.world, self));
      }

      input = this.guard(self, input);
      this.applyInput(client, input, self.activeWeapon);
    } catch (err) {
      this.stats.errors++;
      this.log(`snapshot error: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
    }
  }

  private maybeJoinGame(snap: TwSnapshotUnpacker, ownId: number): void {

    if (this.wantSpectate) return;
    const info = snap.getObjPlayerInfo(ownId);
    if (!info || info.team !== -1) return;
    const now = Date.now();
    if (now - this.lastJoinMs < JOIN_RETRY_MS) return;
    this.lastJoinMs = now;
    this.log("in spectators, joining the game");
    this.client?.game.SetTeam(0);

    this.sendIdentity();
  }

  private sendIdentity(): void {
    const game = this.client?.game;
    if (game === undefined || typeof game.ChangePlayerInfo !== "function") return;
    try {
      game.ChangePlayerInfo({
        name: this.cfg.name,
        clan: this.cfg.clan ?? "",
        country: this.cfg.country ?? -1,
        skin: this.cfg.skin ?? "default",
        use_custom_color: this.cfg.colorBody !== undefined || this.cfg.colorFeet !== undefined,
        color_body: this.cfg.colorBody ?? 0,
        color_feet: this.cfg.colorFeet ?? 0,
      });
    } catch {

    }
  }

  liveMap(): LiveMap | null {
    if (!this.collisionReady) return null;
    const c = this.world.collision;
    const kinds = new Uint8Array(c.width * c.height);
    for (let y = 0; y < c.height; y++) {
      for (let x = 0; x < c.width; x++) {
        const px = x * 32 + 16;
        const py = y * 32 + 16;

        kinds[y * c.width + x] = c.isSolid(px, py)
          ? c.isNoHook(px, py)
            ? 5
            : 1
          : c.isFreeze(px, py)
            ? 2
            : c.isDeath(px, py)
              ? 3
              : c.isUnFreeze(px, py)
                ? 4
                : c.teleAt(px, py).type > 0
                  ? 6
                  : 0;
      }
    }
    return {
      name: this.mapName(),
      width: c.width,
      height: c.height,
      kinds: Buffer.from(kinds).toString("base64"),
      traps: this.deadCells === null ? undefined : Buffer.from(this.deadCells.cells).toString("base64"),
    };
  }

  liveFrame(): LiveFrame | null {
    if (!this.collisionReady || this.ownId < 0) return null;
    const tees: LiveTee[] = [];

    const snap = this.client?.SnapshotUnpacker;
    const cards = new Map((snap?.AllObjClientInfo ?? []).map((c) => [c.id, c]));
    const chars = new Map((snap?.AllObjCharacter ?? []).map((c) => [c.client_id, c]));
    for (const t of this.world.allTees()) {
      if (!t.alive) continue;
      const card = cards.get(t.id);
      const ch = chars.get(t.id);
      const ex = snap?.getObjExDDNetCharacter(t.id);
      tees.push({
        id: t.id,
        name: this.nameOfLive(t.id),
        x: Math.round(t.pos.x),
        y: Math.round(t.pos.y),
        frozen: t.frozen,
        hook: t.hookState,
        hx: Math.round(t.hookPos.x),
        hy: Math.round(t.hookPos.y),
        hooked: t.hookedPlayer,
        clan: card?.clan ?? "",
        skin: card?.skin ?? "default",
        cc: (card?.use_custom_color ?? 0) !== 0,
        cb: card?.color_body ?? 0,
        cf: card?.color_feet ?? 0,
        aim: Math.round(wireAngleRad(t.angle) * 100) / 100,
        wp: t.activeWeapon,
        emote: ch?.emote ?? 0,
        vx: Math.round(t.vel.x * 1000) / 1000,
        vy: Math.round(t.vel.y * 1000) / 1000,
        dir: t.direction,
        jumped: t.jumped,
        atk: Math.max(0, this.world.tick - t.attackTick),
        fz: t.freezeTicksLeft,
        pf: ch?.player_flags ?? 0,
        jl: t.jumpsLeft,
        fzf: t.frozenFor,
        deep: t.deepFrozen === true,
        xf: ex?.m_Flags,
        jt: ex?.m_Jumps,
      });
    }
    const players: LivePlayer[] = [];
    for (const c of cards.values()) {
      const p = snap?.getObjPlayerInfo(c.id);
      players.push({
        id: c.id,
        name: c.name,
        clan: c.clan,
        score: p?.score ?? 0,
        ping: p?.latency ?? 0,
        team: p?.team ?? 0,
        skin: c.skin ?? "default",
        cc: (c.use_custom_color ?? 0) !== 0,
        cb: c.color_body ?? 0,
        cf: c.color_feet ?? 0,
      });
    }
    const gameInfo = (snap as { AllObjGameInfo?: { round_start_tick?: number }[] } | undefined)?.AllObjGameInfo?.[0];

    const exInfo = (snap as { AllObjExGameInfo?: { m_Flags?: number }[] } | undefined)?.AllObjExGameInfo?.[0];
    const goal = this.planner?.travelGoal ?? null;
    const steps = this.trek !== null ? this.trek.steps.slice(this.trek.at) : (this.path?.steps ?? []);
    const self = this.world.getTee(this.ownId);
    const doing =
      self === undefined || !self.alive
        ? t("мёртв")
        : self.frozen
          ? this.inDeadZone(self.pos)
            ? t("во фризе в кармане без выхода")
            : t("во фризе")
          : this.nav !== null
            ? this.navDoing(self)
            : this.trek !== null
            ? t("идёт туда, где игра ({n} шагов)", { n: this.trek.steps.length - this.trek.at })
            : this.targetId >= 0
              ? goal !== null
                ? t("идёт к цели в обход ({n} тайлов до точки)", { n: Math.round(vdistance(self.pos, goal) / 32) })
                : t("дерётся")
              : this.wbHolding() !== null && this.wbChooser.side !== null
                ? this.wbChooser.side === "left"
                  ? t("держит ВБ слева")
                  : t("держит ВБ справа")
                : t("цели нет");
    return {
      tick: this.world.tick,
      selfId: this.ownId,
      target: this.targetId,
      map: this.mapName(),
      tees,
      doing,
      goal: goal === null ? null : { x: Math.round(goal.x), y: Math.round(goal.y) },
      route: steps.slice(0, 40).map((s) => ({ x: s.x, y: s.y, kind: s.kind })),
      cursor: cursorOf(this.prevInput),
      players,
      roundStart: typeof gameInfo?.round_start_tick === "number" ? gameInfo.round_start_tick : undefined,
      timeScore: exInfo === undefined ? undefined : ((exInfo.m_Flags ?? 0) & 1) !== 0,
      mapKey: this.mapKey(),
    };
  }

  private navDoing(self: TeeState): string {
    const nav = this.nav;
    if (nav === null) return t("цели нет");
    const left = nav.tilesLeft(self);
    const n = left < 0 ? "?" : left;
    const f = this.follow;
    if (f !== null) return f.waiting ? t("ждёт {name}", { name: f.name }) : t("идёт к {name} ({n} тайлов)", { name: f.name, n });
    const goal = nav.goal;
    if (this.wbWalk) return this.wbChooser.side === "right" ? t("идёт на ВБ справа ({n} тайлов)", { n }) : t("идёт на ВБ слева ({n} тайлов)", { n });
    if (this.seekingGame || goal === null) return t("идёт туда, где игра ({n} тайлов)", { n });
    return t("идёт в ({x},{y}) ({n} тайлов)", { x: goal.tx, y: goal.ty, n });
  }

  private nameOf(snap: TwSnapshotUnpacker, id: number): string {
    return snap.AllObjClientInfo.find((c) => c.id === id)?.name ?? "?";
  }

  private nearFreeze(pos: Vec2): boolean {
    const c = this.world.collision;
    for (let oy = -SEAL_NEAR_TILES; oy <= SEAL_NEAR_TILES; oy++) {
      for (let ox = -SEAL_NEAR_TILES; ox <= SEAL_NEAR_TILES; ox++) {
        if (c.isFreeze(pos.x + ox * 32, pos.y + oy * 32)) return true;
      }
    }
    return false;
  }

  private guard(self: TeeState, input: PlayerInput): PlayerInput {
    if (self.frozen || !self.alive || !this.collisionReady) return input;
    if (this.shieldSim === null || this.shieldSim.collision !== this.world.collision) {
      this.shieldSim = new SimWorld(this.world.collision, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
    }
    const sim = this.shieldSim;
    try {
      for (const other of sim.allTees()) if (other.id !== self.id) sim.removeTee(other.id);
      if (sim.getTee(self.id) === undefined) sim.addTee(self.id, self.pos);
      sim.applyTeeState(self.id, self);
      for (let t = 0; t < this.lagTicks(); t++) {
        sim.setInput(self.id, this.prevInput);
        sim.step();
      }
      if (escapeExists(sim, self.id, input, 2)) return input;
      return saferInput(sim, self.id, input, 2, new Map(), this.prevInput) ?? input;
    } catch {
      return input;
    }
  }

  private isFriendId(id: number): boolean {
    const card = this.client?.SnapshotUnpacker?.AllObjClientInfo?.find((c) => c.id === id);
    if (card === undefined) return false;
    return this.onList("friend", (card.name ?? "").trim().toLowerCase(), id) || this.clanFriend((card.clan ?? "").trim().toLowerCase(), id);
  }

  private reachable(from: Vec2, tee: TeeState): boolean {
    const w = this.world.collision.width;
    const a = Math.trunc(from.y / 32) * w + Math.trunc(from.x / 32);

    if (this.reachLeft >= 0 && this.world.tick !== this.reachWantTick) {
      this.reachWantTick = this.world.tick;
      this.checkWanted(from, a);
    }
    const b = Math.trunc(tee.pos.y / 32) * w + Math.trunc(tee.pos.x / 32);
    const seen = this.reachAnswers.get(tee.id);
    if (seen !== undefined && seen.from === a && seen.to === b && this.world.tick - seen.tick < REACH_ANSWER_TICKS && this.world.tick >= seen.tick) return seen.ok;
    if (this.reachLeft >= 0) {
      this.reachWanted.add(tee.id);
      return seen === undefined ? true : seen.ok;
    }
    return this.checkReach(from, a, tee, b);
  }

  private checkWanted(from: Vec2, a: number): void {
    const wanted = [...this.reachWanted];
    this.reachWanted.clear();
    const now = this.world.tick;

    const age = (id: number): number => {
      const t = this.reachAnswers.get(id)?.tick;
      return t === undefined || t > now ? -Infinity : t;
    };
    wanted.sort((x, y) => {
      const ax = age(x);
      const ay = age(y);
      return ax === ay ? x - y : ax < ay ? -1 : 1;
    });
    const w = this.world.collision.width;
    for (const id of wanted) {
      if (this.reachLeft <= 0) break;
      const tee = this.world.getTee(id);
      if (tee === undefined) continue;
      const b = Math.trunc(tee.pos.y / 32) * w + Math.trunc(tee.pos.x / 32);
      const seen = this.reachAnswers.get(id);
      if (seen !== undefined && seen.from === a && seen.to === b && now - seen.tick < REACH_ANSWER_TICKS && now >= seen.tick) continue;
      this.reachLeft--;
      this.checkReach(from, a, tee, b);
    }
  }

  private checkReach(from: Vec2, a: number, tee: TeeState, b: number): boolean {
    let ok = true;
    try {
      ok = findRoute(this.world.collision, from, tee.pos, { nearTiles: 3, partial: false, maxNodes: REACH_MAX_NODES, throughFreeze: false }) !== null;
    } catch {
      ok = true;
    }
    this.reachAnswers.set(tee.id, { tick: this.world.tick, from: a, to: b, ok });

    if (this.reachAnswers.size > 64) {
      for (const id of this.reachAnswers.keys()) if (this.world.getTee(id) === undefined) this.reachAnswers.delete(id);
      while (this.reachAnswers.size > 64) {
        let oldest = -1;
        let at = Infinity;
        for (const [id, v] of this.reachAnswers) {
          if (v.tick < at) {
            at = v.tick;
            oldest = id;
          }
        }
        this.reachAnswers.delete(oldest);
      }
    }
    return ok;
  }

  private isSealed(tee: TeeState): boolean {
    const seen = this.sealAnswers.get(tee.id);
    if (seen !== undefined && this.world.tick - seen.tick < SEAL_ANSWER_TICKS && this.world.tick >= seen.tick) return seen.sealed;
    if (this.sealSim === null || this.sealSim.collision !== this.world.collision) {
      this.sealSim = new SimWorld(this.world.collision, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
    }
    let sealed = false;
    try {
      sealed = sealedIn(this.sealSim, tee.id, tee, enemyInputFromSnapshot(tee));
    } catch {
      sealed = false;
    }
    this.sealAnswers.set(tee.id, { tick: this.world.tick, sealed });
    if (this.sealAnswers.size > 64) this.sealAnswers.clear();
    return sealed;
  }

  private refreshInputClock(): void {
    const tick = this.world.tick;
    const cards = this.client?.SnapshotUnpacker?.AllObjClientInfo;
    const names = cards !== undefined && cards.length > 0 ? new Map(cards.map((c) => [c.id, foldName(c.name ?? "")])) : null;
    if (names !== null) for (const id of [...this.lastInputById.keys()]) if (!names.has(id)) this.lastInputById.delete(id);
    const me = this.ownId >= 0 ? this.world.getTee(this.ownId) : undefined;
    const friends = this.world.allTees().filter((t) => t.alive && t.id !== this.ownId && this.isFriendId(t.id));
    for (const id of [...this.atUsById.keys()]) if (this.world.getTee(id) === undefined) this.atUsById.delete(id);
    for (const id of [...this.atFriendById.keys()]) if (this.world.getTee(id) === undefined) this.atFriendById.delete(id);

    for (const a of this.world.allTees()) {
      const was = this.lastPosById.get(a.id);
      if (was !== undefined && tick - was.tick <= 4 && Math.hypot(a.pos.x - was.x, a.pos.y - was.y) > TELEPORT_JUMP_PX) this.lastTouch.delete(a.id);
      this.lastPosById.set(a.id, { x: a.pos.x, y: a.pos.y, tick });
    }
    const roping = new Map<number, number[]>();
    for (const a of this.world.allTees()) {
      if (!a.alive || a.hookedPlayer < 0) continue;
      const on = roping.get(a.hookedPlayer);
      if (on === undefined) roping.set(a.hookedPlayer, [a.id]);
      else on.push(a.id);
    }
    for (const [victim, on] of roping) {
      const had = this.lastTouch.get(victim);
      this.lastTouch.set(victim, { by: had !== undefined && on.includes(had.by) ? had.by : Math.min(...on), tick });
    }
    for (const a of this.world.allTees()) {
      if (!a.alive) continue;
      const seenA = this.lastInputById.get(a.id);
      if (seenA === undefined || seenA.at === tick || a.attackTick === seenA.attack || a.activeWeapon !== WEAPON_HAMMER) continue;
      const r = wireAngleRad(a.angle);
      const hp = { x: a.pos.x + Math.cos(r) * HAMMER_REACH_AHEAD_PX, y: a.pos.y + Math.sin(r) * HAMMER_REACH_AHEAD_PX };
      for (const b of this.world.allTees()) if (b.id !== a.id && b.alive && vdistance(hp, b.pos) < HAMMER_REACH_PX) this.lastTouch.set(b.id, { by: a.id, tick });
    }
    for (const tee of this.world.allTees()) {
      if (!tee.alive) {
        this.thawTickById.delete(tee.id);
        this.lastTouch.delete(tee.id);
      } else if (!tee.frozen && this.frozenSinceById.has(tee.id)) this.thawTickById.set(tee.id, tick);
      if (!tee.alive || !tee.frozen) this.frozenSinceById.delete(tee.id);
      else if (!this.frozenSinceById.has(tee.id)) {
        this.frozenSinceById.set(tee.id, tick);

        if (tick - (this.thawTickById.get(tee.id) ?? -Infinity) > REFREEZE_TICKS) this.onFreezeOnset(tee, me, tick);
      }
      if (!tee.alive) continue;
      const name = names?.get(tee.id);
      const keys = tee.frozen ? -1 : inputKeysOf(tee);
      const seen = this.lastInputById.get(tee.id);
      if (seen === undefined || (name !== undefined && seen.name !== name)) {
        this.lastInputById.set(tee.id, {
          name: name ?? "",
          angle: tee.angle,
          attack: tee.attackTick,
          keys,
          at: tick,
          firstSeen: tick,
          changed: -1,

          settleUntil: 0,
        });
        continue;
      }
      if (seen.at === tick) continue;

      const changed = tee.angle !== seen.angle || tee.attackTick !== seen.attack || (keys >= 0 && seen.keys >= 0 && keys !== seen.keys);
      if (changed && tick >= seen.settleUntil) seen.changed = tick;
      if (me !== undefined && tee.id !== me.id && ((tee.attackTick !== seen.attack && vdistance(tee.pos, me.pos) < SWING_AT_US_PX) || tee.hookedPlayer === me.id)) this.atUsById.set(tee.id, tick);

      if (tee.id !== this.ownId && friends.some((f) => f.id !== tee.id && ((tee.attackTick !== seen.attack && !f.frozen && vdistance(tee.pos, f.pos) < SWING_AT_US_PX) || tee.hookedPlayer === f.id))) this.atFriendById.set(tee.id, tick);
      seen.angle = tee.angle;
      seen.attack = tee.attackTick;
      if (keys >= 0) seen.keys = keys;
      seen.at = tick;
    }
  }

  private inputIdle(t: TeeState, strict = false): boolean {
    const seen = this.lastInputById.get(t.id);
    if (seen === undefined) return strict;
    if (seen.changed < 0) return strict || this.world.tick - seen.firstSeen > AFK_TICKS;
    return this.world.tick - seen.changed > AFK_TICKS;
  }

  private afk(t: TeeState, strict = false): boolean {
    return this.world.serverAfk(t.id) || this.world.notPlaying(t.id) || this.inputIdle(t, strict);
  }

  private parkedInFreeze(t: TeeState): boolean {
    if (!t.frozen) return false;
    return t.deepFrozen === true || this.world.tick - (this.frozenSinceById.get(t.id) ?? this.world.tick) > CROWD_FROZEN_TICKS;
  }

  private spared(t: TeeState, card: TwClientInfo | undefined): boolean {
    const nameKey = (card?.name ?? "").trim().toLowerCase();
    const clanKey = (card?.clan ?? "").trim().toLowerCase();
    if (this.onList("ignore", nameKey, t.id)) return true;
    if (!t.frozen && (this.onList("friend", nameKey, t.id) || this.clanFriend(clanKey, t.id))) return true;
    if (this.outOfGame(t.id)) return true;
    const atWar = this.onList("war", nameKey, t.id) || listed(this.relations.clanWar, clanKey);

    return !atWar && !this.duelNow() && this.afk(t);
  }

  private pickTarget(snap: TwSnapshotUnpacker, ownId: number, selfPos: Vec2): number {
    if (this.cfg.targetName !== undefined) {

      const want = foldName(this.cfg.targetName);
      const info = snap.AllObjClientInfo.find((c) => foldName(c.name ?? "") === want);
      if (!info || info.id === ownId) return -1;
      const tee = this.world.getTee(info.id);
      return tee && tee.alive ? info.id : -1;
    }

    const me = this.world.getTee(ownId);
    let best = -1;
    let bestScore = -Infinity;

    this.refreshInputClock();
    const info = new Map(snap.AllObjClientInfo.map((c) => [c.id, c]));
    let keepSettled = false;

    const wb = this.wbHolding();
    const wbSide = wb === null ? null : this.wbChooser.side;

    const meInLeash = wb !== null && wbSide !== null && inWbHall(wb, wbSide, Math.trunc(selfPos.x / 32), Math.trunc(selfPos.y / 32));

    const ropeOnUs = this.world.allTees().some((t) => t.alive && t.id !== ownId && t.hookedPlayer === ownId && !this.isFriendId(t.id));
    for (const tee of this.world.allTees()) {
      if (tee.id === ownId || !tee.alive) continue;
      const card = info.get(tee.id);
      const nameKey = (card?.name ?? "").trim().toLowerCase();
      const clanKey = (card?.clan ?? "").trim().toLowerCase();

      if (this.onList("friend", nameKey, tee.id) || this.onList("ignore", nameKey, tee.id)) continue;
      if (this.clanFriend(clanKey, tee.id)) continue;
      const atWar = this.onList("war", nameKey, tee.id) || listed(this.relations.clanWar, clanKey);

      if (this.outOfGame(tee.id)) continue;

      if (!atWar && !this.duelNow() && this.afk(tee)) continue;
      const d = vdistance(selfPos, tee.pos);
      if (d > TARGET_MAX_PX) continue;
      let inWb = false;
      let counter = false;
      if (wb !== null && wbSide !== null) {
        const ttx = Math.trunc(tee.pos.x / 32);
        const tty = Math.trunc(tee.pos.y / 32);
        const roped = tee.hookedPlayer === ownId || me?.hookedPlayer === tee.id;

        const atUs = this.world.tick - (this.atUsById.get(tee.id) ?? -Infinity) < AT_US_MEMORY_TICKS;

        if (meInLeash && atUs && d <= TUNING.hookLength + COUNTER_REACH_PX && !inWbLeash(wb, wbSide, ttx, tty)) counter = true;

        if (!roped && !atWar && !counter && (meInLeash ? !inWbLeash(wb, wbSide, ttx, tty) : !atUs)) continue;
        inWb = inWbZone(wb, wbSide, ttx, tty);
      }

      if (this.trapCare() && this.inDeadZone(tee.pos) && !this.inDeadZone(selfPos)) continue;

      const frozenFor = tee.frozen ? this.world.tick - (this.frozenSinceById.get(tee.id) ?? this.world.tick) : 0;

      const sealed = (tee.frozen || (tee.id === this.targetId && this.nearFreeze(tee.pos))) && this.isSealed(tee);

      const wbFinish = WB_FINISH && wb !== null && wbSide !== null && meInLeash && tee.frozen && !sealed && inWb;
      const finishing = wbFinish || (tee.id === this.targetId && tee.frozen && !sealed && frozenFor <= FINISH_BLOCK_TICKS && this.nearFreeze(tee.pos));
      const settled =
        sealed || (!finishing && frozenFor > (this.cfg.plannerCfg?.settledFreezeTicks ?? PLANNER_DEFAULTS.settledFreezeTicks));

      if (settled) {
        if (tee.id === this.targetId) keepSettled = true;
        continue;
      }

      const outOfReach = d >= PATH_NEAR_PX && !atWar && tee.hookedPlayer !== ownId && me?.hookedPlayer !== tee.id && !this.reachable(selfPos, tee);
      let score = 0;
      if (atWar) score += 900;
      if (tee.hookedPlayer === ownId) score += 1000;
      if (me?.hookedPlayer === tee.id) score += 800;

      if (tee.id === this.targetId && tee.frozen && d < BLOCKING_RANGE_PX) score += this.cfg.plannerCfg?.blockHoldScore ?? PLANNER_DEFAULTS.blockHoldScore;
      if (finishing && d < BLOCKING_RANGE_PX) score += FINISH_BLOCK_SCORE;

      if (wbFinish && WB_THAW_URGENCY > 0) score += WB_THAW_URGENCY * (1 - Math.min(1, tee.freezeTicksLeft / 150));
      if (this.world.tick - tee.attackTick < AGGRESSOR_MEMORY_TICKS && d < AGGRESSOR_RANGE_PX) score += 500;
      if (this.world.tick - (this.atFriendById.get(tee.id) ?? -Infinity) < AT_US_MEMORY_TICKS) score += AT_FRIEND_SCORE;

      if (!ropeOnUs && !this.duelNow() && tee.hookedPlayer >= 0 && this.isPartnerNow(tee.hookedPlayer)) score += PARTNER_ROPED_SCORE;
      const prev = this.lastSeenDist.get(tee.id);
      if (prev !== undefined && d < prev - 1) score += 200;

      if (tee.id === this.targetId) {
        const hold = this.cfg.plannerCfg?.targetHold ?? TARGET_HOLD_SCORE;
        score += d <= ENGAGED_PX ? hold : hold * Math.max(0, 1 - (d - ENGAGED_PX) / HOLD_FADE_PX);
      }
      score -= d * TARGET_DIST_WEIGHT;
      if (outOfReach) score -= OUT_OF_REACH_SCORE;
      if (inWb) score += WB_ZONE_SCORE;
      this.lastSeenDist.set(tee.id, d);
      if (score > bestScore) {
        bestScore = score;
        best = tee.id;
      }
    }

    if (best === -1 && keepSettled) return this.targetId;
    return best;
  }

  private lastPlan: RecFrame["plan"] = undefined;

  private readonly aimLog: { tick: number; x: number; y: number }[] = [];
  private readonly echoMiss = new Float64Array(ECHO_MAX_TICKS + 1);
  private readonly echoSeen = new Float64Array(ECHO_MAX_TICKS + 1);
  private echoSamples = 0;
  private snapshotGap = 2;
  private lastSnapTick = -1;

  private recordFrame(self: TeeState): void {
    if (!this.clipMapSet && this.collisionReady) {
      this.clipRing.setMap(this.world.collision);
      this.clipMapSet = true;
    }
    if (!this.clipMapSet) return;

    const near = this.world
      .allTees()
      .filter((t) => t.id !== self.id && t.alive)
      .sort((a, b) => vdistance(a.pos, self.pos) - vdistance(b.pos, self.pos))
      .slice(0, 3);
    this.clipRing.push({
      tick: this.world.tick,
      tees: [snapTee(self), ...near.map(snapTee)],

      inputs: [snapInput(self.id, this.prevInput)],
      events: [],
      plan: this.lastPlan,
      walk: this.nav?.goal?.label,
      ...(this.nav !== null && (this.nav.crossing || this.nav.plannedFreeze) ? { plannedFreeze: true } : {}),
    });

    this.lastPlan = undefined;
    if (++this.framesSinceScan >= CLIP_SCAN_EVERY_FRAMES) {
      this.framesSinceScan = 0;
      this.maybeClip(self.id);
    }
  }

  private clipCrossFail(note: string): void {
    if (this.ownId < 0 || this.world.tick - this.lastCrossClipTick < CROSS_CLIP_COOLDOWN_TICKS) return;
    const rec = this.clipRing.toRecording({ map: this.mapName(), controller: this.brainName(), selfId: this.ownId });
    if (rec === null || rec.frames.length < 50) return;
    this.lastCrossClipTick = this.world.tick;
    const file = this.writeClip(rec, `cross-fail-${this.world.tick}`);
    if (file !== null) this.emit("event", `clip saved: cross-fail (${note}) -> ${file}`);
  }

  private maybeClip(selfId: number): void {
    if (this.world.tick - this.lastClipTick < CLIP_COOLDOWN_TICKS) return;
    const rec = this.clipRing.toRecording({ map: this.mapName(), controller: this.brainName(), selfId });
    if (rec === null || rec.frames.length < 50) return;
    const worst = mergeOverlapping(findIncidents(rec, { selfId }))
      .filter((i) => i.severity >= (CLIP_SEVERITY_BY_KIND[i.kind] ?? CLIP_SEVERITY))

      .filter((i) => i.tick >= rec.frames[Math.floor(rec.frames.length / 2)].tick)
      .sort((a, b) => b.severity - a.severity)[0];
    if (worst === undefined) return;
    this.lastClipTick = this.world.tick;
    const file = this.writeClip(rec, `${worst.kind}-${worst.tick}-s${Math.round(worst.severity)}`);
    if (file !== null) this.emit("event", `clip saved: ${worst.kind} (${worst.note}) -> ${file}`);
  }

  private writeClip(rec: Recording, name: string): string | null {
    try {
      const dir = this.cfg.clipDir ?? DEFAULT_CLIP_DIR;
      mkdirSync(dir, { recursive: true });
      const file = join(dir, `${name.replace(/[^a-zA-Z0-9_-]/g, "_")}.json`);
      const cards = this.client?.SnapshotUnpacker?.AllObjClientInfo ?? [];
      rec.players = cards.map((c) => ({
        id: c.id,
        name: c.name ?? "",
        clan: c.clan ?? "",
        skin: c.skin ?? "default",
        cc: (c.use_custom_color ?? 0) !== 0,
        cb: c.color_body ?? 0,
        cf: c.color_feet ?? 0,
      }));
      writeFileSync(file, JSON.stringify(rec));
      this.stats.clips++;
      this.pruneClips(dir);
      return file;
    } catch (err) {
      this.log(`could not write clip: ${err instanceof Error ? err.message : String(err)}`);
      return null;
    }
  }

  private pruneClips(dir: string): void {
    try {
      const auto: { file: string; kind: string; mtime: number; severity: number }[] = [];
      for (const name of readdirSync(dir)) {
        if (!name.endsWith(".json") || name.startsWith("manual-")) continue;
        const m = /^(.*)-\d+-s(\d+)\.json$/.exec(name);
        if (m === null) continue;
        const file = join(dir, name);
        auto.push({ file, kind: m[1], mtime: statSync(file).mtimeMs, severity: Number(m[2]) });
      }
      auto.sort((a, b) => b.mtime - a.mtime || b.severity - a.severity);
      const perKind = new Map<string, number>();
      let kept = 0;
      for (const c of auto) {
        const n = perKind.get(c.kind) ?? 0;
        if (kept < CLIP_KEEP && n < CLIP_KEEP_PER_KIND) {
          perKind.set(c.kind, n + 1);
          kept++;
          continue;
        }
        rmSync(c.file, { force: true });
        rmSync(c.file.replace(/\.json$/, ".html"), { force: true });
      }
    } catch (err) {
      this.log(`could not prune clips: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  private ensurePlanner(): Planner {
    if (this.planner === null) {
      this.planner = new Planner(this.plannerCfgNow());
      this.planner.setDeadZone(this.deadCells);
      this.planner.setFreezeMemory(this.memory);
      if (this.cfg.opponentDirNet) this.planner.setOpponentDirNet(this.cfg.opponentDirNet);
    }
    return this.planner;
  }

  duelList(): DuelRow[] {
    return readDuelFile(join(dirname(this.cfg.relationsFile ?? RELATIONS_FILE), "duels.json"));
  }

  clipList(): { name: string; size: number; when: number }[] {
    try {
      const dir = this.cfg.clipDir ?? DEFAULT_CLIP_DIR;
      return readdirSync(dir)
        .filter((f) => f.endsWith(".json"))
        .map((f) => {
          const st = statSync(join(dir, f));
          return { name: f, size: st.size, when: st.mtimeMs };
        })
        .sort((a, b) => b.when - a.when)
        .slice(0, 200);
    } catch {
      return [];
    }
  }

  clipPath(name: string): string | null {
    if (!/^[\w.-]+\.json$/.test(name)) return null;
    const dir = this.cfg.clipDir ?? DEFAULT_CLIP_DIR;
    const full = join(dir, name);
    return existsSync(full) ? full : null;
  }

  commandNames(): string[] {
    return [
      "stop","go","war","friend","ignore","clanwar","clanfriend","home","wb","clip","log","mode","try",
      "target","brain","goto","stats","where","emote","reset","kill","yes","no","votes","vote","spec","join","lang","quit","help","seek","say","duel","style","low","strong",
    ];
  }

  knobs(): { key: string; value: unknown; def: unknown; changed: boolean }[] {
    const over = (this.cfg.plannerCfg ?? {}) as Record<string, unknown>;
    return Object.entries(PLANNER_DEFAULTS)
      .map(([key, plannerDef]) => {

        const def = key in this.startCfg ? this.startCfg[key] : plannerDef;
        const value = key in over ? over[key] : def;
        return { key, value, def, changed: value !== def };
      })
      .sort((a, b) => (a.key < b.key ? -1 : 1));
  }

  resetKnobs(): string {
    this.cfg.plannerCfg = { ...this.startCfg } as BotConfig["plannerCfg"];
    for (const k of Object.keys(this.baseCfg)) delete this.baseCfg[k];
    Object.assign(this.baseCfg, this.startCfg);
    this.planner = null;
    this.log("search settings: all back to the release defaults");
    return "all search settings back to the defaults";
  }

  setKnob(key: string, raw: unknown): string {
    const defaults = PLANNER_DEFAULTS as Record<string, unknown>;
    if (!(key in defaults)) return t("нет такой настройки: {key}", { key });

    const def = key in this.startCfg ? this.startCfg[key] : defaults[key];
    let value: unknown = raw;
    if (raw === undefined || raw === null || String(raw).trim() === "") value = def;
    else if (typeof def === "number") {
      const n = typeof raw === "number" ? raw : Number(String(raw).trim().replace(",", "."));
      if (!Number.isFinite(n)) return t("{key}: нужно число", { key });
      value = n;
    } else if (typeof def === "boolean") {
      value = raw === true || raw === "true" || raw === 1 || raw === "1";
    } else if (typeof def === "string") {
      value = String(raw);
    }
    const next = { ...(this.cfg.plannerCfg ?? {}) } as Record<string, unknown>;

    if (value === def && !(key in this.startCfg)) delete next[key];
    else next[key] = value;
    this.cfg.plannerCfg = next as BotConfig["plannerCfg"];

    for (const k of Object.keys(this.baseCfg)) delete this.baseCfg[k];
    Object.assign(this.baseCfg, next);
    this.planner = null;
    this.log(t(value === def ? "настройка {key} = {value} (по умолчанию)" : "настройка {key} = {value}", { key, value: JSON.stringify(value) }));
    return `${key} = ${JSON.stringify(value)}`;
  }

  configInfo(): {
    map: string;
    planner: Record<string, unknown>;
    memory: { events: number; map: string } | null;
    traps: number;
  } {
    let traps = 0;
    if (this.deadCells !== null) for (const v of this.deadCells.cells) traps += v;
    return {
      map: this.mapName(),
      planner: { ...(this.cfg.plannerCfg ?? {}) },
      memory: this.memory === null ? null : { events: this.memory.noted, map: this.mapName() },
      traps,
    };
  }

  private brainName(): string {
    return this.cfg.planner === true ? "planner" : this.cfg.scripted === true ? "scripted" : "net";
  }

  private loadRelations(): void {
    try {
      const raw = JSON.parse(readFileSync(this.cfg.relationsFile ?? RELATIONS_FILE, "utf8")) as Record<string, string[]> & { v?: number };
      this.relationsVersion = typeof raw.v === "number" ? raw.v : 1;
      for (const key of ["war", "friend", "clanWar", "clanFriend", "ignore"] as const) {
        for (const v of raw[key] ?? []) this.relations[key].set(v.trim().toLowerCase(), v);
      }
    } catch {

    }
  }

  private saveLang(l: Lang): void {
    const file = this.cfg.settingsFile;
    if (file === undefined) return;
    try {
      const cur = JSON.parse(readFileSync(file, "utf8").replace(/^\uFEFF/, "")) as unknown;
      if (cur === null || typeof cur !== "object" || Array.isArray(cur) || typeof (cur as Record<string, unknown>).server !== "string") return;
      if ((cur as Record<string, unknown>).lang === l) return;
      writeFileSync(file, JSON.stringify({ ...(cur as Record<string, unknown>), lang: l }, null, 2));
    } catch {

    }
  }

  private saveLowCpu(on: boolean): void {
    const file = this.cfg.settingsFile;
    if (file === undefined) return;
    try {
      const cur = JSON.parse(readFileSync(file, "utf8").replace(/^\uFEFF/, "")) as unknown;
      if (cur === null || typeof cur !== "object" || Array.isArray(cur) || typeof (cur as Record<string, unknown>).server !== "string") return;
      const v = on ? "on" : "off";
      if ((cur as Record<string, unknown>).lowCpu === v) return;
      writeFileSync(file, JSON.stringify({ ...(cur as Record<string, unknown>), lowCpu: v }, null, 2));
    } catch {

    }
  }

  private saveStrong(on: boolean): void {
    const file = this.cfg.settingsFile;
    if (file === undefined) return;
    try {
      const cur = JSON.parse(readFileSync(file, "utf8").replace(/^\uFEFF/, "")) as unknown;
      if (cur === null || typeof cur !== "object" || Array.isArray(cur) || typeof (cur as Record<string, unknown>).server !== "string") return;
      const v = on ? "on" : "off";
      if ((cur as Record<string, unknown>).strong === v) return;
      writeFileSync(file, JSON.stringify({ ...(cur as Record<string, unknown>), strong: v }, null, 2));
    } catch {

    }
  }

  private saveRelations(): void {
    try {
      const file = this.cfg.relationsFile ?? RELATIONS_FILE;
      mkdirSync(dirname(file), { recursive: true });
      writeFileSync(
        file,
        JSON.stringify(
          {
            v: this.relationsVersion,
            war: [...this.relations.war.values()],
            friend: [...this.relations.friend].filter(([key]) => !this.teammates.has(key)).map(([, name]) => name),
            clanWar: [...this.relations.clanWar.values()],
            clanFriend: [...this.relations.clanFriend.values()],
            ignore: [...this.relations.ignore.values()],
          },
          null,
          1,
        ),
      );
    } catch (err) {
      this.log(`could not save the war list: ${err instanceof Error ? err.message : String(err)}`);
    }
    this.onRelationsSaved?.();
  }

  onRelationsSaved: (() => void) | null = null;

  private relationCommand(key: "war" | "friend" | "clanWar" | "clanFriend" | "ignore", arg: string, label: string): string {
    const set = this.relations[key];
    let what = arg.trim();

    if (what !== "" && what.toLowerCase() !== "off" && (key === "war" || key === "friend" || key === "ignore")) {
      const hits = this.playersMatching(what);

      const exact = hits.find((h) => h.toLowerCase() === what.toLowerCase());
      if (exact !== undefined) what = exact;
      else if (hits.length > 1) return t('{label}: "{what}" -- это {n}: {list}. Уточни.', { label, what, n: hits.length, list: hits.join(", ") });
      else if (hits.length === 1) what = hits[0];
    }
    if (what === "") return set.size === 0 ? `${label}: nobody` : `${label}: ${[...set.values()].join(", ")}`;
    if (what.toLowerCase() === "off") {

      const mates = key === "friend" ? [...set].filter(([k]) => this.teammates.has(k)) : [];
      set.clear();
      for (const [k, n] of mates) set.set(k, n);
      this.saveRelations();
      return `${label}: cleared`;
    }
    const who = what.toLowerCase();

    if (key === "friend" && this.teammates.has(who)) {
      this.teammates.delete(who);
      this.saveRelations();
      return `${label}: ${set.get(who) ?? what}`;
    }
    if (key === "war") this.teammates.delete(who);
    if (set.has(who)) {
      const had = set.get(who) ?? what;
      set.delete(who);
      this.saveRelations();
      return `${label}: removed ${had}`;
    }

    const opposites: ("war" | "friend" | "clanWar" | "clanFriend" | "ignore")[] =
      key === "war" ? ["friend", "ignore"] : key === "friend" ? ["war"] : key === "clanWar" ? ["clanFriend"] : key === "ignore" ? ["war"] : ["clanWar"];
    const moved = opposites.filter((o) => this.relations[o].delete(who));
    set.set(who, what);
    this.saveRelations();
    return `${label}: ${what}${moved.length > 0 ? ` (was on the ${moved.join("/")} list)` : ""}`;
  }

  relationsInfo(): Record<"war" | "friend" | "ignore" | "clanWar" | "clanFriend" | "partner", string[]> {
    const r = this.relations;
    return {
      war: [...r.war.values()],
      friend: [...r.friend.values()],
      ignore: [...r.ignore.values()],
      clanWar: [...r.clanWar.values()],
      clanFriend: [...r.clanFriend.values()],

      partner: [...this.partnerKeys],
    };
  }

  private readonly teammates = new Set<string>();

  private readonly partnerKeys = new Set<string>();

  private partnerId: number | null = null;

  setPartnerId(id: number | null): void {
    this.partnerId = id;
  }

  ownClientId(): number {
    return this.phase === "online" ? this.ownId : -1;
  }

  private onList(list: "friend" | "ignore" | "war", nameKey: string, id: number): boolean {
    if (list !== "war" && id >= 0 && id === this.duelFoe()) return false;
    const byId = this.partnerId !== null;
    if (byId && id === this.partnerId && partnerNick(this.partnerKeys, nameKey)) {
      for (const key of this.relations[list].keys()) if (this.partnerKeys.has(key.replace(DUPLICATE_PREFIX, ""))) return true;
      return false;
    }
    return listed(this.relations[list], nameKey, this.partnerKeys, byId);
  }

  private clanFriend(clanKey: string, id: number): boolean {
    return !(id >= 0 && id === this.duelFoe()) && listed(this.relations.clanFriend, clanKey);
  }

  private duelFoe(): number {
    return this.duelSeen && this.duelNow() && this.duelScore !== null ? this.duelScore.id : -1;
  }

  private outOfGame(id: number): boolean {
    return this.duelNow() ? this.world.spectating(id) : this.world.notPlaying(id);
  }

  private isPartnerTee(id: number): boolean {
    if (this.partnerKeys.size === 0) return false;
    const nameKey = this.nameKeyOf(id);
    if (this.partnerId !== null && this.partnerId >= 0) return id === this.partnerId && partnerNick(this.partnerKeys, nameKey);
    return this.partnerKeys.has(nameKey.replace(DUPLICATE_PREFIX, ""));
  }

  private namedChallenger(id: number): boolean {
    const c = this.duelChallenger;
    if (c === "") return false;
    const n = this.nameKeyOf(id);
    if (n === "") return false;
    return [n, n.replace(DUPLICATE_PREFIX, "")].some((k) => k !== "" && (c === k || c.endsWith(` ${k}`) || c.endsWith(`]${k}`) || c.replace(DUPLICATE_PREFIX, "") === k));
  }

  private nameKeyOf(id: number): string {
    const card = this.client?.SnapshotUnpacker?.AllObjClientInfo?.find((c) => c.id === id);
    return (card?.name ?? "").trim().toLowerCase();
  }

  private relationsVersion = 2;
  setTeammate(name: string): void {
    const what = name.trim();
    if (what === "") return;
    const who = what.toLowerCase();

    if (this.relationsVersion < 2) {
      const had = this.relations.friend.delete(who);
      this.relationsVersion = 2;
      if (had) this.saveRelations();
    }

    this.partnerKeys.add(who);
    if (this.relations.friend.has(who)) return;

    this.teammates.add(who);
    this.relations.friend.set(who, what);
  }

  setRelation(list: "war" | "friend" | "ignore", name: string, on: boolean): string {
    const what = name.trim();
    if (what === "" || !["war", "friend", "ignore"].includes(list)) return "";
    const who = what.toLowerCase();

    if (list === "friend" || (list === "war" && on)) this.teammates.delete(who);
    const r = this.relations;
    if (!on) {

      for (const key of [...r[list].keys()]) {

        if (key === who || (key !== "" && !this.partnerKeys.has(key.replace(DUPLICATE_PREFIX, "")) && (who.includes(key) || key.replace(DUPLICATE_PREFIX, "") === who.replace(DUPLICATE_PREFIX, "")))) {
          r[list].delete(key);
          if (list === "friend") this.teammates.delete(key);
        }
      }
      this.saveRelations();
      return `${list}: removed ${what}`;
    }
    if (list === "war") {
      r.friend.delete(who);
      r.ignore.delete(who);
    } else r.war.delete(who);
    r[list].set(who, what);
    this.saveRelations();
    return `${list}: ${what}`;
  }

  private duelScore: { id: number; name: string; startMs: number; ours: number; theirs: number } | null = null;
  private duelSwapSince = -1;
  private endDuelScore(): void {
    const d = this.duelScore;
    this.duelScore = null;
    if (d === null) return;
    this.emit("event", t("дуэль с {name} окончена: заморозил {ours}, заморозили {theirs}", { name: d.name, ours: d.ours, theirs: d.theirs }));
    const file = join(dirname(this.cfg.relationsFile ?? RELATIONS_FILE), "duels.json");
    try {
      let list: unknown[] = [];
      if (existsSync(file)) {
        try {
          const raw = JSON.parse(readFileSync(file, "utf8")) as unknown;
          if (Array.isArray(raw)) list = raw;
        } catch {

          renameSync(file, `${file}.bad-${Date.now()}`);
        }
      }
      list.push({ at: new Date(d.startMs).toISOString(), seconds: Math.round((Date.now() - d.startMs) / 1000), opponent: d.name, ours: d.ours, theirs: d.theirs });
      mkdirSync(dirname(file), { recursive: true });

      writeFileSync(`${file}.tmp`, JSON.stringify(list.slice(-DUEL_LOG_KEEP), null, 1));
      renameSync(`${file}.tmp`, file);
    } catch (err) {
      this.log(`could not save the duel: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  private onFreezeOnset(tee: TeeState, me: TeeState | undefined, tick: number): void {
    if (me === undefined || !me.alive) return;
    const last = this.lastTouch.get(tee.id);
    if (last === undefined || tick - last.tick > BLOCK_CREDIT_TICKS) return;
    if (tee.id === me.id) {
      if (last.by === me.id || this.isFriendId(last.by)) return;
      this.stats.blockedBy++;
      if (this.duelScore !== null && last.by === this.duelScore.id) this.duelScore.theirs++;
      this.emit("event", t("меня заморозил {name}", { name: this.nameOfLive(last.by) }));
      return;
    }
    if (last.by !== me.id || this.isFriendId(tee.id)) return;
    this.stats.blocks++;
    if (this.duelScore !== null && tee.id === this.duelScore.id) this.duelScore.ours++;
    this.emit("event", t("заморозил {name}", { name: this.nameOfLive(tee.id) }));
  }

  private inFreezeTiles(pos: Vec2): boolean {
    return this.inFreezeTilesIn(this.world.collision, pos);
  }

  private rescuable(self: TeeState, t: TeeState | undefined): t is TeeState {
    if (t === undefined || t.id === this.ownId || !t.alive || !t.frozen || t.deepFrozen === true || t.freezeTicksLeft < RESCUE_MIN_FREEZE_TICKS) return false;
    if (!this.isFriendId(t.id) || this.world.notPlaying(t.id)) return false;
    if ((this.rescuePausedUntil.get(t.id) ?? -Infinity) > this.world.tick) return false;
    if (vdistance(self.pos, t.pos) > RESCUE_RANGE_PX) return false;
    const wb = this.wbHolding();
    const side = wb === null ? null : this.wbChooser.side;
    if (wb !== null && side !== null && !inWbLeash(wb, side, Math.trunc(t.pos.x / 32), Math.trunc(t.pos.y / 32))) return false;
    return true;
  }

  private rescueFriend(client: TwClient, self: TeeState, fighting = false): boolean {
    if (self.frozen || this.mode !== "fight") return false;
    if (fighting && this.world.allTees().some((t) => t.id !== this.ownId && t.alive && t.hookedPlayer === this.ownId && !this.isFriendId(t.id))) return false;
    const cands = this.world
      .allTees()
      .filter((t) => this.rescuable(self, t))
      .sort((a, b) => vdistance(self.pos, a.pos) - vdistance(self.pos, b.pos));
    if (cands.length === 0) {
      this.rescueHammerSince = -1;
      this.rescuePullSince = -1;
      return false;
    }

    const inFreezeAll = cands.filter((t) => this.inFreezeTiles(t.pos));
    const pulling = inFreezeAll.find((t) => t.id === this.rescuePullId && this.world.tick - this.rescuePullAt < 10);
    const inFreeze =
      pulling ??
      inFreezeAll.find((t) => {
        const none = this.pullNone.get(t.id);
        return none === undefined || none.key !== this.pullKey(self, t) || this.world.tick - none.tick >= PULL_NONE_TICKS;
      }) ??
      inFreezeAll[0];
    if (inFreeze !== undefined && (!this.lowCpu || pulling !== undefined || this.stats.ticks % 3 === 0)) {
      const him = inFreeze;
      const pull = this.pullLine(self, him);

      if (pull === null && self.hookState !== HOOK_IDLE && self.hookedPlayer !== him.id && this.rescuePullId === him.id && this.world.tick - this.rescuePullAt < 10) {
        this.applyInput(client, this.guard(self, { ...this.prevInput, hook: 0 }), self.activeWeapon);
        return true;
      }

      const safe = pull === null ? null : this.guard(self, pull);
      if (pull !== null && safe === pull) {

        if (this.rescuePullId !== him.id || this.rescuePullSince < 0 || this.world.tick - this.rescuePullAt > 25) {
          this.rescuePullId = him.id;
          this.rescuePullSince = this.world.tick;
        } else if (this.world.tick - this.rescuePullSince > PULL_GIVE_UP_TICKS) {
          this.rescuePausedUntil.set(him.id, this.world.tick + RESCUE_PAUSE_TICKS);
          this.rescuePullSince = -1;
          this.log(`friend ${this.nameOfLive(him.id)} is still in the freeze after ${PULL_GIVE_UP_TICKS / 50}s of pulling; leaving him for ${RESCUE_PAUSE_TICKS / 50}s`);
          return false;
        }
        this.rescuePullAt = this.world.tick;
        if (this.world.tick - this.rescueSaidTick > 5 * 50) {
          this.rescueSaidTick = this.world.tick;
          this.emit("event", `pulling friend ${this.nameOfLive(him.id)} out of the freeze with the rope`);
        }
        this.applyInput(client, safe, self.activeWeapon);
        return true;
      }
    }
    if (inFreeze === undefined) this.rescuePullSince = -1;
    const him = cands.find((t) => !this.inFreezeTiles(t.pos));
    if (him === undefined) return false;
    const name = this.nameOfLive(him.id);
    const d = vdistance(self.pos, him.pos);
    if (d > RESCUE_HAMMER_PX * 3) {
      if (fighting) return false;

      if (this.nav === null && this.world.tick - this.rescueWalkTick > RESCUE_WALK_RETRY_TICKS && this.reachable(self.pos, him)) {
        this.rescueWalkTick = this.world.tick;
        const reply = this.gotoCommand(`@${name}`, { throughFreeze: false, crossings: [] });
        if (this.nav !== null) {
          this.rescueId = him.id;
          this.seekingGame = this.navReturnMode === "fight";
        }
        this.log(`friend ${name} lies frozen ${Math.round(d / 32)} tiles off; walking to him -- ${reply}`);
      }
      return false;
    }
    if (fighting && d > RESCUE_HAMMER_PX) return false;
    const input: PlayerInput = { ...emptyInput(), targetX: him.pos.x - self.pos.x, targetY: him.pos.y - self.pos.y, fire: this.prevInput.fire, wantedWeapon: WEAPON_HAMMER + 1 };
    if (d <= RESCUE_HAMMER_PX && this.rescueHitSafe(self, him)) {
      if (this.rescueHammerId !== him.id || this.rescueHammerSince < 0) {
        this.rescueHammerId = him.id;
        this.rescueHammerSince = this.world.tick;
      } else if (this.world.tick - this.rescueHammerSince > RESCUE_GIVE_UP_TICKS) {

        this.rescuePausedUntil.set(him.id, this.world.tick + RESCUE_PAUSE_TICKS);
        this.rescueHammerSince = -1;
        this.log(`friend ${name} does not thaw under the hammer; leaving him for ${RESCUE_PAUSE_TICKS / 50}s`);
        return false;
      }

      input.fire = this.prevInput.fire + 1;
      if (this.world.tick - this.rescueSaidTick > 5 * 50) {
        this.rescueSaidTick = this.world.tick;
        this.emit("event", `hammering friend ${name} out of the freeze`);
      }
    } else {
      if (fighting) return false;
      if (d > RESCUE_HAMMER_PX && Math.abs(him.pos.x - self.pos.x) > 8) input.direction = him.pos.x > self.pos.x ? 1 : -1;

      if (input.fire & 1) input.fire++;
    }
    this.applyInput(client, this.guard(self, input), self.activeWeapon);
    return true;
  }

  private pullLine(self: TeeState, him: TeeState): PlayerInput | null {
    if (vdistance(self.pos, him.pos) > TUNING.hookLength) return null;
    const onHim = self.hookedPlayer === him.id;

    if (!onHim && self.hookState !== HOOK_IDLE && self.hookState !== HOOK_FLYING) return null;
    const key = this.pullKey(self, him);
    const none = this.pullNone.get(him.id);
    if (none !== undefined && none.key === key && this.world.tick - none.tick < PULL_NONE_TICKS && this.world.tick >= none.tick) return null;
    if (this.pullSim === null || this.pullSim.collision !== this.world.collision) {
      this.pullSim = new SimWorld(this.world.collision, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
    }
    const sim = this.pullSim;

    const others = this.world.allTees().filter((t) => t.alive && t.id !== self.id && t.id !== him.id && vdistance(t.pos, self.pos) < TUNING.hookLength + 64);
    let best: PlayerInput | null = null;
    let start: ReturnType<SimWorld["saveState"]> | null = null;
    try {
      for (const t of sim.allTees()) sim.removeTee(t.id);
      sim.addTee(self.id, self.pos);
      sim.applyTeeState(self.id, self);
      sim.addTee(him.id, him.pos);
      sim.applyTeeState(him.id, him);
      for (const o of others) {
        sim.addTee(o.id, o.pos);
        sim.applyTeeState(o.id, o);
      }
      const hold0 = (): void => {
        sim.setInput(him.id, emptyInput());
        for (const o of others) sim.setInput(o.id, emptyInput());
      };
      for (let t = 0; t < this.lagTicks(); t++) {
        sim.setInput(self.id, this.prevInput);
        hold0();
        sim.step();
      }
      start = sim.saveState();
      const holds = onHim ? [0, ...PULL_HOLDS] : PULL_HOLDS;
      search: for (const hold of holds) {
        for (const dir of [0, -1, 1]) {
          for (const jump of [0, 1]) {
            sim.restoreState(start);
            let caught = onHim;
            let ok = true;
            let first: PlayerInput | null = null;
            for (let t = 0; t < PULL_ROLL_TICKS && ok; t++) {
              const me = sim.getTee(self.id);
              const fr = sim.getTee(him.id);
              if (me === undefined || fr === undefined) {
                ok = false;
                break;
              }
              const input: PlayerInput = {
                ...emptyInput(),
                direction: t < hold ? dir : 0,
                jump: jump === 1 && t < 2 ? 1 : 0,
                hook: t < hold ? 1 : 0,
                targetX: fr.pos.x - me.pos.x,
                targetY: fr.pos.y - me.pos.y,
                wantedWeapon: WEAPON_HAMMER + 1,
              };
              if (t === 0) {
                first = input;

                if (!onHim && hold > 0 && self.hookState === HOOK_IDLE && others.length > 0 && this.ropeCatches(self, input, others, him)) {
                  ok = false;
                  break;
                }
              }
              sim.setInput(self.id, input);
              hold0();
              sim.step();
              const meAfter = sim.getTee(self.id);
              if (meAfter === undefined || !meAfter.alive || meAfter.frozen) ok = false;
              else if (meAfter.hookedPlayer === him.id) caught = true;
              else if (meAfter.hookedPlayer >= 0) ok = false;
            }
            if (!ok || !caught || first === null) continue;
            const end = sim.getTee(him.id);
            if (end === undefined || !end.alive || this.inFreezeTilesIn(sim.collision, end.pos) || restsInFreeze(sim.collision, end.pos, end.vel) > 0) continue;
            best = first;
            break search;
          }
        }
      }
    } catch {
      best = null;
    } finally {
      if (start !== null) sim.restoreState(start);
    }
    if (best === null) this.pullNone.set(him.id, { key, tick: this.world.tick });
    else this.pullNone.delete(him.id);
    return best;
  }

  private pullKey(self: TeeState, him: TeeState): string {
    return `${Math.trunc(self.pos.x / 32)},${Math.trunc(self.pos.y / 32)}:${Math.trunc(him.pos.x / 32)},${Math.trunc(him.pos.y / 32)}:${self.hookState}:${self.hookedPlayer === him.id ? 1 : 0}`;
  }

  private inFreezeTilesIn(c: Collision, pos: Vec2): boolean {
    if (c.isFreeze(pos.x, pos.y)) return true;
    return [-HALF_TEE, HALF_TEE].some((dx) => [-HALF_TEE, HALF_TEE].some((dy) => c.isFreeze(pos.x + dx, pos.y + dy)));
  }

  private rescueHitSafe(self: TeeState, him: TeeState): boolean {
    const dx = him.pos.x - self.pos.x;
    const dy = him.pos.y - self.pos.y;
    const n = Math.hypot(dx, dy) || 1;
    const ux = dx / n;
    const uy = dy / n - 1.1;
    const m = Math.hypot(ux, uy) || 1;
    const vel = { x: him.vel.x + (ux / m) * 10, y: him.vel.y + (uy / m) * 10 };
    if (restsInFreeze(this.world.collision, him.pos, vel) > 0) return false;
    for (const t of this.world.allTees()) {
      if (t.id === this.ownId || t.id === him.id || !t.alive || !t.frozen || this.isFriendId(t.id)) continue;
      if (vdistance(t.pos, self.pos) < RESCUE_HAMMER_PX + 16) return false;
    }
    return true;
  }

  private wbHolding(): WbDef | null {
    if (this.wbDef === null || this.wbMode === "off" || this.home !== null || this.duelNow()) return null;
    if (Date.now() < this.wbPausedUntilMs) return null;
    return this.mode === "fight" || (this.mode === "goto" && this.navReturnMode === "fight") ? this.wbDef : null;
  }

  duelNow(): boolean {
    return this.duelMode === "on" || (this.duelMode === "auto" && this.duelSeen);
  }

  private updateDuel(ownId: number): void {

    const all = this.world.allTees().filter((t) => t.id !== ownId && t.alive);
    const visible = all.length;
    const others = all.filter((t) => !this.isPartnerTee(t.id) && (!this.isFriendId(t.id) || this.namedChallenger(t.id)));
    const accepted = this.duelAnsweredMs > 0 && Date.now() - this.duelAnsweredMs < DUEL_ACCEPT_WINDOW_MS;
    const now = this.world.tick;
    if (!this.duelSeen) {
      if (!(visible === 1 && others.length === 1 && accepted)) {
        this.duelForSince = -1;
        return;
      }
      if (this.duelForSince < 0 || now < this.duelForSince) this.duelForSince = now;
      if (now - this.duelForSince < DUEL_ENTER_TICKS) return;
      this.duelSeen = true;
      this.duelAgainstSince = -1;
      const foe = others[0];
      if (foe !== undefined) this.duelScore = { id: foe.id, name: this.nameOfLive(foe.id), startMs: Date.now(), ours: 0, theirs: 0 };
      if (this.duelMode === "off") return;
      if (this.nav !== null) this.cancelNav("a duel started");
      if (this.trek !== null) this.endTrek();
      this.emit("event", "duel: only the opponent is on screen -- playing it 1 on 1 (no WB, no walks)");
      return;
    }

    const d = this.duelScore;
    if (d !== null && visible === 1 && others.length === 1 && others[0].id !== d.id) {
      const name = this.nameOfLive(others[0].id);
      if (name === d.name) {
        d.id = others[0].id;
        this.duelSwapSince = -1;
      } else {
        if (this.duelSwapSince < 0 || now < this.duelSwapSince) this.duelSwapSince = now;
        if (now - this.duelSwapSince >= DUEL_ENTER_TICKS) {
          this.endDuelScore();
          this.duelScore = { id: others[0].id, name, startMs: Date.now(), ours: 0, theirs: 0 };
          this.duelSwapSince = -1;
        }
      }
    } else this.duelSwapSince = -1;

    if (visible < 2 && others.length === visible) {
      this.duelAgainstSince = -1;
      return;
    }
    if (this.duelAgainstSince < 0 || now < this.duelAgainstSince) this.duelAgainstSince = now;
    if (now - this.duelAgainstSince < DUEL_LEAVE_TICKS) return;
    this.duelSeen = false;
    this.duelForSince = -1;
    this.endDuelScore();

    this.duelAnsweredMs = 0;
    this.duelChallenger = "";
    if (this.duelMode === "auto") this.emit("event", "duel over: the server is on screen again");
  }

  private noteWbWalkDeath(): void {
    this.wbWalkFails++;
    if (this.wbWalkFails < WB_WALK_MAX_FAILS) return;
    this.wbWalkFails = 0;
    this.wbPauses++;
    const pauseMs = Math.min(WB_WALK_PAUSE_MAX_MS, WB_WALK_PAUSE_MS * 2 ** (this.wbPauses - 1));
    this.wbPausedUntilMs = Date.now() + pauseMs;
    const lag = this.lagTicks();
    if (this.nav !== null && this.wbWalk) this.cancelNav("the WB cannot be reached");
    this.emit("event", t("ВБ: {n} попыток подряд погибли по пути (пинг {lag} тиков) -- {min} мин играю там, где стою", { n: WB_WALK_MAX_FAILS, lag, min: Math.round(pauseMs / 60_000) }));
  }

  private wbWalkCuttable(self: TeeState): boolean {
    const def = this.wbHolding();
    const side = this.wbChooser.side;
    if (def === null || side === null || self.frozen) return false;
    const tx = Math.trunc(self.pos.x / 32);
    const ty = Math.trunc(self.pos.y / 32);

    return sideDef(def, side).zone.some((b) => tx >= b.x0 && tx <= b.x1 && ty > b.y0 && ty <= b.y1);
  }

  private updateWbSide(ownId: number, self: TeeState): void {
    const def = this.wbDef;
    if (def === null) return;
    const counts = { left: 0, right: 0 };
    for (const t of this.world.allTees()) {
      if (t.id === ownId || !t.alive) continue;
      if (this.afk(t, true) || this.parkedInFreeze(t)) continue;
      const at = sideAt(def, Math.trunc(t.pos.x / 32), Math.trunc(t.pos.y / 32));
      if (at !== null) counts[at]++;
    }
    this.wbCounts = counts;
    const before = this.wbChooser.side;
    let side: WbSide;
    if (this.wbMode === "left" || this.wbMode === "right") {
      side = this.wbMode;
      this.wbChooser.side = side;
    } else {
      const here = sideAt(def, Math.trunc(self.pos.x / 32), Math.trunc(self.pos.y / 32));

      const nearer = Math.abs(self.pos.x / 32 - def.left.watch.tx) <= Math.abs(self.pos.x / 32 - def.right.watch.tx) ? "left" : "right";
      side = this.wbChooser.update(counts, here, this.world.tick, nearer);

      const tx = Math.trunc(self.pos.x / 32);
      const ty = Math.trunc(self.pos.y / 32);
      const standing = self.alive && !self.frozen ? (["left", "right"] as const).find((s) => inAnyBox(sideDef(def, s).zone, tx, ty)) : undefined;
      if (standing !== undefined && standing !== side) {
        this.wbChooser.adopt(standing);
        side = standing;
      }
    }
    if (before === null || before === side || this.wbHolding() === null) return;
    this.emit("event", `WB: over to the ${side} (${counts.left} playing on the left, ${counts.right} on the right)`);

    if (this.wbWalk && this.nav !== null) {
      this.nav.cancel("the WB side changed");
      this.endNav();
      this.idleSinceTick = this.world.tick - WB_RETURN_TICKS - 1;
    }
  }

  private isPartnerNow(id: number): boolean {
    return this.partnerId !== null && this.partnerId >= 0 && this.isPartnerTee(id);
  }

  private wbSpot(ownId: number, def: WbDef, side: WbSide, here?: { tx: number; ty: number }): { tx: number; ty: number } {
    const all = sideDef(def, side).spots;
    const inside = here !== undefined && inWbHall(def, side, here.tx, here.ty);
    const reachable = inside ? all.filter((p) => here.ty - p.ty < WB_NO_CLIMB_TILES) : all;
    const spots = reachable.length > 0 ? reachable : all;

    const taken = (p: { tx: number; ty: number }): boolean =>
      this.world.allTees().some((t) => {
        if (t.id === ownId || !t.alive) return false;

        if (this.isPartnerNow(t.id) || (!t.frozen && this.isFriendId(t.id))) return onWbSpot({ tx: Math.trunc(t.pos.x / 32), ty: Math.trunc(t.pos.y / 32) }, p);
        return !t.frozen && Math.abs(t.pos.x - (p.tx * 32 + 16)) < 32 && Math.abs(t.pos.y - (p.ty * 32 + 16)) < 32;
      });

    const friendHolds = (p: { tx: number; ty: number }): boolean =>
      this.world.allTees().some((t) => t.id !== ownId && t.alive && !t.frozen && !this.isPartnerNow(t.id) && this.isFriendId(t.id) && onWbSpot({ tx: Math.trunc(t.pos.x / 32), ty: Math.trunc(t.pos.y / 32) }, p));
    return spots.find((p) => (here !== undefined && onWbSpot(here, p)) || !taken(p)) ?? spots.find((p) => !friendHolds(p)) ?? (inside ? here : spots[0]);
  }

  private walkToWb(ownId: number, self: TeeState): void {
    const def = this.wbHolding();
    const side = this.wbChooser.side;
    if (def === null || side === null) return;
    const col = this.world.collision;
    const tx = Math.trunc(self.pos.x / 32);
    const ty = Math.trunc(self.pos.y / 32);
    const inside = inWbHall(def, side, tx, ty);

    const spots = [this.wbSpot(ownId, def, side, { tx, ty }), ...sideDef(def, side).spots];
    let spot: { tx: number; ty: number } | null = null;
    for (const p of spots) {
      if (onWbSpot({ tx, ty }, p)) return;
      if (!inside) {
        spot = p;
        break;
      }

      if (ty - p.ty >= WB_NO_CLIMB_TILES) continue;
      const way = findRoute(col, self.pos, { x: p.tx * 32 + 16, y: p.ty * 32 + 16 }, { nearTiles: 1, partial: false, allowKill: false, throughFreeze: false, maxNodes: REACH_MAX_NODES });
      if (way !== null) {
        spot = p;
        break;
      }
    }
    if (spot === null) {
      this.log(`WB ${side}: no way back to its spots from (${tx},${ty}) without the freeze; holding here`);
      return;
    }
    const reply = this.startNav([tileGoal(col, spot.tx, spot.ty)], inside ? { throughFreeze: false } : { crossings: def.crossings });
    this.wbWalk = this.nav !== null;

    this.seekingGame = this.nav !== null && this.navReturnMode === "fight";
    const line = `WB ${side}: back to (${spot.tx},${spot.ty}) -- ${reply}`;
    if (inside) this.log(line);
    else this.emit("event", line);
  }

  private crowdAt(ownId: number, at: Vec2): { tees: number; busy: number } {
    let tees = 0;
    let busy = 0;
    for (const t of this.world.allTees()) {
      if (t.id === ownId || !t.alive) continue;
      if (vdistance(at, t.pos) > CROWD_RADIUS_PX) continue;
      if (this.afk(t, true) || this.parkedInFreeze(t)) continue;
      tees++;
      if (t.hookState >= HOOK_FLYING || this.world.tick - t.attackTick < ACTION_MEMORY_TICKS) busy++;
    }
    return { tees, busy };
  }

  private engagedNow(ownId: number, self: TeeState): boolean {
    if (self.frozen || self.hookedPlayer >= 0) return true;
    for (const t of this.world.allTees()) {
      if (t.id === ownId || !t.alive) continue;
      if (t.hookedPlayer === ownId) return true;
      if (!t.frozen && vdistance(self.pos, t.pos) < ENGAGED_PX && !this.afk(t)) return true;
    }
    return false;
  }

  private someoneWorthFighting(ownId: number, at: Vec2, within: number): boolean {
    for (const t of this.world.allTees()) {
      if (t.id === ownId || !t.alive || t.frozen) continue;
      if (vdistance(at, t.pos) > within) continue;
      if (!this.afk(t)) return true;
    }
    return false;
  }

  private freezeWithin(tx: number, ty: number, r: number): boolean {
    const c = this.world.collision;
    for (let oy = -r; oy <= r; oy++) for (let ox = -r; ox <= r; ox++) if (c.isFreeze((tx + ox) * 32 + 16, (ty + oy) * 32 + 16)) return true;
    return false;
  }

  private startTrek(from: Vec2, to: { x: number; y: number }): string {

    if (this.trekAvoidMap !== this.world.collision) {
      this.trekAvoid = new Set();
      this.trekAvoidMap = this.world.collision;
    }
    let route = findRoute(this.world.collision, from, to, { nearTiles: 3, partial: true, allowKill: true, throughFreeze: false, avoid: this.trekAvoid });

    if (route !== null && route.steps.length > 0) {
      const end = route.steps[route.steps.length - 1];
      const gx = Math.trunc(to.x / 32);
      const gy = Math.trunc(to.y / 32);
      const short = Math.abs(end.x - gx) > 3 || Math.abs(end.y - gy) > 3;
      if (short && this.freezeWithin(end.x, end.y, 2)) route = null;
    }
    if (route === null || route.steps.length === 0) {
      this.trek = null;
      this.seekingGame = false;
      const t = { x: Math.trunc(to.x / 32), y: Math.trunc(to.y / 32) };
      return `no route there (map ${this.mapName()}, to tile ${t.x},${t.y})`;
    }
    this.trek = { steps: route.steps, at: 0, since: this.world.tick, best: Infinity, bestTick: this.world.tick };
    this.seekingGame = true;
    const hooks = route.steps.filter((s) => s.kind === "hook").length;
    return `walking over: ${route.steps.length} steps${hooks > 0 ? `, ${hooks} on the rope` : ""}`;
  }

  private trekGoal(self: TeeState): Vec2 | null {
    const trek = this.trek;
    if (trek === null) return null;
    while (trek.at < trek.steps.length) {
      const s = trek.steps[trek.at];

      if (s.kind === "kill") {
        trek.at++;
        trek.best = Infinity;
        trek.bestTick = this.world.tick;
        if (this.world.tick - this.lastKillTick >= KILL_COOLDOWN_TICKS) {
          this.lastKillTick = this.world.tick;
          this.stats.selfKills++;
          this.log("the way there starts with a respawn -> /kill");
          try {
            this.client?.game.Kill();
          } catch {

          }
        }
        continue;
      }
      const p = { x: s.x * 32 + 16, y: s.y * 32 + 16 };
      const d = vdistance(self.pos, p);

      const teleNext = s.tele === true ? trek.steps[trek.at + 1] : undefined;
      if (teleNext !== undefined ? vdistance(self.pos, { x: teleNext.x * 32 + 16, y: teleNext.y * 32 + 16 }) < TREK_REACHED_PX : d < TREK_REACHED_PX) {
        trek.at++;
        trek.best = Infinity;
        trek.bestTick = this.world.tick;
        continue;
      }
      if (d < trek.best - 8) {
        trek.best = d;
        trek.bestTick = this.world.tick;
      } else if (this.world.tick - trek.bestTick > TREK_STALL_TICKS) {
        this.log(`the walk stalled ${Math.round(d)}px from step ${trek.at + 1}/${trek.steps.length} (${s.kind}); rethinking without that move`);
        if (s.move !== undefined) {

          if (this.trekAvoid.size >= 24) this.trekAvoid.clear();
          this.trekAvoid.add(s.move);
        }
        this.endTrek();
        return null;
      }
      return p;
    }

    this.trekAvoid.clear();
    this.endTrek();
    return null;
  }

  private pathGoal(self: TeeState, target: TeeState): Vec2 | null {
    if (((this.cfg.plannerCfg as { pathToTarget?: boolean } | undefined)?.pathToTarget ?? PLANNER_DEFAULTS.pathToTarget) !== true) return null;
    const d = vdistance(self.pos, target.pos);
    if (d < PATH_NEAR_PX && this.lineIsClear(self.pos, target.pos)) {
      this.path = null;
      return null;
    }
    const path = this.path;
    const age = path === null ? Infinity : this.world.tick - path.at;
    const stale =
      path === null ||
      path.target !== target.id ||
      age > PATH_REFRESH_TICKS ||
      (age >= PATH_MIN_REFRESH_TICKS && (vdistance(target.pos, path.to) > PATH_MOVED_PX || this.offPath(path, self.pos)));
    if (stale) {

      const route = findRoute(this.world.collision, self.pos, target.pos, { nearTiles: 3, partial: false, maxNodes: 4000, throughFreeze: false });
      this.path = { steps: route === null ? [] : route.steps, at: this.world.tick, target: target.id, to: { x: target.pos.x, y: target.pos.y }, done: 0 };
    }
    return this.pathAhead(this.path as NonNullable<typeof this.path>, self.pos);
  }

  private pathAhead(path: NonNullable<typeof this.path>, at: Vec2): Vec2 | null {
    const steps = path.steps;
    let near = path.done;
    let nearD = Infinity;
    for (let i = path.done; i < Math.min(steps.length, path.done + PATH_PROGRESS_WINDOW); i++) {
      const d = vdistance(at, { x: steps[i].x * 32 + 16, y: steps[i].y * 32 + 16 });
      if (d < nearD) {
        nearD = d;
        near = i;
      }
    }
    path.done = near;
    for (let i = near; i < steps.length; i++) {
      const p = { x: steps[i].x * 32 + 16, y: steps[i].y * 32 + 16 };
      if (vdistance(at, p) > PATH_REACHED_PX) return p;
    }
    return null;
  }

  private offPath(path: NonNullable<typeof this.path>, at: Vec2): boolean {
    const steps = path.steps;
    if (steps.length === 0) return true;
    for (let i = path.done; i < Math.min(steps.length, path.done + PATH_PROGRESS_WINDOW); i++) {
      if (vdistance(at, { x: steps[i].x * 32 + 16, y: steps[i].y * 32 + 16 }) <= PATH_MOVED_PX) return false;
    }
    return true;
  }

  private lineIsClear(a: Vec2, b: Vec2): boolean {
    const steps = Math.max(1, Math.trunc(vdistance(a, b) / 16));
    for (let i = 1; i < steps; i++) {
      const x = a.x + ((b.x - a.x) * i) / steps;
      const y = a.y + ((b.y - a.y) * i) / steps;
      if (this.world.collision.isSolid(x, y) || this.world.collision.isFreeze(x, y) || this.world.collision.isDeath(x, y)) return false;
    }
    return true;
  }

  private endTrek(): void {
    this.trek = null;
    this.seekingGame = false;
    this.planner?.setTravelGoal(null);
  }

  private trapCare(): boolean {
    return (this.cfg.plannerCfg?.deadZoneCost ?? 0) > 0;
  }

  private seekEnabled(): boolean {

    if (this.duelNow()) return false;
    return (this.cfg.plannerCfg as { seek?: boolean } | undefined)?.seek !== false;
  }

  private saveMemory(): void {
    if (this.memory === null || this.memoryPath === null || !this.memoryDirty) return;
    this.memory.save(this.memoryPath);
    this.memoryDirty = false;
  }

  private memoryFile(): string {
    const safe = this.mapName().replace(/[^\w.-]+/g, "_");
    return (this.cfg.memoryDir ?? "runs/memory") + `/${safe}.json`;
  }

  private helperNear(self: TeeState): boolean {
    const cards = this.client?.SnapshotUnpacker?.AllObjClientInfo;
    if (cards === undefined) return false;
    const info = new Map(cards.map((c) => [c.id, c]));
    for (const other of this.world.allTees()) {
      if (other.id === self.id || !other.alive || other.frozen) continue;
      if (vdistance(other.pos, self.pos) > HELPER_RANGE_PX) continue;
      const card = info.get(other.id);
      const nameKey = (card?.name ?? "").trim().toLowerCase();
      const clanKey = (card?.clan ?? "").trim().toLowerCase();
      if (this.onList("friend", nameKey, other.id) || this.onList("ignore", nameKey, other.id)) return true;
      if (this.clanFriend(clanKey, other.id)) return true;
    }
    return false;
  }

  private inDeadZone(pos: Vec2): boolean {
    const dead = this.deadCells;
    if (dead === null) return false;
    const tx = Math.trunc(pos.x / 32);
    const ty = Math.trunc(pos.y / 32);
    for (let oy = -1; oy <= 1; oy++) {
      for (let ox = -1; ox <= 1; ox++) {
        const i = (ty + oy) * dead.width + (tx + ox);
        if (i >= 0 && i < dead.cells.length && dead.cells[i] === 1) return true;
      }
    }
    return false;
  }

  private busiestSpot(ownId: number, from: Vec2): { x: number; y: number; dist: number; tees: number; busy: number } | null {
    const tees = this.world.allTees().filter((t) => t.alive && t.id !== ownId);
    const active = (t: TeeState): boolean => t.hookState >= HOOK_FLYING || this.world.tick - t.attackTick < ACTION_MEMORY_TICKS;

    const awake = (t: TeeState): boolean => !this.afk(t, true) && !this.parkedInFreeze(t);
    let best: { x: number; y: number; dist: number; tees: number; busy: number } | null = null;
    let bestScore = -Infinity;
    for (const centre of tees) {
      if (!awake(centre)) continue;
      let near = 0;
      let busy = 0;
      for (const other of tees) {
        if (vdistance(centre.pos, other.pos) > CROWD_RADIUS_PX) continue;
        if (!awake(other)) continue;
        near++;
        if (active(other)) busy++;
      }
      const dist = vdistance(from, centre.pos);
      const score = near + busy - dist / 1500;
      if (score > bestScore) {
        bestScore = score;
        best = { x: centre.pos.x, y: centre.pos.y, dist, tees: near, busy };
      }
    }

    if (best === null || best.dist < TARGET_MAX_PX / 2) return null;
    return best;
  }

  private gameSpot(ownId: number, from: Vec2): ReturnType<DdnetBot["busiestSpot"]> {
    const spot = this.busiestSpot(ownId, from);
    return spot !== null && wbWalkAllowed(this.wbDef, Math.trunc(spot.x / 32), Math.trunc(spot.y / 32)) ? spot : null;
  }

  private playersMatching(text: string): string[] {
    const needle = text.trim().toLowerCase();
    if (needle === "") return [];
    const snap = this.client?.SnapshotUnpacker;
    const out: string[] = [];
    for (const c of snap?.AllObjClientInfo ?? []) {
      const name = (c.name ?? "").trim();
      if (name !== "" && name.toLowerCase().includes(needle)) out.push(name);
    }
    return out;
  }

  private lagTicks(): number {
    if (this.cfg.lagCompensation === false) return 0;
    const echo = this.echoLagTicks();
    if (echo >= 0) return Math.min(MAX_LAG_TICKS, echo);
    return Math.min(MAX_LAG_TICKS, this.pingLagTicks());
  }

  private pingLagTicks(): number {
    const snap = this.client?.SnapshotUnpacker;

    const info = this.ownId >= 0 && typeof snap?.getObjPlayerInfo === "function" ? snap.getObjPlayerInfo(this.ownId) : undefined;
    const ms = typeof info?.latency === "number" && info.latency > 0 ? info.latency : 0;
    return Math.round(ms / 20);
  }

  private echoLagTicks(): number {
    if (this.echoSamples < ECHO_MIN_SAMPLES) return -1;

    let best = -1;
    let bestMiss = Infinity;
    for (let d = 1; d < this.echoMiss.length; d++) {
      if (this.echoSeen[d] < ECHO_MIN_SAMPLES) continue;
      const miss = this.echoMiss[d] / this.echoSeen[d];
      if (miss < bestMiss) {
        bestMiss = miss;
        best = d;
      }
    }
    if (best < 0 || bestMiss > ECHO_MATCH_MAX) return -1;

    for (let d = 1; d < best; d++) {
      if (this.echoSeen[d] < ECHO_MIN_SAMPLES) continue;
      if (this.echoMiss[d] / this.echoSeen[d] <= bestMiss + ECHO_MATCH_GAP) {
        best = d;
        break;
      }
    }

    return Math.max(0, best - this.snapshotGap);
  }

  private measureEcho(self: TeeState): void {
    const now = this.world.tick;
    if (this.lastSnapTick > 0 && now > this.lastSnapTick && now - this.lastSnapTick <= 4) {
      this.snapshotGap = now - this.lastSnapTick;
    }
    this.lastSnapTick = now;
    if (!self.alive || this.aimLog.length < 2) return;
    const a = wireAngleRad(self.angle);
    const ax = Math.cos(a);
    const ay = Math.sin(a);
    for (let i = 0; i < this.aimLog.length; i++) {
      const dec = this.aimLog[i];
      const d = now - dec.tick;
      if (d < 1 || d >= this.echoMiss.length) continue;
      const before = this.aimLog[i + 1];

      if (before === undefined || dec.x * before.x + dec.y * before.y > 0.98) continue;
      this.echoMiss[d] += 1 - (ax * dec.x + ay * dec.y);
      this.echoSeen[d]++;
      if (d === 1 || d === 2) this.echoSamples++;
    }
  }

  private noteAim(input: PlayerInput): void {
    const n = Math.hypot(input.targetX, input.targetY);
    if (n === 0) return;
    this.aimLog.unshift({ tick: this.world.tick, x: input.targetX / n, y: input.targetY / n });
    if (this.aimLog.length > ECHO_LOG) this.aimLog.length = ECHO_LOG;
  }

  private mapName(): string {
    return this.client?.map?.map_name ?? this.client?.lastMapDetails?.map_name ?? "?";
  }

  private mapKey(): string {
    const crc = (this.client?.map as { crc?: number } | undefined)?.crc;
    return typeof crc === "number" ? `${this.mapName()}#${(crc >>> 0).toString(16)}` : this.mapName();
  }

  private maybeUnstick(client: TwClient, self: TeeState): void {

    const STUCK_FROZEN_TICKS = 9 * 50;
    const STUCK_WEDGED_TICKS = 4 * 50;
    const STUCK_RADIUS = 48;

    const FROZEN_HARD_LIMIT_TICKS = 8 * 50;

    const FROZEN_IN_TILE_TICKS = 4 * 50;

    if (!self.frozen) this.frozenSince = -1;
    else if (this.frozenSince < 0) this.frozenSince = this.world.tick;
    const frozenFor = self.frozen && this.frozenSince >= 0 ? this.world.tick - this.frozenSince : 0;

    if (!this.acting) {
      this.stuckAnchor = null;
      return;
    }

    if (!self.frozen && this.targetId === -1) {
      this.stuckAnchor = null;
      return;
    }

    const inTiles =
      self.frozen &&
      [-HALF_TEE, HALF_TEE].some((dx) =>
        [-HALF_TEE, HALF_TEE].some((dy) => this.world.collision.isFreeze(self.pos.x + dx, self.pos.y + dy)),
      );
    const hooked = this.world.allTees().some((o) => o.id !== self.id && o.alive && o.hookedPlayer === self.id);

    const trapped = !hooked && this.inDeadZone(self.pos);
    const TRAPPED_TICKS = 1.5 * 50;

    const helped = this.helperNear(self);
    const overdue =
      (frozenFor >= FROZEN_HARD_LIMIT_TICKS ||
        (inTiles && !hooked && frozenFor >= FROZEN_IN_TILE_TICKS) ||
        (trapped && frozenFor >= TRAPPED_TICKS)) &&

      (!helped || frozenFor >= HELPED_LIMIT_TICKS);

    const wbDef = this.wbHolding();
    const wbLying =
      wbDef !== null &&
      !(["left", "right"] as const).some((s) => inAnyBox(sideDef(wbDef, s).zone, Math.trunc(self.pos.x / 32), Math.trunc(self.pos.y / 32))) &&
      self.frozen &&
      this.world.collision.isFreeze(self.pos.x, self.pos.y) &&
      !hooked &&
      !helped &&
      Math.hypot(self.vel.x, self.vel.y) < 0.5 &&
      frozenFor >= WB_LYING_TICKS;
    if (wbLying && this.world.tick - this.lastKillTick >= WB_KILL_COOLDOWN_TICKS) {
      this.lastKillTick = this.world.tick;
      this.stuckAnchor = null;
      this.frozenSince = -1;
      this.stats.selfKills++;
      this.log(`WB: lying in the freeze tiles for ${(frozenFor / 50).toFixed(1)}s with no rope on it -> /kill`);
      try {
        client.game.Kill();
      } catch {

      }
      return;
    }
    if (overdue && this.world.tick - this.lastKillTick >= KILL_COOLDOWN_TICKS) {
      this.lastKillTick = this.world.tick;
      this.stuckAnchor = null;
      this.frozenSince = -1;
      this.stats.selfKills++;
      this.log(
        trapped && frozenFor < FROZEN_IN_TILE_TICKS
          ? `frozen ${(frozenFor / 50).toFixed(1)}s where there is no way back to the game -> /kill`
          : `frozen without a break for ${(frozenFor / 50).toFixed(1)}s${inTiles ? " standing in the tiles" : ""} (a strong player's longest in an hour is 4.0s) -> /kill`,
      );
      try {
        client.game.Kill();
      } catch {

      }
      return;
    }

    if (this.stuckAnchor === null || vdistance(self.pos, this.stuckAnchor) > STUCK_RADIUS || self.frozen !== this.stuckAnchorFrozen) {
      this.stuckAnchor = { x: self.pos.x, y: self.pos.y };
      this.stuckAnchorTick = this.world.tick;
      this.stuckAnchorMs = Date.now();
      this.stuckAnchorFrozen = self.frozen;
      return;
    }
    const stuckTicks = this.world.tick - this.stuckAnchorTick;
    if (stuckTicks < (self.frozen ? STUCK_FROZEN_TICKS : STUCK_WEDGED_TICKS)) return;

    if (self.frozen) {

      for (const other of this.world.allTees()) {
        if (other.id !== self.id && other.alive && other.hookedPlayer === self.id) return;
      }

      if (this.helperNear(self) && stuckTicks < HELPED_LIMIT_TICKS) return;

      if (!this.world.collision.isFreeze(self.pos.x, self.pos.y)) return;
    }

    if (!self.frozen) {
      const held = this.targetId >= 0 ? this.world.getTee(this.targetId) : undefined;
      if (held !== undefined && held.alive && held.frozen && (this.duelNow() || self.hookedPlayer === held.id || vdistance(self.pos, held.pos) < TUNING.hookLength)) return;
    }
    if (this.world.tick - this.lastKillTick < KILL_COOLDOWN_TICKS) return;
    this.lastKillTick = this.world.tick;
    const wallMs = Date.now() - this.stuckAnchorMs;
    this.stuckAnchor = null;
    this.stats.selfKills++;
    this.log(`stuck for ${(stuckTicks / 50).toFixed(1)}s of server ticks (${wallMs}ms wall clock)${self.frozen ? " (frozen)" : ""} -> /kill`);
    try {
      client.game.Kill();
    } catch {

    }
  }

  private planAction(ownId: number, targetId: number, self: TeeState): PlayerInput {
    const target = this.world.getTee(targetId);
    if (target === undefined) return emptyInput();

    if (this.planSim === null || this.planSelfId !== ownId || this.planCollision !== this.world.collision) {
      this.planSim = new SimWorld(this.world.collision, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
      this.planSim.addTee(ownId, self.pos);
      this.planOthersIn.clear();
      this.planTargetId = -1;
      this.planSelfId = ownId;
      this.planCollision = this.world.collision;
      this.ensurePlanner().reset();
    }
    const sim = this.planSim;

    if (this.planTargetId !== targetId) {
      if (this.planTargetId >= 0 && !this.planOthersIn.has(this.planTargetId) && sim.getTee(this.planTargetId) !== undefined) {
        sim.removeTee(this.planTargetId);
      }
      if (sim.getTee(targetId) === undefined) sim.addTee(targetId, target.pos);

      this.planOthersIn.delete(targetId);
      this.planTargetId = targetId;
    }
    const enemyInput = enemyInputFromSnapshot(target);

    const lag = this.lagTicks();

    const nOthers = Math.max(0, Math.floor(this.cfg.plannerCfg?.planOthers ?? 0));
    const others =
      nOthers === 0
        ? []
        : this.world
            .allTees()
            .filter((t) => t.alive && t.id !== ownId && t.id !== targetId && vdistance(t.pos, self.pos) <= PLAN_OTHERS_PX)
            .sort((a, b) => vdistance(a.pos, self.pos) - vdistance(b.pos, self.pos))
            .slice(0, nOthers);
    syncOthers(sim, this.planOthersIn, others);

    sim.setGrenades(
      this.world
        .projectiles()
        .filter((p) => p.type === WEAPON_GRENADE && vdistance(p.pos, self.pos) <= GRENADE_WATCH_PX)
        .map((p) => ({ owner: p.owner, spawnPos: p.spawnPos, dir: p.dir, ageTicks: Math.max(0, this.world.tick - p.startTick) })),
    );
    if (this.cfg.plannerCfg?.liveTransfer === "legacy") syncPlanningWorldLegacy(sim, ownId, self, targetId, target, this.prevInput, enemyInput, lag);
    else {
      const flight = this.inFlightInputs(this.world.tick, lag);
      syncPlanningWorld(sim, ownId, self, targetId, target, this.prevInput, enemyInput, lag, flight.inFlight, flight.held);
    }

    const planner = this.ensurePlanner();

    planner.setTravelGoal(this.trek === null ? this.pathGoal(self, target) : this.trekGoal(self));

    planner.setThirdTees(
      (this.cfg.plannerCfg?.thirdTeeExposure ?? 0) > 0
        ? this.world
            .allTees()
            .filter((t) => t.alive && !t.frozen && t.id !== ownId && t.id !== targetId && vdistance(t.pos, self.pos) <= TUNING.hookLength)
            .sort((a, b) => vdistance(a.pos, self.pos) - vdistance(b.pos, self.pos))
            .slice(0, 2)
            .map((t) => ({ x: t.pos.x, y: t.pos.y }))
        : [],
    );

    const bystanders = this.world
      .allTees()
      .filter((t) => t.alive && t.frozen && t.id !== ownId && t.id !== targetId && vdistance(t.pos, self.pos) <= BYSTANDER_PX && !this.isFriendId(t.id));
    planner.setFrozenBystanders(
      bystanders.map((t) => ({ x: t.pos.x, y: t.pos.y })),
      bystanders.map((t) => ({ x: t.vel.x, y: t.vel.y })),
    );

    const cards = new Map((this.client?.SnapshotUnpacker?.AllObjClientInfo ?? []).map((c) => [c.id, c]));
    const spare = this.world
      .allTees()
      .filter((t) => t.alive && t.id !== ownId && t.id !== targetId && vdistance(t.pos, self.pos) <= TUNING.hookLength + 64 && this.spared(t, cards.get(t.id)));
    planner.setSpareBystanders(
      spare.map((t) => ({ x: t.pos.x, y: t.pos.y })),
      spare.map((t) => ({ x: t.vel.x, y: t.vel.y })),
    );
    planner.setOverrides(this.wbPlanOverrides(self));
    planner.setBand(this.wbBand(self));
    planner.setLiveTick(this.world.tick);
    let out = planner.decide(sim, ownId, targetId, this.prevInput, enemyInput);

    if (out.hook !== 0 && (spare.length > 0 || self.hookedPlayer >= 0)) {

      const holding = self.hookedPlayer >= 0 && (spare.some((t) => t.id === self.hookedPlayer) || this.isFriendId(self.hookedPlayer));
      if (holding || (self.hookState === HOOK_IDLE && this.ropeCatches(self, out, spare, target))) out = { ...out, hook: 0 };
    }
    this.lastPlan = {
      target: targetId,
      lag,
      lagPing: this.pingLagTicks(),
      others: others.length,
      arm: this.tryName,
      trek: this.trek === null ? -1 : this.trek.steps.length - this.trek.at,
      ...planner.lastInfo,
    };
    return out;
  }

  private wbBand(self: TeeState): { x0: number; y0: number; x1: number; y1: number } | null {
    const def = this.wbHolding();
    const side = this.wbChooser.side;
    if (def === null || side === null || !inWbHall(def, side, Math.trunc(self.pos.x / 32), Math.trunc(self.pos.y / 32))) return null;
    const sd = sideDef(def, side);
    const foot = sd.crossing.exit[sd.crossing.exit.length - 1];
    const top = sd.zone[0];
    return { x0: (foot.x0 - 1) * 32, x1: (foot.x1 + 2) * 32, y0: (top.y0 + 1) * 32, y1: (top.y1 - 2) * 32 };
  }

  private wbPlanOverrides(self: TeeState): Partial<PlannerConfig> | null {
    const def = this.wbHolding();
    const side = this.wbChooser.side;
    if (def === null || side === null || !inWbHall(def, side, Math.trunc(self.pos.x / 32), Math.trunc(self.pos.y / 32))) return null;

    return this.strong && (this.cfg.plannerCfg?.population ?? PLANNER_DEFAULTS.population) < STRONG_WB.population ? WB_PLAN_STRONG : WB_PLAN_OVERRIDES;
  }

  private ropeCatches(self: TeeState, input: PlayerInput, tees: readonly TeeState[], before?: TeeState): boolean {
    const n = Math.hypot(input.targetX, input.targetY);
    if (n === 0) return false;
    const dir = { x: input.targetX / n, y: input.targetY / n };
    const L = TUNING.hookLength;
    const hit = this.world.collision.intersectLineHook(self.pos, { x: self.pos.x + dir.x * L, y: self.pos.y + dir.y * L });
    let stop = hit.collision !== 0 ? vdistance(self.pos, hit.outPos) : L;
    if (before !== undefined && before.alive) stop = Math.min(stop, ropeCatchAlong(self.pos, dir, before.pos));
    return tees.some((t) => t.id !== self.id && t.alive && ropeCatchAlong(self.pos, dir, t.pos) < stop);
  }

  private applyInput(client: TwClient, input: PlayerInput, activeWeapon: number): void {
    const mv = client.movement;
    if (input.direction < 0) mv.RunLeft();
    else if (input.direction > 0) mv.RunRight();
    else mv.RunStop();

    mv.Jump(input.jump !== 0);

    if (input.hook !== 0 && this.prevInput.hook === 0) this.stats.hooksFired++;
    mv.Hook(input.hook !== 0);

    mv.SetAim(input.targetX, input.targetY);

    this.noteAim(input);

    mv.FlagScoreboard?.(this.world.tick % 50 < 2);

    mv.WantedWeapon(input.wantedWeapon === 0 ? WEAPON_HAMMER + 1 : input.wantedWeapon);

    if (firePressed(this.prevInput, input)) {
      if (client.input.m_Fire & 1) mv.Fire();
      mv.Fire();
      if (activeWeapon === WEAPON_HAMMER) this.stats.hammerFires++;
    } else if ((input.fire & 1) === 0 && client.input.m_Fire & 1) {
      mv.Fire();
    }

    try {
      client.sendInput();
    } catch {

    }

    this.prevInput = { ...input };

    this.recordSent(input);
  }

  private recordSent(input: PlayerInput): void {
    this.sent.push({ tick: this.world.tick, input: { ...input } });
    if (this.sent.length > 16) this.sent.shift();
  }

  private inFlightInputs(tick: number, lag: number): { held: PlayerInput; inFlight: PlayerInput[] } {
    const at = (k: number): PlayerInput => {
      let pick: PlayerInput | undefined;
      for (const d of this.sent) if (d.tick <= tick + k - lag - 1) pick = d.input;
      return pick ?? this.prevInput;
    };
    const inFlight: PlayerInput[] = [];
    for (let k = 1; k <= lag; k++) inFlight.push(at(k));
    return { held: at(0), inFlight };
  }

  private wander(client: TwClient, self: TeeState, anchorX?: number, lookAt?: Vec2): void {
    if (!this.collisionReady || !this.acting) {
      this.idle();
      return;
    }
    const mv = client.movement;
    const tick = this.world.tick;
    if (tick >= this.wanderUntilTick) {

      this.wanderDir = this.wanderRng.nextFloat() < 0.22 ? 0 : this.wanderRng.nextFloat() < 0.5 ? -1 : 1;
      this.wanderUntilTick = tick + 25 + Math.floor(this.wanderRng.nextFloat() * 175);
      this.wanderLook = this.wanderRng.nextFloat() * 2 - 1;
      this.wanderAim = this.wanderLook * Math.PI;
    }
    if (lookAt !== undefined) this.wanderAim = Math.atan2(lookAt.y - self.pos.y, lookAt.x - self.pos.x) + this.wanderLook * 0.2;

    const col = this.world.collision;
    if (anchorX !== undefined && this.wanderDir !== 0 && Math.abs(self.pos.x - anchorX) > 48 && Math.sign(anchorX - self.pos.x) !== this.wanderDir) {
      this.wanderDir = -this.wanderDir;
      this.wanderUntilTick = tick + 40;
    }

    const ahead = { x: self.pos.x + this.wanderDir * 40, y: self.pos.y };
    if (this.wanderDir !== 0 && col !== undefined) {
      const blocked = col.isSolid(ahead.x, ahead.y) || col.isFreeze(ahead.x, ahead.y) || col.isDeath(ahead.x, ahead.y);
      const drop = !col.isSolid(ahead.x, self.pos.y + 40) && !col.isSolid(ahead.x, self.pos.y + 80);
      const hazardBelow = wanderHazardBelow(col, ahead.x, self.pos.y);
      if (blocked || hazardBelow || (drop && this.wanderRng.nextFloat() < 0.7)) {
        this.wanderDir = -this.wanderDir;
        this.wanderUntilTick = tick + 40;
      }
    }

    if (this.wanderDir < 0) mv.RunLeft();
    else if (this.wanderDir > 0) mv.RunRight();
    else mv.RunStop();

    if (anchorX === undefined && tick >= this.wanderJumpUntilTick && this.wanderRng.nextFloat() < 0.03) {
      this.wanderJumpUntilTick = tick + 3 + Math.floor(this.wanderRng.nextFloat() * 8);
    }
    if (anchorX === undefined && tick >= this.wanderHookUntilTick && this.wanderRng.nextFloat() < 0.02) {
      this.wanderHookUntilTick = tick + 15 + Math.floor(this.wanderRng.nextFloat() * 35);
    }
    let jump = tick < this.wanderJumpUntilTick;
    let hook = tick < this.wanderHookUntilTick;

    const want = { ...this.prevInput, direction: this.wanderDir, jump: jump ? 1 : 0, hook: hook ? 1 : 0 };
    const safe = this.guard(self, want);
    const guarded = safe !== want;
    if (guarded) {
      this.wanderDir = safe.direction;
      jump = safe.jump !== 0;
      hook = safe.hook !== 0;

      if (hook) {
        this.wanderHookUntilTick = Math.max(this.wanderHookUntilTick, tick + 20);
        this.wanderAim = Math.atan2(safe.targetY, safe.targetX);
      }
      if (this.wanderDir < 0) mv.RunLeft();
      else if (this.wanderDir > 0) mv.RunRight();
      else mv.RunStop();
      this.wanderUntilTick = tick + 25;
    }

    const cur = Math.atan2(this.prevInput.targetY, this.prevInput.targetX);
    let d = this.wanderAim - cur;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    const step = Math.max(-0.12, Math.min(0.12, d));
    const a = cur + step;
    const tx = guarded && hook ? safe.targetX : Math.round(Math.cos(a) * 300);
    const ty = guarded && hook ? safe.targetY : Math.round(Math.sin(a) * 300);

    if (hook && !guarded && (self.hookedPlayer >= 0 || (self.hookState === HOOK_IDLE && this.ropeCatches(self, { ...this.prevInput, targetX: tx, targetY: ty }, this.world.allTees())))) {
      hook = false;
      this.wanderHookUntilTick = tick;
    }
    mv.Jump(jump);
    mv.Hook(hook);
    if (client.input.m_Fire & 1) mv.Fire();
    mv.SetAim(tx, ty);
    this.prevInput.targetX = tx === 0 && ty === 0 ? 300 : tx;
    this.prevInput.targetY = ty;
    this.prevInput.direction = this.wanderDir;
    this.prevInput.jump = jump ? 1 : 0;
    this.prevInput.hook = hook ? 1 : 0;
    this.noteAim(this.prevInput);
    this.recordSent(this.prevInput);
    client.sendInput();
  }

  private idle(): void {
    const client = this.client;
    if (!client) return;
    const mv = client.movement;
    mv.RunStop();
    mv.Jump(false);
    mv.Hook(false);
    if (client.input.m_Fire & 1) mv.Fire();
    this.prevInput.direction = 0;
    this.prevInput.jump = 0;
    this.prevInput.hook = 0;

    this.recordSent(this.prevInput);
  }

  private trackTravel(pos: Vec2): void {
    if (!this.travel.start) this.travel.start = { ...pos };
    if (this.lastPos) this.travel.distance += vdistance(this.lastPos, pos);
    this.lastPos = { ...pos };
    this.travel.end = this.lastPos;
  }

  private maybeLogStatus(self: { pos: Vec2; vel: Vec2; activeWeapon: number; frozen: boolean }): void {
    if (!this.cfg.verbose) return;
    const now = Date.now();
    if (now - this.lastStatusMs < STATUS_INTERVAL_MS) return;
    this.lastStatusMs = now;
    this.log(
      `status: tick=${this.world.tick} id=${this.ownId} pos=(${self.pos.x}, ${self.pos.y}) vel=(${self.vel.x.toFixed(2)}, ${self.vel.y.toFixed(2)}) weapon=${self.activeWeapon} frozen=${self.frozen} target=${this.targetId} travelled=${this.travel.distance.toFixed(0)}px ${this.statsLine()}`,
    );
  }
}
