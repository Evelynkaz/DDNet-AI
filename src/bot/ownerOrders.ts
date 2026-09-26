export type Order = { command: string; reply: string; args?: Record<string, string> };
export type LlmConfig = { url: string; key: string; model: string };

const L = "\\p{L}\\p{N}_";

const word = (alts: string): RegExp => new RegExp(`(?<![${L}])(?:${alts})(?![${L}])`, "iu");

export function addressed(text: string, botName: string, whisper: boolean): string | null {
  const t = text.trim();
  if (whisper) return t === "" ? null : t;
  const names = ["бот", "bot", "ботик"];
  if (botName.trim() !== "") names.push(botName.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const m = new RegExp(`^@?(?:${names.join("|")})(?![${L}])[\\s,:!.-]*(.*)$`, "iu").exec(t);
  if (m === null) return null;
  const rest = m[1].trim();
  return rest === "" ? null : rest;
}

function playerNamed(said: string, players: readonly string[]): string | null {
  const s = said.trim().replace(/^@/, "").toLowerCase();
  if (s === "") return null;
  const exact = players.find((p) => p.toLowerCase() === s);
  if (exact !== undefined) return exact;
  const starts = players.filter((p) => p.toLowerCase().startsWith(s));
  return starts.length === 1 ? starts[0] : null;
}

export function parseOrder(text: string, owner: string, players: readonly string[]): Order | null {
  const t = text.trim();

  if (/(?<![\p{L}\p{N}_])(?:не|don'?t|do not|never)(?![\p{L}\p{N}_])/iu.test(t) && !word("не лезь").test(t)) return null;

  const hit = /^(?:бей|убей|атакуй|заморозь|фризни|зафризь|килл|attack|freeze|kill|target|block)\s+(.+)$/iu.exec(t);
  if (hit !== null) {
    const who = playerNamed(hit[1], players);
    if (who !== null && who !== owner) return { command: `!target ${who}`, reply: "иду за {name}", args: { name: who } };
  }
  if (hit !== null && word("всех|всем|everyone|everybody|all").test(hit[1])) return { command: "!go", reply: "играю" };
  const self = hit === null || word("yourself|себя|self").test(hit[1]);
  if (self && word("убейся|убей себя|самоубейся|умри|суицид|kill yourself|kill|килл|/kill").test(t)) return { command: "!kill", reply: "ок" };
  if (hit !== null) return null;
  if (word("ко мне|сюда|за мной|к мне|иди ко|come(?!\\s+on)|come here|follow me|follow").test(t)) return { command: `!goto @${owner}`, reply: "иду" };
  if (word("стой|стоп|замри|жди|стоять|stop|wait|stay|hold").test(t)) return { command: "!stop", reply: "стою" };
  if (/(?<![\p{L}\p{N}_])(?:вб|wb|вейблок\p{L}*)(?![\p{L}\p{N}_])/iu.test(t)) {

    if (/лев|left/iu.test(t)) return { command: "!wb left", reply: "держу ВБ слева" };
    if (/прав|right/iu.test(t)) return { command: "!wb right", reply: "держу ВБ справа" };
    return { command: "!style wb", reply: "держу ВБ" };
  }
  if (word("дуэль|дуэли|дуэл|duel").test(t)) return { command: "!style duel", reply: "дуэль" };
  if (word("не лезь|пассив|passive").test(t)) return { command: "!mode passive", reply: "не лезу" };
  if (word("наблюдай|в наблюдатели|спек|spec|spectate").test(t)) return { command: "!spec", reply: "ухожу в наблюдатели" };
  if (word("зайди|вернись в игру|join").test(t)) return { command: "!join", reply: "захожу" };
  if (word("дефолт|default|обычн\\p{L}*|как обычно").test(t)) return { command: "!style default", reply: "играю как обычно" };
  if (word("дерись|играй|бей всех|продолжай|го играть|fight|play|go(?!\\s+to)").test(t)) return { command: "!go", reply: "играю" };
  return null;
}

const ALLOWED = /^!(?:goto @(.{1,20})|target (.{1,20})|stop|go|kill|spec|join|wb (?:left|right|off)|style (?:default|wb|duel)|mode (?:fight|passive|hold))$/u;

export function checkModelAnswer(answer: string, owner: string, players: readonly string[]): Order | null {
  const line = answer.trim().split("\n")[0].trim().replace(/^`+|`+$/g, "");
  const m = ALLOWED.exec(line);
  if (m === null) return null;
  const who = m[1] ?? m[2];
  if (who !== undefined && who !== owner && !players.includes(who)) return null;

  if (m[2] !== undefined && m[2] === owner) return null;
  return { command: line, reply: "ок" };
}

export async function modelOrder(text: string, owner: string, players: readonly string[], llm: LlmConfig, fetchImpl: typeof fetch = fetch, timeoutMs = 8000, retry = true): Promise<Order | null> {
  const system = [
    "You turn an order that a DDNet player gives his block bot in the game chat into exactly one bot command.",
    "Commands: !goto @<nick> (walk to that player), !target <nick> (fight only that player), !stop (stand still, cancel a walk), !go (play and fight as usual),",
    "!kill (respawn), !spec (go to the spectators), !join (back into the game), !wb left | !wb right | !wb off (hold the wayblock on that side),",
    "!style default | !style wb | !style duel, !mode fight | !mode passive | !mode hold.",
    `The player giving the order is "${owner}". Players on the server: ${players.map((p) => JSON.stringify(p)).join(", ")}.`,
    "Answer with the command only. If the order is none of these, answer NONE.",
  ].join(" ");
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetchImpl(`${llm.url.replace(/\/+$/, "")}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", ...(llm.key !== "" ? { authorization: `Bearer ${llm.key}` } : {}) },
      body: JSON.stringify({ model: llm.model, temperature: 0, max_tokens: 30, messages: [{ role: "system", content: system }, { role: "user", content: text }] }),
      signal: ctl.signal,
    });
    if (res.status === 429 && retry) {

      await new Promise((r) => setTimeout(r, 1100));
      return modelOrder(text, owner, players, llm, fetchImpl, timeoutMs, false);
    }
    if (!res.ok) return null;
    const body = (await res.json()) as { choices?: { message?: { content?: unknown } }[] };
    const answer = body.choices?.[0]?.message?.content;
    return typeof answer === "string" ? checkModelAnswer(answer, owner, players) : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export const FREE_LLM: LlmConfig = { url: "https://api.llm7.io/v1", key: "", model: "default" };

export function llmFrom(raw: unknown): LlmConfig | null {
  if (raw === undefined) return FREE_LLM;
  if (raw === null || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const url = typeof o.url === "string" ? o.url.trim() : "";
  const model = typeof o.model === "string" ? o.model.trim() : "";
  if (!/^https?:\/\//.test(url) || model === "") return null;
  return { url, model, key: typeof o.key === "string" ? o.key.trim() : "" };
}
