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
  if (word("клип|сохрани клип|запиши клип|clip|save a clip").test(t)) return { command: "!clip", reply: "ок" };
  if (word("сбрось цель|забудь цель|отстань от него|оставь его|stop targeting|leave him").test(t)) return { command: "!target -", reply: "ок" };
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

export function checkModelAnswer(answer: string, owner: string, allowed: ReadonlySet<string>): Order | null {
  const line = answer.trim().split("\n")[0].trim().replace(/^`+|`+$/g, "").trim();
  if (line.length > 120 || /[\u0000-\u001f\u007f\u2028\u2029]/u.test(line)) return null;
  const m = /^!(\w+)(?:\s+(.*))?$/u.exec(line);
  if (m === null) return null;
  let word = m[1].toLowerCase();
  let rest = (m[2] ?? "").trim();
  if (word === "d") {
    const sub = /^!?(\w+)(?:\s+(.*))?$/u.exec(rest);
    if (sub === null) return null;
    word = sub[1].toLowerCase();
    rest = (sub[2] ?? "").trim();
    if (word === "d") return null;
  }
  if (!allowed.has(word)) return null;

  const named = rest.replace(/^@/, "").trim().replace(/\s+/g, " ").toLowerCase();
  if ((word === "target" || word === "war" || word === "ignore") && named !== "" && named !== "-" && named !== "off" && owner.toLowerCase().replace(/\s+/g, " ").includes(named)) return null;
  return { command: line, reply: "ок" };
}

export async function modelOrder(text: string, owner: string, players: readonly string[], llms: readonly LlmConfig[], reference: string, allowed: ReadonlySet<string>, fetchImpl: typeof fetch = fetch, timeoutMs = 8000, state = ""): Promise<Order | null> {
  const system = [
    "You turn an order that a DDNet player gives his block bot in the game chat into exactly one console command of that bot.",
    `The player giving the order is "${owner}"; "me", "I", "мне", "меня" mean him. Players on the server now: ${players.map((p) => JSON.stringify(p)).join(", ")}.`,
    ...(state !== "" ? [`The bot now: ${state}`] : []),
    "The order may be in Russian, English or any other language, with typos or in translit.",
    "Answer with the one command line only, starting with !, nothing else. If no command fits, answer NONE.",
    "!friend <nick>, !war <nick> and !ignore <nick> put a player on that list, or take him off it if he is on it already. '!friend off', '!war off' and '!ignore off' empty the whole list: only when the order says to clear or empty the list.",
    "'!target <nick>' fights only that player; '!target -' goes back to picking targets itself. '!d <command>' gives the command to the second bot.",
    `Examples: "иди ко мне" -> !goto @${owner}; "убери Chioma из друзей" -> !friend Chioma (she is a friend); "перестань бить kkv" -> !target - if kkv is the target, !war kkv if kkv is on the war list; "второй бот стой" -> !d stop; "очисти список варов" -> !war off; "скажи gg" -> !say gg.`,
    "The bot's commands:",
    reference,
  ].join("\n");
  for (const llm of llms) {
    const answer = await ask(llm, system, text, fetchImpl, timeoutMs, true);
    if (answer === null) continue;
    return checkModelAnswer(answer, owner, allowed);
  }
  return null;
}

async function ask(llm: LlmConfig, system: string, text: string, fetchImpl: typeof fetch, timeoutMs: number, retry: boolean): Promise<string | null> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetchImpl(`${llm.url.replace(/\/+$/, "")}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", ...(llm.key !== "" ? { authorization: `Bearer ${llm.key}` } : {}) },
      body: JSON.stringify({ model: llm.model, temperature: 0, max_tokens: 40, messages: [{ role: "system", content: system }, { role: "user", content: text }] }),
      signal: ctl.signal,
    });
    if (res.status === 429 && retry) {

      clearTimeout(timer);
      await new Promise((r) => setTimeout(r, 1100));
      return ask(llm, system, text, fetchImpl, timeoutMs, false);
    }
    if (!res.ok) return null;
    const body = (await res.json()) as { choices?: { message?: { content?: unknown } }[] };
    const answer = body.choices?.[0]?.message?.content;
    return typeof answer === "string" ? answer : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export const LLM_PRESETS: readonly { id: string; label: string; url: string; model: string; key: "none" | "free" | "paid" | "local" }[] = [
  { id: "llm7", label: "LLM7 (free, no key: 10 a minute, 60 an hour), then ch.at", url: "https://api.llm7.io/v1", model: "default", key: "none" },
  { id: "chat", label: "ch.at (free, no key)", url: "https://ch.at/v1", model: "gpt-4o-mini", key: "none" },
  { id: "groq", label: "Groq (free key: 30 a minute, 14 400 a day)", url: "https://api.groq.com/openai/v1", model: "llama-3.1-8b-instant", key: "free" },
  { id: "cerebras", label: "Cerebras (free key: 30 a minute, 14 400 a day)", url: "https://api.cerebras.ai/v1", model: "llama3.1-8b", key: "free" },
  { id: "gemini", label: "Google Gemini (free key)", url: "https://generativelanguage.googleapis.com/v1beta/openai", model: "gemini-flash-lite-latest", key: "free" },
  { id: "openrouter", label: "OpenRouter (free key, :free models)", url: "https://openrouter.ai/api/v1", model: "meta-llama/llama-3.3-70b-instruct:free", key: "free" },
  { id: "mistral", label: "Mistral (free key, Experiment plan)", url: "https://api.mistral.ai/v1", model: "mistral-small-latest", key: "free" },
  { id: "github", label: "GitHub Models (free with a GitHub token)", url: "https://models.github.ai/inference", model: "openai/gpt-4.1-mini", key: "free" },
  { id: "sambanova", label: "SambaNova (free key)", url: "https://api.sambanova.ai/v1", model: "Meta-Llama-3.1-8B-Instruct", key: "free" },
  { id: "nvidia", label: "NVIDIA NIM (free key)", url: "https://integrate.api.nvidia.com/v1", model: "meta/llama-3.1-8b-instruct", key: "free" },
  { id: "openai", label: "OpenAI (paid key)", url: "https://api.openai.com/v1", model: "gpt-4.1-mini", key: "paid" },
  { id: "anthropic", label: "Anthropic Claude (paid key)", url: "https://api.anthropic.com/v1", model: "claude-haiku-4-5-20251001", key: "paid" },
  { id: "ollama", label: "Ollama (a model on this PC)", url: "http://localhost:11434/v1", model: "llama3.2", key: "local" },
];

export const FREE_LLM: LlmConfig = { url: "https://api.llm7.io/v1", key: "", model: "default" };

export const CHAT_AT_LLM: LlmConfig = { url: "https://ch.at/v1", key: "", model: "gpt-4o-mini" };

export const llmChain = (llm: LlmConfig | null): LlmConfig[] => (llm === null ? [] : llm.url === FREE_LLM.url && llm.key === "" ? [llm, CHAT_AT_LLM] : [llm]);

export function llmFrom(raw: unknown): LlmConfig | null {
  if (raw === undefined) return FREE_LLM;
  if (raw === null || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const url = typeof o.url === "string" ? o.url.trim() : "";
  const model = typeof o.model === "string" ? o.model.trim() : "";
  if (!/^https?:\/\//.test(url) || model === "") return null;
  return { url, model, key: typeof o.key === "string" ? o.key.trim() : "" };
}
