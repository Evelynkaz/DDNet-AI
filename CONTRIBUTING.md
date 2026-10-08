# Как участвовать · Contributing

Спасибо за интерес к проекту! Документация проекта — на русском, код и комментарии — на английском.
*English summary at the end.*

## Правила проекта

- **Честная игра.** Бот не обходит кики и баны (ни прокси, ни сменой IP или ника), не пишет в игровой чат сам и
  играет только там, где это разрешено. Изменения, которые помогают обходить защиту серверов или выдавать бота за
  человека, не принимаются. Подробно — раздел «Безопасность и честная игра» в [README](README.md).
- **Доказательства, а не обещания.** Изменение силы бота подтверждается замером в арене (парные сиды, интервалы,
  заранее объявленный порог) и записью в `docs/EXPERIMENTS.md`; решения — в `docs/DECISIONS.md`.
- **Побитная точность.** Физика и планировщик сверяются с эталонами (C++ DDNet, TS-оракул). Оптимизации не должны
  менять результат; изменение поведения — отдельным переключателем со своим замером.
- **Никогда не коммитить** данные коннектома, веса, демки, клипы, карты, настройки, пароли и токены (проверяют
  gitleaks и `tools/ci/no-weights.sh`).

## Как собрать и проверить

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo deny check
bash tools/ci/no-weights.sh
```

Коммиты — в стиле Conventional Commits на английском (`feat:`, `fix:`, `docs:`, `perf:`, `test:`, `refactor:`).
Один логический шаг — один коммит.

## Сообщить о проблеме

Ошибки и предложения — через Issues. Уязвимости — см. [SECURITY.md](SECURITY.md), не публикуйте их в открытых
Issues.

---

**English.** Contributions are welcome. The project's rules: fair play (the bot never evades kicks or bans, never
chats on its own, plays only where allowed); evidence over claims (strength changes need a paired arena measurement
recorded in `docs/EXPERIMENTS.md`); bit-exactness (optimisations must not change results); never commit connectome
data, weights, demos, clips, maps, settings or secrets. Run the checks above before opening a pull request. Docs are in
Russian, code and comments in English; commits follow Conventional Commits. Report vulnerabilities privately (see
[SECURITY.md](SECURITY.md)).
