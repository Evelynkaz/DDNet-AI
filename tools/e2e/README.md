# E2E: реальный браузер для `ddai-web` (задачи 5.1, 5.3)

`web-login.spec.ts` (задача 5.1) — проверка в headless Chromium (Playwright) против локального,
только что запущенного процесса: вход по паролю → статус показывает «подключено» (WebSocket) →
выход → снова форма входа. Один прогон на десктопном viewport, один на телефонном (360×740).
Скриншоты сохраняются в `~/aiddnet/data/screenshots/5.1-*.png`.

`web-login-https.spec.ts` (задача 5.3) — тот же сценарий, но по-настоящему: через реальный
Caddy/HTTPS (`https://89-58-7-133.sslip.io`, настоящий сертификат Let's Encrypt), паролем из
`~/aiddnet/data/secrets/web-password.txt` (читается в момент прогона, никогда не печатается).
Скриншоты — `~/aiddnet/data/screenshots/5.3-*.png`. Без переменной `DDAI_E2E_BASE_URL` тест
пропускается (`test.skip`), так что обычный `npx playwright test` без окружения не пытается
стучаться в интернет и остаётся зелёным (см. "Как запустить" ниже).

Ни один из двух не входит в CI (см. constraints задач 5.1/5.3) — запускать руками после изменений
в `ddai-web` или в развёртывании (`deploy/`).

## Как запустить

```bash
# 1. Собрать бота (нужен target/debug/ddnet-ai) — только для web-login.spec.ts (локального)
cargo build -p ddnet-ai

# 2. Один раз поставить зависимости и headless-shell Chromium
cd tools/e2e
npm install
npx playwright install --only-shell chromium
# на новой машине могут не хватать системных библиотек (шрифты, mesa, libasound…),
# тогда один раз: sudo npx playwright install-deps chromium

# 3a. Локальный прогон (5.1) — сам поднимает `ddnet-ai web-passwd` и
#     `ddnet-ai web --listen 127.0.0.1:0` во временный каталог данных (не трогает
#     ~/aiddnet/data/secrets) и гасит процесс после прогона. Единственный, что запускает
#     обычный `npx playwright test` без переменных окружения:
npx playwright test web-login.spec.ts

# 3b. Против настоящего HTTPS-развёртывания (5.3) — нужен уже поднятый Caddy+ddnet-ai-web
#     (deploy/install.sh) и сгенерированный пароль (ddnet-ai web-passwd):
DDAI_E2E_BASE_URL=https://89-58-7-133.sslip.io npx playwright test web-login-https.spec.ts
```
