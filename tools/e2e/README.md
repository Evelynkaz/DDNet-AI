# E2E: реальный браузер для `ddai-web` (задача 5.1)

`web-login.spec.ts` — проверка в headless Chromium (Playwright): вход по паролю → статус
показывает «подключено» (WebSocket) → выход → снова форма входа. Один прогон на десктопном
viewport, один на телефонном (360×740). Скриншоты сохраняются в `~/aiddnet/data/screenshots/`.

Не входит в CI (см. constraints задачи 5.1) — запускать руками после изменений в `ddai-web`.

## Как запустить

```bash
# 1. Собрать бота (нужен target/debug/ddnet-ai)
cargo build -p ddnet-ai

# 2. Один раз поставить зависимости и headless-shell Chromium
cd tools/e2e
npm install
npx playwright install --only-shell chromium
# на новой машине могут не хватать системных библиотек (шрифты, mesa, libasound…),
# тогда один раз: sudo npx playwright install-deps chromium

# 3. Прогнать
npx playwright test
```

Тест сам поднимает `ddnet-ai web-passwd` и `ddnet-ai web --listen 127.0.0.1:0` во временный
каталог данных (не трогает `~/aiddnet/data/secrets`) и гасит процесс после прогона.
