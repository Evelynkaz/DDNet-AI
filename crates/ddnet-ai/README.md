# ddnet-ai

Единственный бинарник проекта: `play`, `web`, `arena`, `train`, `fly`, `clip`, `demo`, `dataset`, `record`, `rec`, `servers`, `servers-cache`, `proxy-check`, `launch`, `web-passwd`, `trace`, `map`. Общий обзор — в корневом `README.md`; здесь — команды, которые связывают сайт с юнитами (запуск бота, браузер серверов).

## Запуск бота с сайта и браузер серверов (задачи 5.9, 5.12; D-089, D-099)

Сайт ничего не запускает и не ходит в сеть: он пишет маленькие файлы, а эти команды (в отдельных юнитах `deploy/systemd/`, ставит `deploy/install-launcher.sh`) их проверяют и исполняют. Форматы — `docs/formats.md` §34 и §36, развёртывание — `deploy/README.md`.

| Команда | Кто запускает | Что делает |
|---|---|---|
| `launch apply` | root, `ddnet-ai-launch.service` | Читает `request.json`, **удаляет его до обработки**, проверяет каждое значение по фиксированным спискам (`local`, запись `live-servers.toml` с `ready`, или избранное из `launch/favourites.json`: публичный `ip:порт`, ник, `direct` / `proxy:<имя>`, согласие владельца), память банов (бан закрывает все адреса с тем же IP до «Открыть снова» / правки списка), паузы и лимиты; пишет окружение и drop-in cgroup (для `relay = "public"` — запрет IP игрового сервера, 2.6b) и вызывает `systemctl`. Прокси и IP **никогда не подбираются сами**: нет файла прокси — отказ. |
| `launch exited` | root, `ExecStopPost=` юнита бота | Записывает, чем кончился бот (код 3/4 → бан-память), пишет статус и `/run/ddnet-ai/blocked.json` для страницы. |
| `launch check-proxy` | `ubuntu`, `ddnet-ai-proxycheck.service` | Кнопка «Проверить»: читает `launch/proxycheck-request.json`, делает `proxy-check` (в игровой сервер не уходит ничего), пишет `launch/proxycheck-result.json` с кодами и числами, без адресов и учётных данных. |
| `servers-cache` | `ubuntu`, `ddnet-ai-servers.service` | Единственный, кто ходит на мастер-серверы DDNet (HTTPS, ≤ 8 МиБ, без редиректов); пишет ограниченный, строго разобранный кэш `data/servers/master.json` (публичные IPv4, чистый текст), не чаще раза в 60 с. |
| `proxy-check --proxy <имя>` | руками | Проверка прокси из терминала (2.6, 2.6b). |
| `servers [--block]` | руками | Таблица списка мастеров (только чтение; метки `READY` / `listed` — из `live-servers.toml`). |

Файлы: `src/launch_cmd.rs` (помощник и `check-proxy`), `src/servers_cache_cmd.rs` (загрузка кэша), `src/proxy_cmd.rs` (`proxy-check` и подготовка клиента: `prepare_client` добавляет избранное в ворота бота и назначает прокси).

Проверка: `cargo test -p ddnet-ai` (модульные: избранное и бан-память помощника, кэш; `tests/launch_apply.rs` — настоящий помощник с поддельным `systemctl`; `tests/check_proxy.rs`, `tests/servers_cache.rs` (против loopback-стенда), `tests/deploy_units.rs` — **веб-юнит без сетевых прав**, новые юниты — малые и без root). Локально целиком: `tools/e2e/servers-e2e.sh` (частный сервер на 127.0.0.1:8463; `tools/e2e/README.md`).

Cargo-признак `loopback-favourites` (по умолчанию выключен; `cargo test` включает его dev-зависимостью) нужен только e2e: он разрешает избранное на loopback. Обычная сборка такого избранного не принимает.
