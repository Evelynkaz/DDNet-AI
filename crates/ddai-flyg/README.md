# ddai-flyg

Формат `.flyg` v1 — подграф коннектома мухи (топология, знаки синапсов, рецептивные поля,
выходные группы), который загружает модель мухи (фаза 7). Типы, чтение/запись (`postcard` +
`zstd`, как `ddai_connectome::tables`) и структурная валидация — здесь; сам алгоритм выбора
подграфа из полного коннектома — в `crates/ddai-connectome` (модуль `subgraph`), который зависит
от этого крейта, а не наоборот.

**Без `arrow`, без зависимости на `ddai-connectome`.** Это единственное, от чего должна зависеть
будущая модель мухи (фаза 7): ей не нужны ни Arrow/Feather, ни офлайн-сборка подграфа — только
готовый `.flyg`.

Подробное описание каждого поля — `docs/formats.md` (в корне репозитория, раздел «`.flyg` v1»).
Прозу про сам алгоритм выбора подграфа (пути, знаки, рецептивные поля, AN-выбор по данным,
выходные группы) — `crates/ddai-connectome/README.md`.

## Использование

```rust
use ddai_flyg::{load, save, validate, Flyg};

let flyg: Flyg = load(Path::new("fly-S-v1.flyg"))?; // validates on load
validate(&flyg)?; // also callable standalone, e.g. before save()
save(&flyg, Path::new("out.flyg"))?; // does NOT validate first — see its own doc comment
```

## Тесты

`cargo test -p ddai-flyg` — полностью офлайн. `tests/roundtrip.rs`: сохранение/загрузка
байт-в-байт, и набор испорченных файлов/структур (не по индексу типа/ребра, поломанный CSR,
`NaN`, автапсы, несогласованная `summary`, …), каждый из которых `load`/`validate` должен
отклонить с понятной причиной, а не тихо принять или паникнуть.
