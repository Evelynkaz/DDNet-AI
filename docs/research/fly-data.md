# Данные коннектома дрозофилы для «мухи» (фаза 0)

Дата проверки: 2026-09-27. Всё, что ниже не помечено «не проверено», проверено сегодня:
HTTP-запросами (HEAD, листинги GCS JSON API, Range-чтение футеров Arrow), скачиванием
малых файлов и анонимными запросами к neuPrint. Идентификаторы и имена колонок даны как в оригинале.

Скачано за сессию ≈ 293 МБ (лимит 300 МБ): 89,5 МБ целых файлов (лежат в
`~/aiddnet/data/connectome/samples/`) и ≈ 201 МБ Range-чтений (футеры, отдельные батчи; на диск не сохранялись).
Из этих 201 МБ ≈ 122 МБ ушло на одну неудачную пробу (первые батчи `Neuprint_Neurons.feather` оказались очень крупными).

---

## 0. Коротко

* **Основной источник: MaleCNS v1.0** (Janelia FlyEM, Cambridge, MRC LMB, Google Research). Лицензия **CC-BY 4.0**.
  Готовые плоские таблицы в формате Arrow Feather v2 (сжатие LZ4_FRAME) лежат в публичном бакете GCS; скачивать можно без авторизации.
* **Минимальный набор (3 файла, 565 791 790 байт ≈ 566 МБ):**
  1. `body-annotations-male-cns-v1.0-minconf-0.5.feather`: типы, superclass/class, стороны, группы (14,5 МБ);
  2. `body-neurotransmitters-male-cns-v1.0.feather`: медиатор на нейрон и на тип (43,3 МБ);
  3. `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather`: рёбра нейрон→нейрон с числом синапсов между всеми 165 122 Traced-телами (508 МБ, 25 563 197 рёбер).
* ROI (нейропили) в этих файлах отсутствуют. Если нужны, есть три пути: neuPrint API (без токена сейчас отвечает только `male-cns:v0.9`),
  большие файлы (≥ 3 ГБ) или Codex (нужен вход).
* FlyWire FAFB v783 и BANC v888 годятся для перекрёстной проверки. У FlyWire противоречивые лицензии: на странице FlyWire указано **CC BY-NC 4.0**, на Zenodo **CC-BY 4.0**.
  MANC полностью покрыт MaleCNS (VNC того же пола), поэтому нужен только для сверки.

---

## 1. MaleCNS v1.0

### 1.1 Статус релиза, публикации, лицензия

* Сайт: https://male-cns.janelia.org/ . Страница загрузки: https://male-cns.janelia.org/download/
  * Release notes: **v1.0 (June 8, 2026)**: «Minor proofreading changes, Refinement of neuron annotations»; v0.9 вышла 5 октября 2025.
  * Новости на главной: «2026-09-03 — MaleCNS paper published!»
* Статья (проверено через Crossref): Berg, Beckett, Costa, Schlegel, … Rubin, Jefferis (111 авторов),
  *«Sexual dimorphism in the complete Drosophila male central nervous system connectome»*, **Cell 189(18):5504–5526.e15**, сентябрь 2026,
  DOI **10.1016/j.cell.2026.08.015**. Статья распространяется под CC-BY 4.0. Препринт: bioRxiv 10.1101/2025.10.09.680999 (v2 от 30.10.2025).
* Пост в блоге Google Research **от 3 сентября 2026** (проверено): «A connectomics milestone: Mapping the complete male fruit fly brain»
  (авторы Michał Januszewski, Viren Jain), https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/ .
  Цифры из поста: «166,000+ neurons, 125 million synaptic connections». Наша проверка: в `syn-partners-…-traced-only` ровно 124 025 046 пар синапсов.
* **Лицензия:** на странице загрузки написано «The Male CNS is licensed under CC-BY» со ссылкой на https://creativecommons.org/licenses/by/4.0/ .
  Отдельного текста для цитирования на сайте нет. Нужно цитировать статью в Cell и указывать источник (CC-BY требует атрибуции).
  Для GPL-проекта это совместимо: данные мы не распространяем, а атрибуцию указываем в README и NOTICE.

### 1.2 neuPrint API

* Страница загрузки предлагает `Client("https://neuprint.janelia.org", dataset='male-cns:v1.0', token=token)`.
* `GET https://neuprint.janelia.org/api/serverinfo` (проверено) возвращает `IsPublic: true` и объявление
  **«neuPrint has moved to a new authorization system. Existing API tokens no longer work…» (announcement-id `auth-migration-2026-08`)**.
* Анонимный `GET /api/dbmeta/datasets` показывает `hemibrain:v1.2.1`, **`male-cns:v0.9`**, `manc:v1.0`, `manc:v1.2.1`, `manc:v1.2.3`,
  `mushroombody`, `optic-lobe:v1.0.1`, `optic-lobe:v1.1`. **Датасета `male-cns:v1.0` в этом списке нет.**
  Анонимный `POST /api/custom/custom` с `dataset: male-cns:v1.0` возвращает 404 `dataset "male-cns:v1.0" not available in stores`.
* Анонимные Cypher-запросы к `male-cns:v0.9` и `manc:v1.2.3` **работают без токена** (проверено).
  Доступен ли v1.0 с новым токеном, **не проверено**: токена у нас нет.
* Вывод: neuPrint подходит для разовых исследовательских запросов (ROI, сверка). В пайплайн его не закладываем:
  авторизация меняется, v1.0 без токена недоступен, а воспроизводимость хуже, чем у зафиксированных файлов.

### 1.3 Все файлы `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`

HTTP-адрес: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/<file>`.
Метаданные взяты из GCS JSON API (`https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o?prefix=v1.0/connectome-data/`)
и совпадают с заголовками HEAD (`x-goog-hash`, `x-goog-generation`, `content-length`). Число строк получено из футера и метаданных батчей Arrow.

| файл | байт | строк | md5 (hex) | crc32c (b64) | generation | обновлён |
|---|---:|---:|---|---|---|---|
| body-annotations-male-cns-v1.0-minconf-0.5.feather | 14 483 314 | 211 577 | 50a7718770c57220f160ba4f431ab89e | vjz9cg== | 1780494878811468 | 2026-06-03 |
| body-neurotransmitters-male-cns-v1.0.feather | 43 282 834 | 1 835 518 | 3d842b12fe5c49eefade528d7dd24a1f | jcpNFg== | 1780894899156750 | 2026-06-08 |
| body-stats-male-cns-v1.0-minconf-0.5.feather | 778 062 826 | 88 384 522 | 404c3349c28580148e16815eb99f382a | MGOCPQ== | 1780494888472305 | 2026-06-03 |
| connectome-weights-male-cns-v1.0-minconf-0.5.feather | 1 051 241 946 | 151 856 684 | f30e9dcca25cfd021bf1e7b3d975599e | dKRPVQ== | 1780494887545976 | 2026-06-03 |
| connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather | 508 025 642 | 25 563 197 | 6601d4ad0afa99fd03eb087965ef2423 | iakIVA== | 1780494884279095 | 2026-06-03 |
| connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather | 502 169 298 | 25 568 639 | 09f2f833f7161a46ad33cd6f9c9072f6 | RNohUA== | 1780494884449936 | 2026-06-03 |
| syn-partners-male-cns-v1.0-minconf-0.5.feather | 6 777 179 098 | не проверено | 58efcf712f8c4d4de5f2ad51e97def76 | jTlNIA== | 1780494942562468 | 2026-06-03 |
| syn-partners-male-cns-v1.0-minconf-0.5-traced-only.feather | 2 965 367 002 | 124 025 046 | f5bc1c5ce34a01b68956414b530edda8 | 29gh0g== | 1780494912394119 | 2026-06-03 |
| syn-partners-male-cns-v1.0-minconf-0.5-significant-only.feather | 2 965 702 122 | не проверено | a5bcf46d8a8b825ab8ca3273c47795e3 | 49jLBw== | 1780494917774871 | 2026-06-03 |
| syn-points-male-cns-v1.0-minconf-0.5.feather | 13 061 489 098 | не проверено | c69d08758de07582035cc8843574493a | 1E7gRw== | 1780494991007477 | 2026-06-03 |
| tbar-neurotransmitters-male-cns-v1.0.feather | 2 651 680 218 | 45 656 140 | 51b02c11690662aedef28f86d394ff0d | RrR5/g== | 1780894927871192 | 2026-06-08 |

На странице загрузки файлы `-traced-only` и `-significant-only` **не перечислены**. Они есть в бакете, и их смысл понятен из кода экспорта
`janelia-flyem/flyem-snapshot`, `flyem_snapshot/outputs/flat.py`:
`MIN_SIGNIFICANT_STATUS = "Sensory Anchor"`, `MIN_TRACED_STATUS = "Leaves"`. Файл `traced-only` содержит рёбра, у которых **оба** конца имеют
`statusLabel >= "Leaves"` (это упорядоченная категория). Мы проверили: это ровно 165 122 тела со `status == "Traced"`.
`significant-only` отбирает `>= "Sensory Anchor"` (165 424 тела), то есть добавляет Anchor-тела.
Страница загрузки описывает `body-stats` как «summary statistics (synapse counts) of all segments».

Прочие каталоги бакета, для справки:
* `v1.0/database/neuprint-inputs/`: CSV и Feather, из которых собран neo4j. Среди них `Neuprint_Neurons.feather`
  (4 648 794 426 Б, 88 682 452 строки, 53 колонки, включая `roiInfo:string` (JSON по ROI) и NT-поля `predictedNt`, `consensusNt` и др.);
  `Neuprint_Neuron_Connections.feather` (3 530 452 946 Б, 151 856 684 строки: `:START_ID(Body-ID)`, `:END_ID(Body-ID)`, `weightHR:int`,
  `weight:int`, `weightHP:int`, `roiInfo:string` — **веса по ROI для каждого ребра**); `Neuprint_Neuron_Connections.csv` (16,9 ГБ) и др.
  У больших объектов нет `md5Hash` (это composite upload), есть только `crc32c`.
  Число строк `Neuprint_Neuron_Connections` совпадает с полной `connectome-weights`, так что `weight` из neuPrint, по-видимому,
  равен весу при `minconf 0.5` (не проверено поштучно).
* `v1.0/database/neo4j/`: готовая БД neo4j 4.4.16.
* `v1.0/nblasts/`: сопоставления с FlyWire783. Например, `matches_mcns_v1.0_flywire783.feather` (7,2 МБ, md5 b64 `v/seo5hDXzdPfiKngGC73w==`),
  `matches_flywire783_mcns_v1.0.feather` (6,4 МБ); полные матрицы NBLAST занимают 0,6–21 ГБ.
* ROI-объёмы: `gs://flyem-male-cns/rois/fullbrain-roi-v4` (мозг, 96 меток, 256 нм), `gs://flyem-male-cns/rois/malecns-vnc-neuropil-roi-v0` (VNC, 27 меток).
  Нейроглансер-сцена: `gs://flyem-male-cns/v1.0/male-cns-v1.0.json`.
* Скелеты SWC: `gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/<bodyId>.swc`. Для нашей задачи не нужны.
* `README_RELEASE_BUCKET.md` в корне бакета устарел: в нём ещё написано про v0.9.

### 1.4 Схемы (проверено)

**body-annotations-…-minconf-0.5.feather** (скачан; sha256 `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`, md5 совпал с GCS)
211 577 строк, `bodyId` уникален, 36 колонок:
`assignedOlHex1: double, assignedOlHex2: double, bodyId: int64, flywireType: string, group: double, instance: string, somaSide: string,
statusLabel: dictionary<string, int8, ordered>, superclass: string, type: string, vfbId: string, hemibrainType: string, itoleeHl: string,
supertype: string, birthtime: string, mancBodyid: double, mancGroup: double, mancType: string, subclass: string, synonyms: string,
class: string, rootSide: string, somaNeuromere: string, trumanHl: string, dimorphism: string, matchingNotes: string, entryNerve: string,
mancSerial: double, mcnsSerial: double, serialMotif: string, fruDsx: string, exitNerve: string, receptorType: string,
somaLocation: list<int64>, tosomaLocation: list<int64>, status: string`.
* Значения `status`: Traced 165 122, Orphan 15 925, Glia 11 864, Unimportant 10 751, NaN 5 472, Assign 1 832, Anchor 611.
* `somaSide`: L / R / M / NaN. `instance` содержит суффикс `_L`, `_R` или `_M` и синонимы в скобках, например `DNp01(GF)_R`.
* `group` связывает гомологов слева и справа (float, но по смыслу это ID; у Traced заполнен в 145 524 случаях).
* Колонки `hemilineage` нет, есть `itoleeHl` и `trumanHl`. ROI в этом файле нет.

**body-neurotransmitters-male-cns-v1.0.feather** (скачан; sha256 `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621`, md5 совпал)
1 835 518 строк (все сегменты), `body` уникален:
`body: int64, cell_type: string, total_nt_predictions: int32, predicted_nt_confidence: double, predicted_nt: string, ground_truth: string,
celltype_total_nt_predictions: int32, celltype_predicted_nt: string, celltype_predicted_nt_confidence: double, consensus_nt: string`.
* Классы: acetylcholine, glutamate, gaba, histamine, dopamine, serotonin, octopamine, `unclear`. **Гистамин есть** (7 медиаторов).
* У 164 620 из 165 122 Traced есть строка, у 502 нет.
* `predicted_nt_confidence` у Traced: медиана 0,934, минимум 0,187, IQR 0,83–0,97.
* **`consensus_nt` задан на уровне типа**: ни у одного из 11 751 типа нет больше одного значения. Судя по всему, в нём объединены
  предсказания и ground truth (например, у дофамина: `predicted` 4 443 против `consensus` 392). Точное определение в методах статьи, **не проверено**.
* `consensus_nt` у Traced: ACh 103 718, Glu 29 296, GABA 22 055, His 5 910, unclear 3 100, NaN 502, DA 392, OA 101, 5-HT 48.

**connectome-weights-…-traced-only.feather** (только футер и 3 батча по Range)
`body_pre: int64, body_post: int64, weight: int64, type_pre: string, type_post: string`, 25 563 197 строк, 391 батч по 65 536 строк.
В полном `connectome-weights-…minconf-0.5.feather` нет колонок `type_*` (только `body_pre, body_post, weight`), зато 151 856 684 строки.
**Рёбра отсортированы по `weight` по убыванию** (проверено на всех трёх вариантах файла).
Распределение порогов в traced-only (проверено бинарным поиском по батчам):

| порог | рёбер | байт от начала файла до конца нужного батча |
|---|---:|---:|
| weight ≥ 2 | 15 270 273 | 327 898 328 |
| weight ≥ 3 | 10 511 038 | 235 211 144 |
| weight ≥ 5 | 6 235 682 | 143 947 264 |
| weight ≥ 10 | 2 749 407 | 65 363 264 |

(Для сравнения: Codex показывает у «MCNS v1.0» 166 700 нейронов и 6 242 118 connections, что близко к нашему порогу ≥ 5.)

**tbar-neurotransmitters-male-cns-v1.0.feather** (футер): предсказания на каждый пресинапс, 45 656 140 строк:
`point_id: uint64, x,y,z: int32, conf: float, sv: int64, body: int64, major: dict, primary: dict, nt_acetylcholine_prob … nt_serotonin_prob: float (7 шт.), split: dict`.

**syn-partners-…-traced-only.feather** (футер): `x_pre,y_pre,z_pre: int32, body_pre: int64, conf_pre: float, x_post,y_post,z_post: int32, body_post: int64, conf_post: float, primary_post: dictionary<int16>`,
124 025 046 строк. Это единственный плоский источник ROI для каждого синапса (`primary_post`).

**body-stats-…-minconf-0.5.feather** (футер): `body, pre, post, status_fine, superclass, class, type, instance, downstream, synweight, rank`, 88 384 522 строки.
ROI здесь тоже нет, для нас файл лишний.

---

## 2. FlyWire FAFB (самка, только мозг)

* Последний публичный релиз: **материализация 783** (снимок октября 2023). Это указано на https://flywire.ai/guidelines .
  Более нового публичного релиза FAFB нет. Материализация 888 относится к BANC, а не к FAFB.
* **Лицензия, противоречие:**
  * https://flywire.ai/guidelines: «FlyWire's public release data is made available under license **CC BY-NC 4.0**… all data available in Codex for snapshot 783 is publicly released»;
  * https://flywire.ai/tos: правки и аннотации пользователей «will be made freely available under a CC-BY-NC 4.0 license»;
  * Zenodo 10.5281/zenodo.10676866 (connectivity v783): метаданные говорят **`cc-by-4.0`**;
  * у репозитория `flyconnectome/flywire_annotations` на GitHub **нет файла LICENSE** (API возвращает `license: null`).
  * Вывод: для GPL-проекта считаем FlyWire **NC** (самый строгий вариант). Использовать только для сверки, данные и производные таблицы не распространять,
    в модель веса из FlyWire не вкладывать. Если понадобится, спросить у FlyWire (flywire@princeton.edu).
* Цитирование: Dorkenwald et al. 2024 (Nature), Schlegel et al. 2024 (Nature), для NT Eckstein, Bates et al. 2024 (Cell),
  для аннотаций версий 3.x ещё Berg et al. и Matsliah et al. 2024.
* **Codex** (https://codex.flywire.ai/api/download): **для скачивания нужен вход через Google-аккаунт**. Проверено: страница просит «Sign in to browse brain maps».
  Точный список и размеры файлов Codex (neurons, classification, connections, NT и т. п.) **не проверены**.
  Codex перечисляет датасеты: FAFB v783 (139 255 нейронов, 3 732 460 connections), BANC v888, MANC v1.2.1, MAOL v1.1, MCNS v1.0.

### 2.1 Файлы без логина

| файл | URL | байт | контрольная сумма | строк / схема |
|---|---|---:|---|---|
| proofread_connections_783.feather | https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content | 852 022 274 | md5 f48f972d262323a102aed49af1396b8a | 16 847 997; `pre_pt_root_id, post_pt_root_id: int64, neuropil: string, syn_count: int64, gaba_avg, ach_avg, glut_avg, oct_avg, ser_avg, da_avg: double` (одна строка на пару нейронов **в каждом нейропиле**, без порога) |
| per_neuron_neuropil_count_pre_783.feather | …/files/per_neuron_neuropil_count_pre_783.feather/content | 16 853 770 | md5 90fcdb42c1ba05ed92820840fa1e6ba0 | 2 781 037; `pre_pt_root_id, neuropil, count` |
| per_neuron_neuropil_count_post_783.feather | …/files/per_neuron_neuropil_count_post_783.feather/content | 233 843 050 | md5 bb5999f10920ade803d9f37097a43a56 | не проверено |
| proofread_root_ids_783.npy | …/files/proofread_root_ids_783.npy/content | 1 114 168 | md5 e0e6c19732fd8c7a4e39a2d170105421 | — |
| flywire_synapses_783.feather | …/files/flywire_synapses_783.feather/content | 9 492 998 242 | md5 f8f1b97c9d4b0ea9b4c8b287f6b99091 | ~130 млн синапсов |
| Supplemental_file1_neuron_annotations.tsv (v3.1.0) | https://raw.githubusercontent.com/flyconnectome/flywire_annotations/v3.1.0/supplemental_files/Supplemental_file1_neuron_annotations.tsv | 31 718 505 | git blob sha1 afea3e15a5671f5da0b9f7dd2e932d328c3b57a0; sha256 9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be | 139 248 (скачан) |

Zenodo 10676866: «FlyWire Whole-brain Connectome Connectivity Data», version 783.0, опубликован 2024-06-02, Range-запросы поддерживаются (206).
Колонки TSV (v3.1.0, релиз 2026-07-21, синхронизирован с MaleCNS v1.0): `supervoxel_id, root_id, pos_x,pos_y,pos_z, soma_x,soma_y,soma_z, nucleus_id,
flow, super_class, cell_class, cell_sub_class, supertype, cell_type, hemibrain_type, ito_lee_hemilineage, hartenstein_hemilineage,
top_nt, top_nt_conf, known_nt, known_nt_source, side, nerve, vfb_id, fbbt_id, status, dimorphism, matching_notes, fru_dsx, synonyms`.
* `super_class`: optic 77 541, central 32 383, sensory 16 907, visual_projection 8 038, ascending 1 750, descending 1 303,
  sensory_ascending 612, visual_centrifugal 524, motor 110, endocrine 80.
* `top_nt` содержит **6 классов, гистамина нет**: ACh 86 193, Glu 24 875, GABA 19 171, DA 5 909, 5-HT 2 282, OA 216, NaN 602.
  Серотонин сильно перепредсказан, в том числе среди DN (65 против 2 в MaleCNS).
* Сторонняя компиляция в бакете lee-lab (см. раздел 3): `compiled_data/fafb_783/fafb_783_simple_edgelist.feather` (302 625 658 Б,
  md5 60f1b7ee30a7d6c428122aa58a9ba6e0, generation 1777046058501477, 15 023 799 строк) и `fafb_783_meta.feather` (13 539 866 Б, md5 8ec1ea8ccf68c58abf85d3ca0d7f37ec, generation 1787944265562460).

---

## 3. BANC v888 (самка, мозг и нервный тяж)

* Статья: Bates, Phelps, Kim, Yang et al., *«Distributed control circuits across a brain-and-cord connectome»*, **Nature 2026**
  (Crossref: 2026-06-08), DOI 10.1038/s41586-026-10735-w, open access.
* Статичный цитируемый снимок: **Harvard Dataverse DOI 10.7910/DVN/7WTH1N**. Так указано в README `htem/BANC-project`;
  в поисковой выдаче встречался другой DOI (8TFGGB), он, вероятно, ошибочный.
  Лицензия данных на Dataverse **CC BY 4.0** (по README). API Dataverse вернул нам 403, поэтому список файлов **не проверен**.
* Около 188 тыс. нейронов (`banc_888_meta` содержит 188 508 строк) и примерно 199 млн предсказанных синапсов. Codex (нужен вход) показывает «158,262 neurons».
* Публичный бакет **без авторизации**: `gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/`
  (HTTP: `https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/<file>`).
  **Файлы в нём перезаписываются**: `meta` обновлялся 2026-06-26, 06-29, 08-21, рядом лежат бэкапы. Для закрепления версии нужны generation и md5.
  Лицензия бакета явно не указана (не проверено).

| файл | байт | md5 (hex) | generation | строк / схема |
|---|---:|---|---|---|
| banc_888_meta.feather | 57 503 026 | 8c2b93a608c7163ec9d94d68e0756ff4 | 1787336614757441 | 188 508 × 81 (все string, кроме метрик): `banc_888_id, root_id, …, side, region, hemilineage, nerve, flow, super_class, cell_class, cell_sub_class, cell_type, fafb_cell_type, manc_cell_type, malecns_cell_type, …, neurotransmitter_predicted, neurotransmitter_score, neurotransmitter_verified, …` |
| banc_888_edgelist_simple_v3.feather | 359 161 658 | 08542b0771db7418ed474be60dc9886c | 1786578377086929 | 13 620 865; `pre: string, post: string, count: int32, norm: double, post_count: int32, pre_count: int32` |
| banc_888_edgelist_split_v3.feather | 940 605 786 | 1e177918f090aaf895d59faf0a4534a5 | 1786578383288924 | связность компартментов (аксон/дендрит) |
| banc_888_neurotransmitter_prediction_v2.csv | 21 107 592 | 4ebbd1d6e05d4192ad0c6db27739a8e3 | 1778713090033523 | `root_id, acetylcholine, dopamine, gaba, glutamate, histamine, octopamine, serotonin, tyramine, neurotransmitter_predicted, neurotransmitter_score, count, supervoxel_id, position, cell_type, cell_type_neurotransmitter_predicted, cell_type_neurotransmitter_score` (8 классов, включая гистамин и тирамин) |
| banc_888_synapses_v3_enriched.parquet | 19,7 ГБ | — | — | синапсы |

Там же лежат компиляции `compiled_data/malecns_09/` (устаревшая v0.9, 3,4 ГБ edgelist), `manc_121/`, `fafb_783/`, `hemibrain_121/`, `fanc_1116/`.

---

## 4. MANC (самец, только VNC)

* Janelia: https://www.janelia.org/project-team/flyem/manc-connectome, «The MANC is licensed under **CC-BY**».
  Цитировать Takemura et al. 2024, Marin et al. 2024, Cheong et al. 2025 (eLife); так указано в описании датасета на neuPrint.
* neuPrint: `manc:v1.2.3` (последний; 23 665 Traced-нейронов), `manc:v1.2.1`, `manc:v1.0`. **Анонимные запросы работают.**
* Плоские экспорты: `gs://flyem-manc-exports/v1.0/` (только v1.0, 2023):
  `manc-traced-adjacencies-v1.0/traced-connections.csv` (75 262 163 Б, md5 c543dc9a0c367d9f7c88901d892d78a0; `bodyId_pre,bodyId_post,weight`),
  `traced-neurons.csv` (629 548 Б, md5 69a79aceb7e054885f4e47364f239c6a; `bodyId,type,instance`),
  `traced-connections-per-roi.csv` (159 МБ; `bodyId_pre,bodyId_post,roi,weight`), `manc-v1.0-neuron-properties.feather` (17 МБ).
  Для v1.2: `gs://manc-seg-v1p2/manc-v1.2-synapse-partners-minconf-0.0.feather` (1,94 ГБ).
* Компиляция lee-lab: `compiled_data/manc_121/manc_121_meta.feather` (1 446 722 Б, 23 650 строк, md5 631f0fecb9c6fbdfa9a8171124f73e34),
  `manc_121_simple_edgelist.feather` (87 386 906 Б, 5 305 354 строк, md5 ee4a344df499899d4418f1763bc39a30; `pre, post, count, norm, total_input`).
* Нам MANC не нужен: VNC уже входит в MaleCNS, а у 1 302 из 1 314 DN и 1 833 из 1 846 AN в MaleCNS заполнен `mancType`.

---

## 5. Подсчёты (MaleCNS v1.0, по `body-annotations`, только `status == Traced`)

Всего 165 122 Traced-тела, из них 162 517 с `type`; 11 751 тип.

| superclass | нейронов | типов |
|---|---:|---:|
| ol_intrinsic | 89 390 | 271 |
| cb_intrinsic | 32 160 | 6 605 |
| vnc_intrinsic | 13 151 | 2 799 |
| **visual_projection** | **9 201** | **346** |
| vnc_sensory | 6 365 | 171 |
| cb_sensory | 4 868 | 158 |
| ol_sensory (фоторецепторы) | 4 114 (из 6 098) | 11 |
| **ascending_neuron** | **1 846** | **567** |
| **descending_neuron** | **1 314** | **480** |
| vnc_motor | 708 | 142 |
| visual_centrifugal | 563 | 108 |
| sensory_ascending | 537 | 28 |
| прочие (cb_motor 107, vnc_efferent 94, endocrine, ENS, *_tbc …) | ~420 | |

**DN** (`superclass == descending_neuron`): 1 314 нейронов, 480 типов; стороны L 656, R 648, M 10.
Семейства (нейронов / типов): DNge 429/161, DNg 427/133, DNp 162/74, DNpe 158/52, DNa 32/15, DNb 24/9, DNae 20/10, DNbe 16/7,
DNde 14/6, DNd 8/4, DNc 4/2, MDN 4/1. Ещё 12 DN с типом без префикса «DN»: pIP1 (`mancType` DNxl053), aSP, pMP, CB, DNxl.
Помимо этого есть `sensory_descending` 12, `efferent_descending` 4 и `descending_neuron_tbc` 2.
Все проверенные классические типы на месте, по одному на сторону: DNa01 (`DNa01(VES006)_L/R`), DNa02, DNa03, DNa04 (PS015), DNa07, DNb05, DNb06,
DNp01 (Giant Fiber), DNp03, DNp09, DNg13, DNg29, DNge104.
Медиатор DN (`consensus_nt`): ACh 931, GABA 241, Glu 104, unclear 36, 5-HT 2.

**VPN** (`visual_projection`): 9 201 нейрон, 346 типов (L 4 589, R 4 612). Семейства: LC 4 164/45, MeTu 1 009/13, MeVP 811/62,
LLPC 761/5, LoVP 676/102, TmY 477/1, LPLC 417/4, LPC 380/3, **LT 164/39**, LPT 91/16, aMe 77/11, MeVPMe 47/8, OCG 20/9,
VS 18 и HS* по 2. Медиатор: ACh 7 889, Glu 794, unclear 481, GABA 36.

**AN** (`ascending_neuron`): 1 846 нейронов, 567 типов (AN* 1 608/478, ANXXX 211/75, IN 15). Медиатор: ACh 1 271, GABA 407, Glu 127, unclear 37.

**FlyWire v783** для сравнения: DN 1 303 / 473 типа, VPN 8 038 / 326, AN 1 750 / 567 (141 без типа).

**«Ядро»** CX + LAL + AOTU + PS (IPS, SPS) + GNG посчитано анонимным neuPrint по **v0.9**, это оценка.
Критерий: Traced-нейрон, у которого в ROI группы ≥ 100 синапсов (pre+post) либо ≥ 20 % синапсов и ≥ 20 штук. «Большинство» означает ≥ 50 % синапсов в группе.

| группа ROI | нейронов | из них «большинство» | типов |
|---|---:|---:|---:|
| CX (EB, FB, PB, NO, AB) | 2 988 | 2 804 | 304 |
| LAL | 2 912 | 471 | 999 |
| AOTU | 1 969 | 702 | 219 |
| PS (IPS + SPS) | 5 191 | 1 016 | 1 745 |
| GNG | 7 643 | 3 259 | 2 330 |
| **объединение** | **17 501** | **8 955** | **4 217** |

Состав объединения по superclass: cb_intrinsic 10 991, VPN 1 763, cb_sensory 1 382, AN 1 323, DN 1 258, sensory_ascending 333, VC 281, прочие около 170.
**Подграф VPN ∪ DN ∪ AN ∪ ядро** (v0.9): **25 518 нейронов, 988 282 ребра с weight ≥ 5, 18,4 млн синапсов**.
На уровне типов он значительно меньше, порядка 5 тыс. типов (точное число не посчитано).
Числа superclass в v0.9 почти совпадают с v1.0 (VPN 9 201, DN 1 314, AN 1 846 в обеих версиях).

---

## 6. Рекомендация

### 6.1 Минимальный набор (MaleCNS v1.0)

| # | файл | зачем | байт |
|---|---|---|---:|
| 1 | `body-annotations-male-cns-v1.0-minconf-0.5.feather` | узлы: `bodyId, type, instance, superclass, class, subclass, somaSide, group, status, mancType, flywireType` | 14 483 314 |
| 2 | `body-neurotransmitters-male-cns-v1.0.feather` | знак синапса: `consensus_nt` (на тип), `predicted_nt` + `predicted_nt_confidence` (на нейрон), `ground_truth` | 43 282 834 |
| 3 | `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` | рёбра `body_pre → body_post`, `weight` (число синапсов), `type_pre/type_post` | 508 025 642 |
| | **итого** | | **565 791 790 (≈ 566 МБ, 540 МиБ)** |

После скачивания сразу строим производные таблицы (узлы и рёбра подграфа, агрегаты по типам) в Parquet или Feather. Сырые файлы храним,
но в репозиторий не кладём. Порог `weight ≥ 5` (или ≥ 3) выбираем при сборке подграфа, а не при скачивании.

Опционально:
* **ROI** (если ядро выбираем по нейропилям, а не по графовой близости VPN → DN):
  (а) neuPrint `fetch_neurons` с roiInfo для нескольких десятков тысяч bodyId; для v1.0 нужен новый токен, без токена есть только v0.9;
  (б) `syn-partners-…-traced-only.feather` (2,97 ГБ): агрегировать `primary_post` по `body_post` и `body_pre`;
  (в) `Neuprint_Neuron_Connections.feather` (3,53 ГБ): `roiInfo` по рёбрам.
  В `Neuprint_Neurons.feather` Traced-нейроны сосредоточены в первых батчах (около 96 % в первых 4 батчах, это 121 МБ Range),
  но порядок строк не гарантирован, поэтому на этот вариант лучше не полагаться.
* Сверка с FlyWire (только локально, NC): TSV v3.1.0 (31,7 МБ) и `proofread_connections_783.feather` (852 МБ) или компиляция lee-lab (303 МБ).
  Соответствие типов берём из `flywireType` в аннотациях MaleCNS (у 141 169 Traced заполнено; в 95 345 случаях совпадает с `type`)
  или из `v1.0/nblasts/matches_mcns_v1.0_flywire783.feather`.
* Сверка с BANC: `banc_888_meta.feather` + `banc_888_edgelist_simple_v3.feather` + NT CSV, около 438 МБ.

Экономный вариант для медленного канала: traced-only отсортирован по убыванию `weight`, поэтому Range `bytes=0-143947263`
плюс футер (последние ~100 КБ) дают **все рёбра с weight ≥ 5** (6 235 682 шт.) за 144 МБ.
Минус: md5 всего файла такую частичную загрузку не проверит, придётся хранить собственный sha256 диапазона.
Рекомендую скачивать файл целиком: 508 МБ немного.

### 6.2 Как закрепить версию в скрипте загрузки

Манифест, например `data/connectome/manifest.json` или `.toml`, содержит для каждого файла:
`name, url, gcs_generation, size, md5_hex, crc32c_b64, sha256, license, source_version, citation`.

* **GCS (MaleCNS, lee-lab, MANC):** скачивать по адресу `https://storage.googleapis.com/<bucket>/<object>?generation=<N>`.
  Проверено: с верным generation приходит 200, с неверным 404. Если объект перезапишут, закреплённый адрес перестанет работать
  (бакеты без версионирования), и это правильно: скрипт упадёт, а не молча возьмёт новые данные.
  До скачивания делаем HEAD и сравниваем `content-length`, `x-goog-generation`, `x-goog-hash: md5=…`. После скачивания проверяем md5 (hashlib) и **sha256**, записанный в манифест.
  У всех трёх файлов минимального набора md5 есть (у composite-объектов, например `Neuprint_*`, есть только crc32c, в stdlib его нет).
  ETag у таких объектов совпадает с md5 в hex.
* **Zenodo:** record id версии (10676866 = v783) неизменяем. Контрольная сумма из `https://zenodo.org/api/records/10676866` (`files[].checksum = "md5:…"`), плюс собственный sha256.
* **GitHub:** брать raw-файл по тегу `v3.1.0` или по SHA коммита, проверять git blob sha1 (`git hash-object`) и sha256.
* Докачка через `Range` с продолжением, запись во временный файл и атомарный `rename`. Лишний раз не качать, если sha256 совпадает.
* Для чтения нужен **pyarrow**: Feather v2 со сжатием LZ4_FRAME stdlib не прочитает. Версию pyarrow закрепить (здесь проверено на pyarrow 25.0.1 и pandas 3.0.6).

sha256 файлов, скачанных в этой сессии (эталон для манифеста):
* `body-annotations-male-cns-v1.0-minconf-0.5.feather`: `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`
* `body-neurotransmitters-male-cns-v1.0.feather`: `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621`
* `Supplemental_file1_neuron_annotations.tsv@v3.1.0`: `9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be`
* sha256 для traced-only весов **не посчитан**: файл целиком не скачивался. Скрипт должен вычислить его при первой загрузке после проверки md5 `6601d4ad0afa99fd03eb087965ef2423` и записать в манифест.

### 6.3 Подводные камни

1. **ID.** В MaleCNS `bodyId` имеет тип int64 (≤ 1,57·10⁹, в float64 и JSON помещается), но в аннотациях `group`, `mancBodyid`, `mancGroup` хранятся как **double**, их нужно привести к Int64.
   У FlyWire и BANC `root_id` 18-значные (720575940…) и **больше 2⁵³**: через float и JSON-числа они портятся, нужно хранить int64 или строку (lee-lab уже хранит строками).
   ID из разных версий (v0.9 и v1.0, 630 и 783) не смешивать: после вычитки bodyId меняются.
2. **Отбор узлов.** Traced определяется как `status == "Traced"`, и это ровно множество из traced-only файла.
   DN выбираем через `superclass`, а не регуляркой по `type`: pIP1, MDN, aSP и другие DN не начинаются с «DN».
   Нужно решить, включать ли `sensory_descending` и `efferent_descending`. У 516 Traced-тел нет `superclass`.
3. **Лево и право.** В MaleCNS `somaSide` принимает значения L/R/M (у 60 851 тела NaN, это в основном не нейроны); в `instance` суффикс `_L/_R/_M`; `rootSide` задан для сенсорных нейронов.
   В FlyWire `side` = left/right/center (у сенсорных и восходящих это сторона входа нерва). ROI в MaleCNS записываются как `LAL(R)`, в FlyWire как `LAL_R`.
   **Сторона сомы у DN не обязательно совпадает со стороной аксона и мишеней в VNC** (бывают контралатеральные DN). Для отображения DN на управление
   влево и вправо нужна проверка по связности или по ROI в VNC (не проверено для конкретных типов).
   Шаг проектирования: делить параметры по типу (общие для L и R) и использовать `group` для сопоставления гомологов.
4. **Мозг и VNC.** Веса считаются по нейрону целиком, по всем ROI. Рёбра DN→VNC и AN (сома в VNC)→мозг присутствуют, поэтому нужно явно фильтровать по множеству узлов.
   Если VNC не моделируем, AN становятся входными узлами, а DN выходными. Фоторецепторы (ol_sensory) трассированы неполностью (4 114 из 6 098); вход разумнее подавать на VPN или на колончатые нейроны OL.
5. **NT и знак.** В MaleCNS 7 классов и `unclear`, в FlyWire 6 (без гистамина, серотонин завышен).
   Предлагаю брать `consensus_nt` (на тип) и откатываться к `celltype_predicted_nt` или `predicted_nt`.
   `unclear` есть у 3 100 Traced, у 502 нет строки NT. Для них и для случаев `predicted_nt_confidence < 0.5` (VPN 124, DN 16, AN 26) знак делать обучаемым или нулевым.
   ACh означает «+», GABA «−». **Glu в ЦНС мухи обычно тормозной (GluCl), но не всегда.** His «−». DA, 5-HT, OA модуляторные.
   Это решение модели, а не данные.
6. **Качество весов.** Флаг `minconf 0.5` задаёт порог уверенности синапса. В neuPrint есть ещё `weightHP` и `weightHR`, в плоских файлах их нет.
   Слабые рёбра (1–2 синапса) часто ложные, порог ≥ 5 стандартен. Автапсы (`body_pre == body_post`) выбросить (сколько их, не проверено).
7. **Изменчивость источников.** `body-neurotransmitters` и `tbar-neurotransmitters` в бакете MaleCNS перезалиты 2026-06-08, позже остальных (2026-06-03).
   Бакет lee-lab (BANC) перезаписывается регулярно. Поэтому фиксируем generation и хэши.
8. **neuPrint:** после миграции авторизации 2026-08 старые токены не работают; `male-cns:v1.0` без токена недоступен.
   В пайплайн neuPrint не включать, использовать только для разовых запросов.
9. **Лицензии:** MaleCNS, MANC и BANC (Dataverse) распространяются под CC-BY 4.0; достаточно атрибуции (Berg et al. 2026 Cell и т. д.) в NOTICE.
   FlyWire FAFB под CC BY-NC 4.0 по guidelines (Zenodo указывает CC-BY): не распространять и не встраивать в релизы, использовать только для локальной сверки.
10. **Память.** Traced-only в pandas занимает около 25,6 млн × 5 колонок, 1,5–2,5 ГБ RAM (оценка). Лучше читать через `pyarrow.ipc.open_file` по батчам
    и сразу фильтровать по множеству bodyId. В полном файле весов (1,05 ГБ, 151,9 млн рёбер) почти всё составляют фрагменты; нам он не нужен.

---

## 7. Служебное

* venv: `~/aiddnet/data/research/venv` (pyarrow 25.0.1, pandas 3.0.6).
* Вспомогательный модуль: `~/aiddnet/data/research/tools/remote_arrow.py`. Читает схему, футер, отдельные батчи и число строк удалённых Arrow/Feather-файлов через HTTP Range
  (`RangeFile`, `inspect`, `footer_blocks`, `batch_lengths`, `total_rows`).
* Образцы: `~/aiddnet/data/connectome/samples/` (два файла MaleCNS, TSV FlyWire и сохранённые HTTP-заголовки).
* Не проверено: список файлов Codex и Dataverse (BANC); определение `consensus_nt` по методам статьи; доступ к `male-cns:v1.0` с новым токеном;
  лицензия бакета lee-lab; число автапсов; сторона аксонов DN.
