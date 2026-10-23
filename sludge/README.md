# Микроорганизмы активного ила: результаты экспериментов

Снимок локальных результатов на **23.10.2026**. Сохранены результаты всех имеющихся
этапов: исходные updated-сравнения, grouped CV, три ночи, частичные специалисты,
перенос MAA из auto_micro и inference-only пилот SAM/EfficientSAM.

## С чего начать

- [Частичные специалисты: итоговый отчёт](NIGHT_REPORT_2026_10_23_HYBRIDS.md),
  [полные численные таблицы](experiments_updated_2026_10_02/overnight_2026_10_23_hybrids/final_metrics.md),
  [машиночитаемая сводка](experiments_updated_2026_10_02/overnight_2026_10_23_hybrids/summary.json).
- [Каскад и полные специалисты](NIGHT_REPORT_2026_10_23.md).
- [Вторая ночь: калибровка и контроль cosine30](NIGHT_REPORT_2026_10_22.md).
- [Первая ночь](NIGHT_REPORT_2026_10_21.md), [grouped CV](NIGHT_GROUPED_CV_RESULTS.md),
  [диагностический разбор](NIGHT_DIAGNOSTIC_REVIEW.md).
- [auto_micro / MAA: код переноса и синтетическая проверка](research/maa_transfer_v1/README.md).
- [SAM ViT-B / EfficientSAM: фактический FIT-only пилот](research/sam_pilot_v1/REPORT.md),
  [результаты24 проходов](research/sam_pilot_v1/results.json).
- [Исходная документация проекта](README_ORIGINAL.md).

## Главное о результатах

Контроль cosine30: PRIMARY pooled OOF macro-F1 mean **0.311569** по двум seeds.
У частичных специалистов максимальный mean macro-F1 **0.332936** у
`h_priority_rare`, но **ни один из12 кандидатов не прошёл STRICT eligibility**:
нужного улучшения count MAE attached и filamentous нет. Контроль не заменён.
Это не доказательство превосходства над релизом.

MAA подключён изолированно к FPN P4; выполнена синтетическая CPU-проверка,
но не обучение на реальных данных и не оценка F1.
SAM/EfficientSAM проверены на12 FIT-кадрах: автоматический recall25%/12.5%
при фиксированных PRIMARY настройках. GT-box диагностика гораздо лучше, но
не измеряет автоматическое обнаружение или точность масок.

Outer folds многократно просматривались: это **adaptive development, не
независимый test**. Student_6 — приоритетная разметка. В поздних этапах исходный
test не переоценивался; архив также сохраняет **более ранние исторические**
`test_usage.json`, test metrics/predictions. Их наличие не означает нового запуска
test в ночных этапах, MAA, SAM или при публикации. Train IDs релиза неизвестны.

## Структура

- `experiments_updated_2026_10_02/`: метрики, сводки, конфигурации, замороженные
  протоколы/регистрации, thresholds, split metadata и итоговые integrity audits.
- `outputs_updated_2026_10_02/`: численные predictions (Parquet), training logs,
  complete.json, тайминги и состояния завершённых очередей — **без checkpoints**.
- `research/`: MAA/SAM пилоты, численные результаты и закреплённые upstream sources.
- `src/`, `scripts/`, `tests/`, `configs/`, `pyproject.toml`, `requirements.lock`:
  снимок фактического кода/окружения; это не только исходный git commit.
- `model/`: паспорта моделей без весов.
- `publication/manifest.json`: SHA256 каждого опубликованного файла, происхождение
  скопированных файлов и явный список исключений внутри отобранных деревьев.

## Что намеренно не загружено

Исходные изображения, архивы датасетов, подготовленные изображения/разметка из
`data/`, визуальные overlays с исходными пикселями, модельные веса/checkpoints
(около48ГБ), cache, OS locks, виртуальная среда и вложенная git-история.
Доступны derived results: ID/группы/хеши, координаты предсказаний и отдельных GT,
метрики и пороги; этот архив **не является полностью обезличенным набором**.

Ссылки на локальные веса и их существующие hashes сохранены в исходных документах,
регистрациях и [weight_index.json](experiments_updated_2026_10_02/overnight_2026_10_23_hybrids/weight_index.json).
Для SAM/EfficientSAM upstream URL и SHA256 лежат в
[registration.json](research/sam_pilot_v1/registration.json).
Никакие внешние веса автоматически не скачиваются при просмотре архива.

## Целостность и воспроизведение

Скопированные документы/код/данные сохранены **побайтно**, без переписывания
исторических путей и без обновления frozen checksums. `.gitattributes` отключает
автоматическую нормализацию переводов строк. Поэтому часть ссылок ведёт на
намеренно отсутствующие локальные данные/веса/изображения. Старые абсолютные
пути внутри provenance — исторические, не инструкция запускать их на другой машине.

Проверка опубликованных файлов без GPU, данных и сторонних пакетов:

```console
python sludge/publication/export_snapshot.py --verify
```

Для нового запуска нужны отдельно исходные данные, веса и точное окружение;
upstream snapshots не содержат `.git` и не подменяют требуемые runner-ом чистые
git checkouts. Исторические `--run` **не следует перезапускать**; регистрации и
run_started sentinels сохранены. Новый научный эксперимент требует нового протокола.
При публикации обучение, inference, пересчёт старых метрик и deploy не выполнялись.

## Происхождение и права

Основной проект: https://github.com/PavelMarian/bio_collaboration,
база `48a8f6e01086af08efca78d3ebb17f4f760badb0` плюс фактические локальные изменения;
точные файлы определяет SHA256-манифест, а не одна ссылка на commit.
Эта публикация не назначает новую лицензию исходному проекту или данным.

Включены необходимые Python sources и оригинальные LICENSE закреплённых зависимостей:

- auto_micro — https://github.com/artyomnaz/auto_micro,
  `6646311c98f409b1a2185fba26c5e223af359153`, MIT;
- SAM — https://github.com/facebookresearch/segment-anything,
  `dca509fe793f601edb92606367a655c15ac00fdf`, Apache-2.0;
- EfficientSAM — https://github.com/yformer/EfficientSAM,
  `d525f622e6f640acf5a0fc37c7ca1f243da5bde0`, Apache-2.0.

История и результаты старого рабочего каталога сохранены; публикация собрана
в отдельном checkout `nicl-nno/eco-cv-exp`.
