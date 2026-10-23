# auto_micro → Faster R-CNN: изолированный перенос MAA v1

Дата: 23.10.2026. Основание: новая просьба пользователя попробовать наработки
`https://github.com/artyomnaz/auto_micro`. Это **не возобновление ночных очередей**
и не изменение их замороженных протоколов.

## Что сделано и чего пока нет

Реальный модуль MAA подключён к текущему типу детектора
`fasterrcnn_resnet50_fpn_v2` через отдельный Python-wrapper. Код `src/sludge_micro`,
ночные scripts/configs/registration, старые веса, результаты и пороги не менялись.
Используются уже установленные зависимости; pip install и загрузки весов не было.

Выполнена CPU-проверка полного детектора с локальной COCO-инициализацией на
**синтетическом** RGB кадре 128×160 с двумя объектами и отдельно с пустым target.
Выполнен backward, но **ни одного optimizer step**; обученных MAA-весов нет.
Ни реальные FIT/calibration/outer кадры, ни исходный test не открывались.
F1, count MAE и переносимость на наши микроорганизмы пока **не оценены**.

Результат: [probe_results_20261023.json](probe_results_20261023.json).

Заключительная проверка в16:28 МСК: **11 тестов пройдены**, Black и Ruff
(E/F/I/B/UP) прошли; upstream checkout чистый. Дополнительно сверены **241
зарегистрированный input** завершённых протоколов22/23.10 и hybrids: hashes
совпадают, пакет неизменён.

- Точное совпадение boxes/scores/labels исходного детектора и wrapper при gate=0.
- Штатные torchvision RPN и RoI losses; изменение второго объекта меняет loss.
- Конечный backward, ненулевые градиенты всех шести lambda и QKV при gate=0.1.
- Пустые targets обрабатываются без NaN/Inf.
- Сохранение/загрузка state_dict адаптера воспроизводит результат точно.
- Параметры: **43 286 903 → 44 226 929**, добавлено **940 026** (~2.17%).
- Это не измерение производственной latency/VRAM и не доказательство улучшения качества.
- SHA256 установленного пакета до и после совпал:
  `fb698c88c123e16f6fc1174ff46e6b77924bea92abb654984d4e5502b64c3310`.

## Источник, лицензия и точность заявлений

Upstream клонирован в `../auto_micro_upstream`, без изменений, commit
`6646311c98f409b1a2185fba26c5e223af359153`.
MIT © 2026 artyomnaz, оригинальная лицензия сохранена в clone.
`upstream_manifest.json` содержит SHA256 89 Python-файлов и LICENSE.
Перед импортом проверяются полный список Python-файлов и их контрольные суммы.
При расхождении wrapper отказывает в запуске; автоматически менять pins нельзя.

Первичные источники:

- [MAA block](https://github.com/artyomnaz/auto_micro/blob/6646311c98f409b1a2185fba26c5e223af359153/maa/src/maa/attention/maa/block.py)
- [TokenAttentionAdapter](https://github.com/artyomnaz/auto_micro/blob/6646311c98f409b1a2185fba26c5e223af359153/maa/src/maa/models/adapters.py)
- [Опубликованные таблицы](https://github.com/artyomnaz/auto_micro/blob/6646311c98f409b1a2185fba26c5e223af359153/maa/README.md)
- [MIT license](https://github.com/artyomnaz/auto_micro/blob/6646311c98f409b1a2185fba26c5e223af359153/LICENSE)

Числа 0.975 weighted F1 / 0.928 mAP50 / 0.862 mIoU — **заявленные авторами
результаты на их данных**, здесь не воспроизведены. Они не сопоставимы напрямую
с нашим macro-F1 или count MAE. В просмотренном дереве и GitHub Releases
готовые benchmark checkpoints не найдены.

## Что обнаружено в опубликованном коде

1. `DetectionModelBase.compute_losses` выбирает `labels[:, 0]` и `boxes[:, 0]`,
   усредняет пространственные предсказания. Изменение второго объекта не меняет
   эти losses — подтверждено тестом. Этот detector training pipeline не переносится.
2. `InternImage-H` — локальный упрощённый host. В `DeformableConv2d.forward`
   offsets вычисляются, но не используются при свёртке; изменение offset weights
   не меняет выход — подтверждено тестом. Не называем его официальным InternImage-H/DCNv3.
3. `build_dinov3` отбрасывает аргумент pretrained (`del pretrained`) и создаёт
   локальный `PatchTransformerEncoder`. Это не загрузчик официальных DINOv3-весов.
4. `DensityPrior(use_mass=False)` по умолчанию считает KDE на координатах сетки,
   не на найденных клетках или image-dependent mass. Для одинаковой геометрии
   разные изображения дают одинаковый density bias — подтверждено тестом.
5. `build_token_grid` для несquare числа токенов строит линию; compactness/density
   не используют переданные tokens_h/tokens_w для координат. Здесь оставлена
   upstream square grid 16×16, без молчаливого исправления алгоритма.

Это ограничения конкретного опубликованного snapshot, **не доказательство
ошибочности всех научных результатов авторов**. MAA можно исследовать отдельно
от их detector hosts, сохраняя проверенную многoобъектную обработку torchvision.

## Локальная адаптация и её ограничения

`adapter.py` использует неизменённый upstream `TokenAttentionAdapter`:

1. Изображение поступает в штатный torchvision transform и ResNet-50/FPN.
2. Из normalized RGB восстанавливается диапазон [0,1] для image priors.
3. Только P4 (ключ FPN `"2"`, 256 каналов) передаётся в MAA: 16×16 токенов,
   dim256, 8 heads, один блок; RGB для priors resize512×512 как в upstream.
4. `P4_new = P4 + gate * (MAA_adapter(P4, RGB) - P4)`, trainable gate initial0.
   Остальные FPN уровни не меняются. RPN, RoIAlign, detector heads и losses штатные.
5. Все BN running statistics заморожены, включая BN новых prior networks.

Это **новая интеграция**, а не точная реплика авторского benchmark. Gate=0
сохраняет pretrained feature stream, но на первом optimizer step градиент
attention-весов нулевой: сначала должен сдвинуться gate. Не выдавать начальное
совпадение за улучшение. Synthetic backward проверен при явно включённом gate=0.1.

Для реальных кадров v1 принимает только один RGB кадр **960×1280**: размеры
кратны32, transform не меняет геометрию, padding отсутствует. Другие размеры и
batch>1 отвергаются, а не молча обрабатываются с неправильными priors.
В синтетическом smoke явно зарегистрирован отдельный размер128×160.
Производственный full-resolution memory/latency benchmark ещё не проводился.

Resize изображения в512×512 и pooling16×16 искажают физическое соотношение осей
для прямоугольного кадра. Приоры — приближения, а не измеренные биологические
границы/плотность/органеллы. Новые learned priors не имеют специализированной
разметки органелл. Эти ограничения сохранены и раскрыты, не исправлены post hoc.

`priors_enabled=False` отключает все шесть bias-поправок, оставляя токеновый
attention/MLP и тот же выделенный параметрический бюджет для будущего контроля
добавленной ёмкости. Неактивные prior-параметры в этом режиме не обучаются;
это не обещание одинакового активного compute. Ни один такой сравнительный fit
не запускался.

## Запуск проверок

Из корня `microorganism_detector`, PowerShell:

```powershell
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe -m pytest research/maa_transfer_v1/test_transfer.py
.venv\Scripts\python.exe -u research/maa_transfer_v1/smoke.py --out research/maa_transfer_v1/probe_results_NEW.json
```

`smoke.py` отказывает в перезаписи существующего результата. Он не использует
GPU, не меняет optimizer/weights, не читает manifests и не вызывает night runner.
При переносе окружения clone надо восстановить именно на указанный commit;
локальные hashes учитывают фактические байты файлов текущего checkout.

Подключать wrapper можно только **после** загрузки исходных COCO/base weights и
настройки trainable backbone layers. State dict теперь содержит дополнительные
модули и новые префиксы; стандартный release-loader его не поддерживает.
Для будущего checkpoint требуется отдельный формат с `export_metadata()` и
class/config/protocol pins; обычным `sludge_micro/1` его помечать нельзя.
Текущий тест проверяет roundtrip параметров, не создаёт production-loader.

## Следующий эксперимент — отдельная регистрация, ещё не запущен

Не включать MAA сразу во все специалисты: это смешает эффект внимания с
class-routing и кратно увеличит стоимость. Минимальный следующий вопрос:
**даёт ли один общий Faster R-CNN + MAA выигрыш относительно cosine30?**

До запуска отдельно согласовать бюджет/дедлайн и заморозить новую регистрацию:
контроль cosine30, такой же attention-bridge без priors и MAA; одинаковые
COCO initialization / schedule / seeds / grouped FIT-calibration-outer splits.
Пороги только calibration и frozen до outer; Student_6 primary; исходный test
не открывать. Сохранить count_guarded, диагностические политики и прежний строгий
критерий count MAE / class-F1, не выбирать рецепт по постфактум улучшившейся метрике.
Измерить full-resolution память/время до длинной очереди, использовать прежние
shared OS locks и один GPU worker. Ночные задачи остаются выключенными.

Старые outer folds по-прежнему adaptive development. Нужна свежая независимая
проверка для сильного вывода; данные Zenodo нельзя просто смешать с нашими:
кокки/бациллы не соответствуют семи классам активного ила. Возможный pretraining
на них — отдельная гипотеза, требующая domain/data-overlap и split audit.

Автоматизации физического микроскопа не выполнялось; управляющие команды
Arduino/двигателям не отправлялись. Данные Zenodo не скачивались. Деплоя,
субагентов, коммитов, PR, публикаций данных и платных сервисов не было.
