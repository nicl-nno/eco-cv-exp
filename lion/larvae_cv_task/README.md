# Обучение ResNet34 U-Net: методика статьи + улучшения fold 0

Папка содержит независимый воспроизводимый код для обучения сегментации трёх классов:

- `0` — фон;
- `1` — active;
- `2` — mature;
- `255` — игнорируемая область (COCO category id `2`).

По умолчанию запускается **нулевой фолд** — тот же вариант, на котором был получен лучший подтверждённый результат авторесерча. Никакие веса, обученные на изображениях исследования, в инициализацию не подмешиваются: только ImageNet-pretrained ResNet34.

## Что перенесено из статьи

- исходные изображения и COCO-разметка используются без изменения масштаба;
- U-Net с энкодером ResNet34, предобученным на ImageNet;
- нативные патчи `512×512`, восемь патчей на исходное изображение за эпоху;
- `ShiftScaleRotate`, горизонтальное и вертикальное отражения, `ElasticTransform(alpha=40, sigma=6)`, яркость/контраст и ImageNet-нормализация;
- функция потерь `weighted cross-entropy + multiclass Dice`;
- веса классов `w_c ∝ (1/p_c)^0.5`, нормированные до суммы 3;
- AdamW (`lr=1e-4`, `weight_decay=1e-3`), ReduceLROnPlateau (`factor=0.5`, `patience=5`);
- максимум 60 эпох, early stopping после 10 эпох без улучшения;
- пять GroupKFold-разбиений по исходным изображениям;
- выбор лучшей эпохи по loss центральных validation-кропов;
- финальная оценка целых изображений неперекрывающимися тайлами `512×512`;
- Flip4 TTA: исходное изображение, HFlip, VFlip и поворот на 180°, с усреднением **вероятностей**;
- IoU, Dice, precision и recall отдельно для active и mature.

Аугментации реализованы локально через Pillow/SciPy с параметрами статьи. Именно эта реализация использовалась в успешном прогоне fold 0, поэтому код не зависит от версии Albumentations/OpenCV.

## Подтверждённые улучшения fold 0

- 70% train-кропов привязываются к пикселю foreground, чтобы патчи реже содержали только фон;
- веса классов рассчитываются только по train-части текущего фолда — validation не влияет на обучение;
- gradient clipping с нормой `1.0`;
- AMP на CUDA;
- атомарные `last.pt` и `best.pt`;
- полноценное возобновление из `last.pt`, включая optimizer, scheduler, scaler и состояния генераторов случайных чисел;
- текущие central-crop IoU для active/mature и mIoU сохраняются после каждой эпохи в `progress.json`;
- перед финальной оценкой явно проверяется отсутствие пересечения train/validation ID.

Не включены непроверенные или ухудшившие результат варианты: ручное завышение веса mature, class-specific crop bias, focal/boundary loss, морфология в расчёте метрик и инициализация весами, уже видевшими этот датасет.

## Запуск

Из этой папки:

```powershell
python -m pip install -r requirements.txt
python train.py --fold 0
```

Все пять фолдов последовательно:

```powershell
python train.py --all-folds
```

Повторная команда автоматически продолжит незавершённый фолд из `last.pt`. Для обучения заново следует указать новый `output_dir` в `config.yaml`; так существующий результат не будет перезаписан.

Пути и гиперпараметры находятся в `config.yaml`. Сейчас `data_root` указывает на существующий набор из 69 изображений: `../task/coco_export`.

## Результат контрольного fold 0

Референсный прогон, из которого взяты улучшения:

| Показатель | Значение |
|---|---:|
| Лучшая эпоха | 47 |
| Central validation loss | 0.68650 |
| Active IoU, full image + Flip4 | 0.56144 |
| Mature IoU, full image + Flip4 | 0.62693 |
| Foreground mIoU | 0.59419 |

Новый запуск стохастический и может немного отличаться, но разделение данных, архитектура, loss, preprocessing и процедура оценки зафиксированы.

## Артефакты

Для каждого фолда создаётся отдельная папка:

```text
runs/article_plus_fold0/fold_0/
  best.pt       # лучшая эпоха по central validation loss
  last.pt       # последняя полностью завершённая эпоха, пригодна для resume
  progress.json # текущие loss, LR и validation IoU
  result.json   # full-image метрики лучшего checkpoint с Flip4 TTA
  candidate_epoch_*.pt # необязательные снимки для выбора по full-image IoU
  selection.json # сравнение исходного и выбранного по full-image IoU checkpoint
  selected_full_miou.pt # выбранные веса без состояния оптимизатора
```

`best.pt` и `last.pt` дополнительно содержат списки `train_ids` и `validation_ids`, fold-local веса классов и полную конфигурацию запуска.

При `candidate_every: 3` каждая третья эпоха и каждое улучшение central validation
loss сохраняются отдельно без состояния оптимизатора. Это позволяет сравнить
эпохи на целых validation-изображениях,
не меняя исходное правило выбора `best.pt` по loss центрального кропа.

После обучения выбрать checkpoint по целевой метрике можно командой
`python select_checkpoint.py --fold-dir runs/article_plus_fold0/fold_0`.
Это использует те же validation-изображения, поэтому полученный IoU —
валидационный показатель выбора модели, а не независимый тест.
Для сравнения исходных неперекрывающихся тайлов с перекрытием используйте
`python evaluate_tiling.py --checkpoint runs/article_plus_fold0/fold_0/selected_full_miou.pt --stride 448`.

Чтобы продолжить fold 0 до 60 эпох без изменения исходных результатов,
из этой папки выполните:

```powershell
New-Item -ItemType Directory -Force runs/iou_extended/fold_0 | Out-Null
Copy-Item runs/article_plus_fold0/fold_0/last.pt runs/iou_extended/fold_0/last.pt
Copy-Item runs/article_plus_fold0/fold_0/best.pt runs/iou_extended/fold_0/best.pt
python train.py --config config_iou.yaml --fold 0
python select_checkpoint.py --config config_iou.yaml --fold-dir runs/iou_extended/fold_0
python evaluate_tiling.py --config config_iou.yaml --checkpoint runs/iou_extended/fold_0/selected_full_miou.pt --stride 448 --tta flip4 --output runs/iou_extended/fold_0/tiling_comparison.json
```

Перед копированием используйте новую пустую папку `iou_extended`, иначе
существующие checkpoint могут быть перезаписаны. Для восьмикратной TTA
передайте `--tta dihedral8`; она медленнее, а прибавку IoU следует проверять
отдельно на других фолдах. Снимки и параметры тайлинга здесь подбирались по
validation fold 0, поэтому для оценки переноса улучшения нужна проверка
на остальных фолдах или независимом тестовом наборе.

## Локальная проверка оптимизации (fold 0)

На RTX 5080, с одним и тем же разделением по исходным изображениям:

| Вариант | Active IoU | Mature IoU | mIoU |
|---|---:|---:|---:|
| Исходный `best.pt`, эпоха 28, Flip4 | 0.51416 | 0.59411 | 0.55414 |
| Выбор по full-image mIoU, эпоха 50, Flip4 | 0.56425 | 0.64764 | 0.60595 |
| Эпоха 50, перекрытие 448, Flip4 | 0.57164 | 0.65480 | 0.61322 |
| Эпоха 50, перекрытие 448, Dihedral8 | 0.57190 | 0.65713 | 0.61452 |

Для практического применения рекомендованы выбранные веса и перекрытие 448
с Flip4: почти весь прирост достигается без дополнительной восьмикратной TTA.
Это подбор на validation fold 0, а не оценка на независимом тесте.

## Ночная проверка архитектур

`config_night.yaml` задаёт одинаковое обучение на 60 эпох для исходного U-Net,
U-Net с SE-вниманием, U-Net с ASPP-контекстом и ResNet34-FPN. Случайные
генераторы сбрасываются после создания модели, чтобы разница в числе новых
параметров не меняла порядок train-кропов и аугментаций между архитектурами.

```powershell
python smoke_models.py
python overnight.py --until 08:00
```

`overnight.py` сохраняет состояние в `runs/night/overnight_status.json`,
подробные логи — в `runs/night/logs/`, а по каждому фолду — `result.json`,
`selection.json`, `selected_full_miou.pt` и `tiling_comparison.json`.

## Дополнительные данные и copy-paste

Набор [Zenodo 10.5281/zenodo.13359376](https://doi.org/10.5281/zenodo.13359376)
(`BSF_Larvae_v1.zip`, CC BY 4.0) проверен по опубликованному MD5. В архиве
321 пара JPG/YOLO-полигон; по именам это 185 исходных сцен с повторными
аугментациями. Разметка содержит два числовых класса (0/1), но сам архив не
объясняет соответствие наших стадий `active`/`mature`. Фото сняты на белом
фоне, сильно отличающемся от субстрата в целевом наборе. Поэтому внешний
архив сохранён в `external/`, но **не включён в обучение** и не затрагивает
ни один validation fold. Возможное применение после выяснения классов —
отдельное предварительное обучение бинарной сегментации личинок.

Для эксперимента с синтетикой `config_synthetic.yaml` включает вставку 1–2
размеченных личинок в случайные train-кропы с вероятностью 0.35. Доноры
берутся исключительно из train ID текущего фолда; объектам обоих классов
даются равные шансы. Вставка не перекрывает размеченные объекты и ignore,
маска переносится вместе с полигоном; исходные аугментации применяются после
вставки. Обычные прогоны не изменены (вероятность по умолчанию 0).

```powershell
python smoke_synthetic.py
python train.py --config config_synthetic.yaml --fold 0
python select_checkpoint.py --config config_synthetic.yaml --fold-dir runs/copy_paste/fold_0
python evaluate_tiling.py --config config_synthetic.yaml --checkpoint runs/copy_paste/fold_0/selected_full_miou.pt --stride 448 --tta flip4
```

Синтетический прогон использует отдельный каталог и должен сравниваться с
ночным `resnet34_unet_se/fold_0` на тех же validation ID; результаты фолда 0
будут лишь предварительной проверкой, не независимым тестом. Фоновый
`finish_synthetic.py` после завершения обучения автоматически выбирает
checkpoint и оценивает перекрывающиеся тайлы; состояние пишется в
`runs/copy_paste/fold_0/followup_status.json`.

## Внешнее предобучение и новый backbone

`pretrain_external.py` читает Zenodo `BSF_Larvae_v1.zip` (CC BY 4.0),
оставляет по одному изображению на исходный stem и обучает бинарной маске
личинка/фон 12 фиксированных эпох без целевой валидации. Классы источника
0/1 не сопоставляются со стадиями active/mature. Затем `config_external.yaml`
загружает **только энкодер** ResNet34 из
`runs/external_pretrain/encoder.pt` в новую U-Net. `config_convnext.yaml`
использует ImageNet-pretrained ConvNeXt Tiny с U-Net декодером.
Оба варианта применяют `config_scene.yaml`-совместимые группы и целевую
рецептуру и сравниваются на сценовых фолдах 0 и 3 с отдельными результатами:

```powershell
python pretrain_external.py
python train.py --config config_external.yaml --output-dir runs/data_backbone/external_resnet34 --fold 0
python train.py --config config_convnext.yaml --output-dir runs/data_backbone/imagenet_convnext --fold 0
```

`run_transfer_backbone.py` последовательно выполняет предобучение, оба
целевых фолда для каждого варианта, выбор checkpoint по full-image IoU и
тайлинг Flip4/448. Возобновляемый статус: `runs/data_backbone/status.json`.
Сначала сравниваются фолды 0 и 1; лучший из новых вариантов затем проверяется
на фолдах 2–4 рядом с исходной архитектурой, пока не наступит указанное время.
Прогон можно повторить после остановки: завершённые фолды не обучаются заново.
