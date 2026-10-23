"""Build the local report index from personally recorded per-image notes."""
import json
from html import escape
from pathlib import Path
from PIL import Image

out=Path(__file__).parent/'reports/deep_audit_20261020'
rows=json.loads((out/'per_image.json').read_text(encoding='utf-8'))
notes=json.loads((out/'visual_notes.json').read_text(encoding='utf-8'))
assert len(rows)==len(notes)==69
md=['# Покадровый визуальный аудит: все 69 изображений','',
    'Зелёный = active, пурпурный = mature, золотой = ignore. Карточки: оригинал / GT / OOF.',
    'Числа A/M — аннотации, не гарантированно отдельные особи. Все метрики диагностические, CV не независимый; A/G связаны визуально.','',
    '| Файл / ID | Сцена / fold | A / M | Binary IoU | Наблюдение |','|---|---|---:|---:|---|']
sections=[]
for r in rows:
    number=int(Path(r['file']).stem);note=notes[str(number)]
    name=f"cards/{number:02d}_{r['id']}.jpg"
    score=f"{r['binary_iou']:.4f}" if 'binary_iou' in r else 'исключён'
    md.append(f"| [{r['file']}]({name}) / {r['id']} | {r['scene']} / {r['fold']} | {r['annotations_active']} / {r['annotations_mature']} | {score} | {note} |")
    sections.append(f'<section id="image-{number}"><h2>{r["file"]} · ID {r["id"]} · {r["scene"]} / fold {r["fold"]}</h2><p>{escape(note)}</p><a href="{name}"><img loading="lazy" src="{name}" alt="{r["file"]}: original, GT and OOF prediction"></a></section>')
(out/'PER_IMAGE.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
# Standalone local evidence index, not a website deployment.
html='<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Аудит 69 изображений</title><style>body{font:16px/1.5 system-ui;margin:24px auto;max-width:1400px;padding:0 12px;color:#222;background:#fff}img{width:100%;height:auto}section{margin:40px 0;border-top:1px solid #bbb}a{color:#185ba5}</style><h1>Аудит 69 изображений</h1><p>Оригинал / разметка / out-of-fold prediction. Зелёный: active; пурпурный: mature; золотой: ignore. Это внутренний CV с выявленной связью A/G, не независимый тест.</p><p><a href="REPORT.md">Полный отчёт</a> · <a href="PER_IMAGE.md">Покадровая таблица</a></p>'+''.join(sections)+'</html>'
(out/'gallery.html').write_text(html,encoding='utf-8')
source_dir=Path(__file__).parent.parent/'task/coco_export/images'
Image.open(source_dir/'1.jpg').crop((500,1100,1012,1612)).resize((300,300)).save(out/'crops/evidence_1.jpg',quality=92)
Image.open(source_dir/'80.jpg').crop((484,1512,1000,2028)).resize((300,300)).save(out/'crops/evidence_80.jpg',quality=92)
for number in [19,45]:
    trip=Image.open(out/f'crops/{number:02d}_native.jpg')
    for col,kind in enumerate(['source','gt','pred']):
        trip.crop((col*512,38,(col+1)*512,550)).resize((300,300)).save(out/f'crops/evidence_{number}_{kind}.jpg',quality=92)
for r in rows:
    assert (out/f"cards/{int(Path(r['file']).stem):02d}_{r['id']}.jpg").is_file()
assert sum('binary_iou' in r for r in rows)==67
print('Verified: 69 notes, 69 cards, 67 evaluated images; report index and evidence crops generated.')
