"""Create a field-readable PDF summary from the visual COCO review."""

from io import BytesIO
from pathlib import Path
import re

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table,
    TableStyle, PageBreak, Image, KeepTogether,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'REVIEW.md'
OUT = ROOT / 'REVIEW_for_specialists.pdf'
CROPS = ROOT.parent / 'deep_audit_20261020' / 'crops'

pdfmetrics.registerFont(TTFont('Arial-ru', r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-ru-bold', r'C:\Windows\Fonts\arialbd.ttf'))
pdfmetrics.registerFontFamily('Arial-ru', normal='Arial-ru', bold='Arial-ru-bold')

W, H = landscape(A4)
NAVY = colors.HexColor('#24364a')
GRAY = colors.HexColor('#edf2f5')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleRu', fontName='Arial-ru-bold', fontSize=19, leading=23, textColor=NAVY, spaceAfter=13))
styles.add(ParagraphStyle(name='HeadRu', fontName='Arial-ru-bold', fontSize=12.5, leading=16, textColor=NAVY, spaceBefore=11, spaceAfter=6))
styles.add(ParagraphStyle(name='BodyRu', fontName='Arial-ru', fontSize=9.4, leading=13.6, spaceAfter=7))
styles.add(ParagraphStyle(name='SmallRu', fontName='Arial-ru', fontSize=8.1, leading=11, spaceAfter=4))
styles.add(ParagraphStyle(name='TableRu', fontName='Arial-ru', fontSize=7.2, leading=10))
styles.add(ParagraphStyle(name='TableHeadRu', fontName='Arial-ru-bold', fontSize=7.5, leading=10, textColor=colors.white))


def P(text, style='BodyRu'):
    return Paragraph(text, styles[style])


def photo(name, width):
    # Source/GT only; OOF is a model output, not evidence of an annotation error.
    with PILImage.open(CROPS / name) as full:
        cut = full.crop((0, 0, round(full.width * 2 / 3), full.height))
        stream = BytesIO()
        cut.save(stream, format='PNG')
        stream.seek(0)
        return Image(stream, width=width, height=width * cut.height / cut.width)


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#b9c5ce'))
    canvas.line(36, 34, W - 36, 34)
    canvas.setFont('Arial-ru', 7.4)
    canvas.setFillColor(colors.HexColor('#546575'))
    canvas.drawString(36, 22, 'Визуальная проверка COCO-разметки по ТЗ  •  20.10.2026')
    canvas.drawRightString(W - 36, 22, f'Страница {document.page}')
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=(W, H), leftMargin=36, rightMargin=36,
                      topMargin=34, bottomMargin=45, title='Проверка разметки личинок',
                      author='Визуальный аудит')
doc.addPageTemplates(PageTemplate(id='Normal', frames=[Frame(36, 45, W-72, H-79, leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)], onPage=footer))
story = []

story += [P('Проверка разметки личинок: что исправить и что уточнить', 'TitleRu')]
story += [P('<b>Для кого:</b> специалистов, определяющих стадию личинки, и команды разметки. Проверено 69 фотографий: исходный снимок сопоставлен с цветным наложением разметки. Зеленый — active, розовый — mature, желтый — ignore. Это обзор всех кадров и крупный просмотр выбранных мест, <b>не попиксельная экспертиза каждого объекта</b>.')]
story += [P('Как читать результаты', 'HeadRu')]
story += [P('<b>Точно установлено</b> — один и тот же файл имеет несовместимые варианты разметки либо вовсе не размечен. <b>Нужна оценка специалиста</b> — по снимку обнаружена вероятная проблема, но без решения о стадии конкретной личинки менять класс нельзя. <b>Не доказано</b> — несовпадение с прогнозом модели: модель тоже ошибается.')]
story += [P('1. Что точно нужно решить в выдаче', 'HeadRu')]
story += [P('<b>7.jpg:</b> снимок в точности совпадает с 6.jpg. На 6.jpg отмечено 107 участков mature, на 7.jpg нет ни одной метки. Если 7.jpg — технический дубль, официально исключите его из перечня размеченных кадров; если он нужен в поставке, это полный пропуск разметки.')]
story += [P('<b>8.jpg и 9.jpg:</b> это также один и тот же файл, но версии маски разные: 80 и 96 участков mature соответственно; в 9.jpg дополнительно два ignore. Нужно выбрать и утвердить единую правильную версию. Разница в количестве меток сама по себе не означает, что все дополнительные 16 правильны.')]
story += [P('2. Где нужна проверка стадии', 'HeadRu')]
story += [P('<b>13.jpg и 14.jpg:</b> сцена почти та же, но на 13.jpg отмечены 27 active и 105 mature, на 14.jpg — 95 active и 1 mature. Это самый важный вопрос предметной проверки: сопоставить одни и те же тела на двух кадрах и решить, изменился ли видимый цвет или произошло несогласованное назначение классов.')]
story += [P('<b>Сцены E и G:</b> на кадрах 40–45, 48, 49, 52, 53 и 1, 2, 72, 73, 76, 80 встречается много серых и кремово-коричневых тел с разными метками. ТЗ допускает зрелых особей серого оттенка на иллюстрации. Поэтому <b>нельзя просто заменить все розовые метки на зеленые</b>; нужен набор эталонных примеров от специалиста и сверка одинаковых тел между кадрами.')]
story += [P('3. Что проверить в форме масок', 'HeadRu')]
story += [P('<b>19.jpg:</b> на увеличенном фрагменте видны светлые сегментированные тела без соответствующей маски; у некоторых размеченных тел маска выглядит слишком узкой. В ТЗ требуется покрывать всю видимую поверхность тела, не дорисовывая скрытую часть. Аналогичные плотные области просмотреть на 17–23.jpg.')]
story += [P('<b>24–27, 49, 52, 53.jpg:</b> общая маска для соприкасающихся личинок допустима. Проверять нужно не «слиты ли личинки», а захвачен ли между ними субстрат или корм. По обзорному изображению объявить все эти групповые маски ошибкой нельзя.')]
story += [P('<b>Формат:</b> в проверенном каталоге имеются JPG и COCO-полигоны, но нет индексированных PNG-масок, требуемых в ТЗ. Возможно, они переданы отдельно — это нужно проверить у исполнителя.')]

story += [PageBreak(), P('Пример для предметной сверки: 13.jpg и 14.jpg', 'TitleRu')]
story += [P('В каждой паре слева исходное изображение, справа то же место с разметкой. Розовым обозначен mature, зеленым active. Сравните соответствующие сегментированные тела, а не только общую долю классов. Для переходной окраски ТЗ предписывает один класс по более чем 50% видимой поверхности тела; при невозможности определить стадию — ignore.')]
pair = Table([[photo('13_native.jpg', 355), photo('14_native.jpg', 355)]], colWidths=[382, 382])
pair.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0)]))
story += [pair, Spacer(1, 5), P('Слева — участок 13.jpg (многие тела mature), справа — близкий участок 14.jpg (преимущественно active). Изображения показаны без третьей колонки с прогнозом модели.', 'SmallRu')]
story += [P('Вопрос эксперту: какие из видимых здесь серовато-кремовых тел действительно имеют зрелую окраску? Достаточно выбрать 10–20 соответствующих тел и зафиксировать решение как эталон для всей сцены.', 'BodyRu')]
story += [P('<b>Что записать при сверке:</b> номер кадра и место на нём; решение active/mature/ignore; видимая доля тёмной окраски; остаётся ли решение тем же на соседнем ракурсе. Если тело закрыто кормом, оценивать только его видимую часть.', 'BodyRu')]
story += [PageBreak(), P('Другой пример: неполные маски на 19.jpg', 'TitleRu')]
story += [photo('19_native.jpg', 650), Spacer(1, 3), P('Здесь полезно оценивать полноту покрытия видимых тел, а не считать каждую светлую точку личинкой: сомнительные фрагменты по ТЗ можно пометить ignore.', 'SmallRu')]
story += [Spacer(1, 12), P('<b>Решение о приемке:</b> пока открыты пустая копия 7.jpg и несовпадающие маски 8/9, говорить о готовой согласованной разметке нельзя. Долю ошибочных пикселей и порог «менее 5%» этот обзор не измеряет: для этого нужен исправленный контрольный набор масок на исходном разрешении.', 'BodyRu')]

story += [PageBreak(), P('Где искать замечания: все 69 кадров', 'TitleRu')]
story += [P('«Проверить» означает конкретный участок работы для повторной разметки; это <b>не список доказанных ошибок</b>. На кадрах без выделенного дефекта также нельзя подтвердить точность каждого пикселя только обзорным просмотром.')]

source = SOURCE.read_text(encoding='utf-8')
lines = source.splitlines()
start = next(i for i, x in enumerate(lines) if x.startswith('| Кадры |'))
rows = []
for line in lines[start + 2:]:
    if not line.startswith('| '):
        break
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    rows.append(cells)

short = [
    'Сверить оттенки и стадии с 72, 73, 76, 80.',
    'Мелкие тела среди корма: тело или волокно?',
    'Точный дубль: 7 без меток.',
    'Точный дубль с разными масками.',
    'Резкое различие active/mature внутри одной сцены.',
    'Сверить те же тела на двух близких ракурсах.',
    'Проверить отдельные mature-метки.',
    'Полнота мелких тел; на 19 видны пропуски.',
    'Полнота мелких тел и контуры у субстрата.',
    'Крупная групповая маска; 27 смазан.',
    'Скачок доли mature между соседними кадрами.',
    'Проверить несколько зрелых меток.',
    'Исключён ли субстрат между телами?',
    'Серые тела: нужен эталон стадии от эксперта.',
    'Сверить окраску; прогноз вне лотка на 45 — не GT.',
    'Плотная группа: проверить фон между телами.',
    'Мелкие тела, спорные участки и полнота.',
    'Проверить слияния и фон в групповых масках.',
    'Частично видимые тела; границы и ignore.',
    'Согласовать классы на схожих ракурсах.',
]
assert len(rows) == len(short) == 20
table_data = [[P('Кадры', 'TableHeadRu'), P('Итог', 'TableHeadRu'), P('Задание на повторную проверку', 'TableHeadRu')]]
statuses = {'К':'Подтвержден конфликт', 'Р':'Проверить', 'О':'Нет доказанного дефекта'}
for cells, task in zip(rows, short):
    frame = re.sub(r'\s*\([^)]*\)', '', cells[0])
    table_data.append([P(frame, 'TableRu'), P(statuses[cells[1]], 'TableRu'), P(task, 'TableRu')])

t = Table(table_data, colWidths=[174, 134, W-72-308], repeatRows=1, hAlign='LEFT')
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY), ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,GRAY]),
    ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
    ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
    ('LINEBELOW',(0,0),(-1,0),0.7,NAVY),
]))
story += [t]
doc.build(story)
print(OUT)
