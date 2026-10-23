"""Contact sheet of source images with the old fold number."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

from data import image_path, read_coco


here = Path(__file__).resolve().parent
root = here.parent / "task" / "coco_export"
images, _ = read_coco(root)
folds = {}
for fold in range(5):
    result = json.loads((here / "runs" / "night" / "resnet34_unet" /
                         f"fold_{fold}" / "result.json").read_text(encoding="utf-8"))
    folds.update({i: fold for i in result["validation_ids"]})
ordered = sorted(images, key=lambda i: int(Path(images[i]["file_name"]).stem))
columns, tile_w, tile_h = 7, 190, 245
sheet = Image.new("RGB", (columns * tile_w, ((len(ordered) + columns - 1) // columns) * tile_h), "white")
draw = ImageDraw.Draw(sheet)
colors = ["#c82626", "#2272c6", "#248447", "#a45cc4", "#d89b23"]
for position, image_id in enumerate(ordered):
    x, y = (position % columns) * tile_w, (position // columns) * tile_h
    with Image.open(image_path(root, images[image_id])) as im:
        im.thumbnail((tile_w - 10, tile_h - 30))
        sheet.paste(im, (x + (tile_w - im.width) // 2, y + 22))
    draw.rectangle((x + 2, y + 2, x + tile_w - 3, y + tile_h - 3), outline=colors[folds[image_id]], width=3)
    draw.text((x + 8, y + 5), f"{Path(images[image_id]['file_name']).stem}  fold {folds[image_id]}", fill="black")
destination = here / "runs" / "hypotheses" / "source_contact_sheet.jpg"
sheet.save(destination, quality=90)
print(destination)
