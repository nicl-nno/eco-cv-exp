"""Derived diagnostics only: native crops, size summaries, optional fixed 2x probe."""
import argparse
import json
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from deep_visual_audit import OUT, HERE, SOURCE, FONT, overlay, metrics, dump
from data import read_coco, image_path, make_mask
from model import build_model
from evaluate_tiling import blended_probabilities


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scale-probe', action='store_true')
    args=parser.parse_args()
    rows=json.loads((OUT/'per_image.json').read_text(encoding='utf-8'))
    objects=json.loads((OUT/'per_object.json').read_text(encoding='utf-8'))
    images,annotations=read_coco(SOURCE)
    scene_of={r['id']:r['scene'] for r in rows}
    size_stats={}
    for scene in 'ABCDEFG':
        obs=[o for o in objects if scene_of[o['image_id']]==scene]
        size_stats[scene]={'objects_with_visible_mask':len(obs),
            'median_bbox_short_side':float(np.median([o['min_side'] for o in obs])),
            'median_polygon_area':float(np.median([o['area'] for o in obs])),
            'fraction_bbox_short_side_lt16':float(np.mean([o['min_side']<16 for o in obs])),
            'fraction_zero_foreground_overlap':float(np.mean([o['binary_pixel_recall']==0 for o in obs]))}
    dump('size_stats.json',size_stats)
    print(json.dumps(size_stats,indent=2),flush=True)
    # Native 512px crops, without source resampling. Coordinates are explicit.
    specs=[(9,200,500),(13,850,100),(14,850,100),(19,1300,550),
           (27,1450,650),(45,400,2048),(49,350,1300),(64,250,1200),(80,350,1500)]
    for number,x,y in specs:
        r=next(r for r in rows if r['file']==f'{number}.jpg'); i=r['id']
        im=np.asarray(Image.open(image_path(SOURCE,images[i])).convert('RGB'))
        gt=make_mask(images[i],annotations[i]); pred=np.asarray(Image.open(OUT/'predictions'/f'{i}.png'))
        x=min(x,im.shape[1]-512);y=min(y,im.shape[0]-512)
        src=im[y:y+512,x:x+512]; g=gt[y:y+512,x:x+512];p=pred[y:y+512,x:x+512]
        canvas=Image.new('RGB',(1536,550),'white');d=ImageDraw.Draw(canvas)
        for k,(panel,title) in enumerate(zip([Image.fromarray(src),overlay(src,g),overlay(src,p)],['SOURCE','GT','OOF'])):
            d.text((k*512+5,5),f'{number}.jpg ({x},{y}) {title}',font=FONT,fill='black')
            canvas.paste(panel,(k*512,38))
        canvas.save(OUT/'crops'/f'{number:02d}_native.jpg',quality=95)
    if not args.scale_probe:return
    # One predeclared scale, all fold-3 images (B and G), no fitting or selection.
    ck=torch.load(HERE/'runs/data_backbone/imagenet_convnext/fold_3/selected_full_miou.pt',map_location='cpu',weights_only=False)
    model=build_model('convnext_tiny_unet',pretrained=False).cuda().eval()
    model.load_state_dict(ck['model'])
    cms=defaultdict(lambda:np.zeros((3,3),np.int64)); results=[]
    for r in rows:
        if r['fold']!=3:continue
        i=r['id']; assert i in ck['validation_ids'] and i not in ck['train_ids']
        im=Image.open(image_path(SOURCE,images[i])).convert('RGB'); w,h=im.size
        start=time.perf_counter()
        probs=blended_probabilities(model,np.asarray(im.resize((2*w,2*h),Image.Resampling.BILINEAR)),torch.device('cuda'),448,'flip4')
        pred=F.interpolate(probs[None],size=(h,w),mode='bilinear',align_corners=False)[0].argmax(0).numpy()
        gt=make_mask(images[i],annotations[i]); valid=gt!=255
        cm=np.bincount(gt[valid].astype(np.int64)*3+pred[valid],minlength=9).reshape(3,3)
        cms[r['scene']]+=cm
        results.append({'file':r['file'],'id':i,'scene':r['scene'],'seconds':time.perf_counter()-start,
                        'baseline_binary_iou':r['binary_iou'],**metrics(cm)})
        print(json.dumps(results[-1]),flush=True)
    dump('scale_probe_2x.json',{'exploratory_only':True,'scale':2,'protocol':'Resize RGB 2x bilinear; existing fold3 checkpoint; tile512/448 Flip4; resize probabilities back bilinear; original GT unchanged',
         'per_image':results,'scenes':{s:metrics(cm) for s,cm in cms.items()}})


if __name__=='__main__':main()
