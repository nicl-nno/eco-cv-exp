"""Read-only source/model audit; writes only derived reports and OOF predictions."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from data import read_coco, image_path, make_mask, IGNORE
from model import build_model
from evaluate_tiling import blended_probabilities

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'task' / 'coco_export'
OUT = HERE / 'reports' / 'deep_audit_20261020'
COLORS = np.array([[0,0,0],[35,220,90],[245,70,210]], dtype=np.uint8)
FONT = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 22)
SMALL = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)

def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

def metrics(cm):
    vals = {}
    for c, name in [(1,'active'),(2,'mature')]:
        tp = int(cm[c,c]); fp = int(cm[:,c].sum()-tp); fn = int(cm[c,:].sum()-tp)
        vals[name+'_iou'] = tp/max(1,tp+fp+fn)
        vals[name+'_recall'] = tp/max(1,int(cm[c,:].sum()))
    tp = int(cm[1:,1:].sum()); fp=int(cm[0,1:].sum()); fn=int(cm[1:,0].sum())
    vals['foreground_miou']=(vals['active_iou']+vals['mature_iou'])/2
    vals['binary_iou']=tp/max(1,tp+fp+fn)
    vals['binary_precision']=tp/max(1,tp+fp)
    vals['binary_recall']=tp/max(1,tp+fn)
    vals['stage_accuracy_on_detected_gt']=int(cm[1,1]+cm[2,2])/max(1,tp)
    vals['confusion']=cm.tolist()
    return vals

def overlay(im, mask):
    out=im.copy(); valid=(mask>0)&(mask<3)
    out[valid]=(0.42*im[valid]+0.58*COLORS[mask[valid]]).astype(np.uint8)
    out[mask==255]=(0.4*im[mask==255]+0.6*np.array([255,190,30])).astype(np.uint8)
    return Image.fromarray(out)

def fit(im, width=600, height=760):
    im=im.copy(); im.thumbnail((width,height))
    canvas=Image.new('RGB',(width,height),(245,245,245))
    canvas.paste(im,((width-im.width)//2,(height-im.height)//2)); return canvas

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in ('predictions','cards','boards','crops'): (OUT/p).mkdir(exist_ok=True)
    images, annotations=read_coco(SOURCE)
    config=yaml.safe_load((HERE/'config_convnext.yaml').read_text())
    ordered=sorted(images,key=lambda k:int(Path(images[k]['file_name']).stem))
    fold_of={}; checkpoints={}; baseline={}
    for fold in range(5):
        directory=HERE/'runs/data_backbone/imagenet_convnext'/f'fold_{fold}'
        res=json.loads((directory/'result.json').read_text())
        assert not set(res['train_ids'])&set(res['validation_ids'])
        for i in res['validation_ids']:
            assert i not in fold_of
            fold_of[i]=fold
        checkpoints[fold]=directory/'selected_full_miou.pt'
        baseline[fold]=json.loads((directory/'tiling_comparison.json').read_text())
    assert len(fold_of)==67
    for fold in range(5):
        todo=[i for i in ordered if fold_of.get(i)==fold and not (OUT/'predictions'/f'{i}.png').exists()]
        if not todo: continue
        ck=torch.load(checkpoints[fold],map_location='cpu',weights_only=False)
        assert ck['config']['architecture']=='convnext_tiny_unet'
        assert set(ck['validation_ids'])=={i for i,f in fold_of.items() if f==fold}
        model=build_model('convnext_tiny_unet',pretrained=False).cuda().eval()
        model.load_state_dict(ck['model']); del ck
        for i in todo:
            im=np.array(Image.open(image_path(SOURCE,images[i])).convert('RGB'))
            with torch.inference_mode():
                probs=blended_probabilities(model,im,torch.device('cuda'),448,'flip4')
            Image.fromarray(probs.argmax(0).numpy().astype(np.uint8)).save(OUT/'predictions'/f'{i}.png')
            print(json.dumps({'predicted':images[i]['file_name'],'id':i,'fold':fold}),flush=True)
        del model; torch.cuda.empty_cache()
    rows=[]; hashes=defaultdict(list); cms=defaultdict(lambda:np.zeros((3,3),dtype=np.int64))
    fold_cms=defaultdict(lambda:np.zeros((3,3),dtype=np.int64)); object_rows=[]
    for i in ordered:
        info=images[i]; filename=Path(info['file_name']).name; number=int(Path(filename).stem)
        scene=next(s['name'] for s in config['scene_groups'] if s['first']<=number<=s['last'])
        im=np.array(Image.open(image_path(SOURCE,info)).convert('RGB')); gt=make_mask(info,annotations[i])
        hashes[hashlib.sha256(im.tobytes()).hexdigest()].append(i)
        counts=Counter(a['category_id'] for a in annotations[i])
        row={'id':i,'file':filename,'scene':scene,'fold':fold_of.get(i),'width':im.shape[1],'height':im.shape[0],
             'annotations_active':counts[1],'annotations_mature':counts[3],'annotations_ignore':counts[2],
             'gt_active_pixels':int((gt==1).sum()),'gt_mature_pixels':int((gt==2).sum()),
             'gt_ignore_pixels':int((gt==255).sum()),'pixels':gt.size}
        predpath=OUT/'predictions'/f'{i}.png'
        pred=np.array(Image.open(predpath)) if predpath.exists() else None
        if pred is not None:
            valid=gt!=IGNORE
            cm=np.bincount(gt[valid].astype(np.int64)*3+pred[valid],minlength=9).reshape(3,3)
            row.update(metrics(cm)); cms[scene]+=cm; fold_cms[fold_of[i]]+=cm
            for a in annotations[i]:
                if a['category_id'] not in (1,3):continue
                x,y,w,h=a['bbox']; x0=max(0,int(x)); y0=max(0,int(y)); x1=min(im.shape[1],int(x+w)+1); y1=min(im.shape[0],int(y+h)+1)
                if x1<=x0 or y1<=y0:continue
                pm=Image.new('L',(x1-x0,y1-y0),0); d=ImageDraw.Draw(pm)
                for poly in a['segmentation']:
                    pts=[(poly[k]-x0,poly[k+1]-y0) for k in range(0,len(poly),2)]
                    if len(pts)>2:d.polygon(pts,fill=1)
                cls=1 if a['category_id']==1 else 2
                sel=(np.asarray(pm)>0)&(gt[y0:y1,x0:x1]==cls)
                if not sel.any(): continue
                pp=pred[y0:y1,x0:x1][sel]
                object_rows.append({'image_id':i,'file':filename,'annotation_id':a['id'],'class':cls,
                    'area':float(a['area']),'min_side':float(min(w,h)),'bbox':[x,y,w,h],
                    'binary_pixel_recall':float((pp>0).mean()),'correct_pixel_recall':float((pp==cls).mean())})
        rows.append(row)
        card=Image.new('RGB',(1800,840),'white'); d=ImageDraw.Draw(card)
        label=f"{filename} | task/image ID {i} | scene {scene} | fold {fold_of.get(i,'EXCLUDED')} | GT objects A={counts[1]} M={counts[3]} ignore={counts[2]}"
        d.text((10,3),label,font=FONT,fill='black')
        metric='' if pred is None else f"Binary IoU {row['binary_iou']:.3f} | A IoU {row['active_iou']:.3f} M IoU {row['mature_iou']:.3f} | stage acc. on detected GT {row['stage_accuracy_on_detected_gt']:.3f}"
        d.text((10,29),metric or 'Excluded duplicate: no CV prediction used',font=SMALL,fill='black')
        panels=[Image.fromarray(im),overlay(im,gt),overlay(im,pred) if pred is not None else Image.new('RGB',(600,760),'#ddd')]
        for col,(panel,title) in enumerate(zip(panels,['SOURCE','GT: active GREEN / mature MAGENTA / ignore GOLD','OOF PREDICTION'])):
            d.text((col*600+8,56),title,font=SMALL,fill='black');card.paste(fit(panel),(col*600,80))
        card.save(OUT/'cards'/f'{number:02d}_{i}.jpg',quality=93)
    for j in range(0,len(rows),2):
        board=Image.new('RGB',(1800,840*min(2,len(rows)-j)),'white')
        for k,row in enumerate(rows[j:j+2]):
            card=Image.open(OUT/'cards'/f"{int(Path(row['file']).stem):02d}_{row['id']}.jpg")
            board.paste(card,(0,k*840))
        board.save(OUT/'boards'/f'board_{j//2+1:02d}.jpg',quality=93)
    summary={'images':len(rows),'evaluated_images':len(fold_of),'annotation_counts':dict(Counter(a['category_id'] for aa in annotations.values() for a in aa)),
             'pixel_identical_groups':[[{'id':i,'file':Path(images[i]['file_name']).name} for i in ids] for ids in hashes.values() if len(ids)>1],
             'scenes':{},'folds':{},'aggregate':metrics(sum(cms.values()))}
    for scene,cm in cms.items():
        rr=[r for r in rows if r['scene']==scene and r['fold'] is not None]
        summary['scenes'][scene]={'n_images':len(rr),'folds':sorted({r['fold'] for r in rr}),
             'annotations_active':sum(r['annotations_active'] for r in rr),
             'annotations_mature':sum(r['annotations_mature'] for r in rr),**metrics(cm)}
    for fold,cm in fold_cms.items():
        mm=metrics(cm); assert abs(mm['foreground_miou']-baseline[fold]['overlap']['foreground_miou'])<1e-10
        summary['folds'][fold]=mm
    dump('per_image.json',rows);dump('per_object.json',object_rows);dump('summary.json',summary)
    with (OUT/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row if k!='confusion'}));writer.writeheader()
        writer.writerows({k:v for k,v in r.items() if k!='confusion'} for r in rows)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
