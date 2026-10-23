"""Exploratory multi-scale NCC of source RGB crops: no split/data edits."""
import json
import numpy as np
from PIL import Image,ImageDraw
from scipy.signal import fftconvolve
from deep_visual_audit import SOURCE,OUT,FONT,dump

source=Image.open(SOURCE/'images/1.jpg').convert('RGB')
results=[]
for target_number in [80,73,32]:
    target=Image.open(SOURCE/f'images/{target_number}.jpg').convert('RGB')
    small=target.resize((target.width//4,target.height//4),Image.Resampling.BILINEAR)
    a=np.asarray(small.convert('L'),dtype=float)
    canvas=Image.new('RGB',(920,330*3),'white');draw=ImageDraw.Draw(canvas)
    for row,(x,y) in enumerate([(400,450),(500,1100),(400,1750)]):
        crop=source.crop((x,y,x+512,y+512));best=None
        for scale in np.linspace(.65,1.45,41):
            size=round(128*scale)
            b=np.asarray(crop.resize((size,size),Image.Resampling.BILINEAR).convert('L'),dtype=float)
            b-=b.mean();kernel=np.ones(b.shape)
            total=fftconvolve(a,kernel,mode='valid')
            sq=fftconvolve(a*a,kernel,mode='valid')
            den=np.sqrt(np.maximum(sq-total*total/b.size,1e-8)*np.sum(b*b))
            corr=fftconvolve(a,b[::-1,::-1],mode='valid')/den
            yy,xx=np.unravel_index(corr.argmax(),corr.shape);score=float(corr[yy,xx])
            if best is None or score>best['ncc']:
                best={'target':target_number,'source_xy':[x,y],'target_xy':[int(xx*4),int(yy*4)],'target_size':size*4,'scale':float(scale),'ncc':score}
        results.append(best)
        tx,ty=best['target_xy'];sz=best['target_size']
        target_crop=target.crop((tx,ty,tx+sz,ty+sz))
        draw.text((5,row*330),f"1.jpg ({x},{y}) / {target_number}.jpg NCC={best['ncc']:.3f}",font=FONT,fill='black')
        canvas.paste(crop.resize((300,300)),(0,row*330+30))
        canvas.paste(target_crop.resize((300,300)),(310,row*330+30))
        # Difference aids review; correspondence is exploratory, not biological identity.
        diff=np.abs(np.asarray(crop.resize((300,300))).astype(float)-np.asarray(target_crop.resize((300,300))).astype(float))
        canvas.paste(Image.fromarray(np.clip(diff*2,0,255).astype(np.uint8)),(620,row*330+30))
    canvas.save(OUT/'crops'/f'scene_1_vs_{target_number}.jpg',quality=94)
dump('scene_match_ncc.json',results)
print(json.dumps(results,indent=2))
