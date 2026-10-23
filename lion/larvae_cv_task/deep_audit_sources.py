"""Archive public research metadata/abstracts for this diagnostic report."""
import json
import re
import urllib.request
import sys
from html import unescape
from pathlib import Path

OUT=Path(__file__).parent/'reports/deep_audit_20261020/research'
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
OUT.mkdir(exist_ok=True)
urls={
 'bsf_yolo':'https://api.crossref.org/works/10.1163/23524588-bja10482',
 'optical_flow':'https://api.openalex.org/works/doi:10.1016/j.inpa.2025.05.001',
 'external_dataset':'https://zenodo.org/api/records/13359376',
 'sahi':'https://arxiv.org/abs/2202.06934',
 'pointrend':'https://arxiv.org/abs/1912.08193',
 'hrnet':'https://arxiv.org/abs/1908.07919',
 'dinov3':'https://arxiv.org/abs/2508.10104',
 'dinov3_full':'https://arxiv.org/html/2508.10104v1',
}
for name,url in urls.items():
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'LarvaeResearchAudit/1.0'})
        raw=urllib.request.urlopen(req,timeout=20).read().decode('utf-8')
        if raw.lstrip().startswith('{'):
            data=json.loads(raw)
            (OUT/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            if name=='optical_flow':
                index=data.get('abstract_inverted_index') or {}
                positions={pos:word for word,places in index.items() for pos in places}
                abstract=' '.join(positions[k] for k in sorted(positions))
                print(abstract,flush=True)
                print(json.dumps(data.get('locations'),indent=2),flush=True)
                (OUT/f'{name}_abstract.txt').write_text(abstract,encoding='utf-8')
        else:
            plain=re.sub(r'<(script|style)\b[^>]*>.*?</\1>','',raw,flags=re.S|re.I)
            plain=unescape(re.sub('<[^>]+>',' ',plain))
            plain=re.sub(r'\s+',' ',plain)
            (OUT/f'{name}.txt').write_text(plain,encoding='utf-8')
            if name=='dinov3_full':
                for pattern in ['patch size','16 × 16','16x16','high-resolution','segmentation']:
                    for m in list(re.finditer(re.escape(pattern),plain,re.I))[:3]:
                        print(plain[max(0,m.start()-250):m.end()+700],flush=True)
        print('OK',name,flush=True)
    except Exception as exc:
        print('FAILED',name,str(exc),flush=True)
