import sys, cv2, numpy as np
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection","landmark_2d_106"])
app.prepare(ctx_id=-1, det_size=(640,640))
def run(path):
    cap=cv2.VideoCapture(path); res=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        row=[np.nan,np.nan]
        for fc in app.get(f):
            if fc.det_score<0.5: continue
            lm=fc.landmark_2d_106; side=0 if (fc.bbox[0]+fc.bbox[2])/2<f.shape[1]/2 else 1
            # inner lip gap (upper inner 62/70ish vs lower inner 60/57) normalised by face width
            gap=np.linalg.norm(lm[60]-lm[62])+np.linalg.norm(lm[57]-lm[70])+np.linalg.norm(lm[54]-lm[66])
            row[side]=gap/(fc.bbox[2]-fc.bbox[0])
        res.append(row)
    return np.array(res)
o=run(sys.argv[1]); s=run(sys.argv[2]); b=run(sys.argv[3])
np.save('mouth.npy',np.stack([o,s,b]))
for k,name in enumerate(['left','right']):
    for lab,x in [('swap',s),('final',b)]:
        m=~np.isnan(o[:,k])&~np.isnan(x[:,k])
        print(name,lab,'corr=%.2f'%np.corrcoef(o[m,k],x[m,k])[0,1],'amp ratio=%.2f'%(x[m,k].std()/o[m,k].std()), 'orig mean %.3f now %.3f'%(o[m,k].mean(),x[m,k].mean()))
