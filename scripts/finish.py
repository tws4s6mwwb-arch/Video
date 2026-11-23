"""Realism finishing pass.
1. Temporal de-flicker: inside swapped regions, the output may only change frame-to-frame
   as much as the ORIGINAL video changed there (removes per-frame swap shimmer, no ghosting).
2. Sharpness match: soften the left man's swapped face/beard to the video's own detail level.
3. Skin tone: darken the left man's skin (face, ears, neck, arms, hands) by GAIN.
"""
import sys, os, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

HERE = os.path.dirname(os.path.abspath(__file__))
vin, vorig, vout = sys.argv[1:4]
GAIN = float(sys.argv[4]) if len(sys.argv) > 4 else 0.82
SOFT = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
debug_frames = [int(x) for x in sys.argv[6].split(",")] if len(sys.argv) > 6 else None
Y0, Y1 = 330, 690

seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=f"{HERE}/models/selfie_multiclass.tflite"),
    output_category_mask=True))
cap, capo = cv2.VideoCapture(vin), cv2.VideoCapture(vorig)
fps = cap.get(cv2.CAP_PROP_FPS); W, H = int(cap.get(3)), int(cap.get(4))
writer = None if debug_frames else cv2.VideoWriter(vout, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
xs = np.arange(W)[None, :]

def left_skin(roi):
    cm = np.squeeze(seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).copy())).category_mask.numpy_view())
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clothes = cm == 4
    skin = ((cm == 2) | (cm == 3)).astype(np.uint8)
    out = np.zeros(g.shape, np.float32)
    ncomp, lab, st, cen = cv2.connectedComponentsWithStats(skin, connectivity=8)
    for c in range(1, ncomp):
        if st[c, cv2.CC_STAT_AREA] < 40: continue
        comp = lab == c
        is_face = (cm[comp] == 3).mean() > 0.5
        ring = (cv2.dilate(comp.astype(np.uint8), k9) > 0) & ~comp & clothes
        if not is_face and ring.sum() > 15:
            mine = (g[ring] > 80).mean() > 0.5   # comes out of the light-grey shirt = left man
        else:
            mine = cen[c][0] < W * 0.5
        if mine: out[comp] = 1
    return out

prev_o = prev_out = prev_skin = None
n = 0
while True:
    ok, f = cap.read(); ok2, o = capo.read()
    if not ok or not ok2 or (debug_frames and n > max(debug_frames)): break
    ff, of = f.astype(np.float32), o.astype(np.float32)
    # regions the swap changed (faces + beard)
    swapped = cv2.GaussianBlur((np.abs(ff - of).max(2) > 6).astype(np.float32), (0, 0), 3)
    swapped = np.clip(swapped * 2, 0, 1)
    out = ff.copy()
    # 1. de-flicker
    if prev_out is not None:
        excess = (ff - prev_out) - (of - prev_o)               # change the original did NOT have
        still = np.clip((10 - np.abs(of - prev_o).max(2)) / 10, 0, 1)  # don't fight real motion
        out = ff - (0.4 * swapped * still)[..., None] * excess
    # 2. sharpness match on the left man's swapped region
    soft = cv2.GaussianBlur(out, (0, 0), 0.8)
    wl = (swapped * (xs < W * 0.5) * SOFT)[..., None]
    out = out * (1 - wl) + soft * wl
    # 3. skin tone (left man)
    roi = out[Y0:Y1].clip(0, 255).astype(np.uint8)
    sk = cv2.GaussianBlur(left_skin(roi), (0, 0), 2.0)
    if prev_skin is not None: sk = 0.6 * sk + 0.4 * prev_skin
    prev_skin = sk
    g = roi.mean(2)
    protect = np.clip((g - 30) / 30, 0, 1) * np.clip((235 - g) / 20, 0, 1)   # beard/pupils/highlights
    protect *= cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 1] < 40               # TikTok logo
    w = (sk * protect)[..., None]
    out[Y0:Y1] = out[Y0:Y1] * (1 - w * (1 - GAIN))
    prev_o, prev_out = of, out.copy()
    res = out.clip(0, 255).astype(np.uint8)
    if debug_frames:
        if n in debug_frames:
            cv2.imwrite(f"fin_{n}.png", np.hstack([o[Y0:Y1], f[Y0:Y1], res[Y0:Y1]]))
    else:
        writer.write(res)
    n += 1
    if n % 100 == 0: print(n, flush=True)
if writer: writer.release()
print("done", n)
