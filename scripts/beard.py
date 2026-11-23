import sys, cv2, numpy as np
from scipy.spatial import Delaunay
from insightface.app import FaceAnalysis

src_path, vin, vout, vorig = sys.argv[1:5]   # vin = face-swapped video, vorig = original video
debug_frames = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else None

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                   allowed_modules=["detection", "landmark_2d_106"])
app.prepare(ctx_id=-1, det_size=(640, 640))

# extra points below/around the jaw, in face-relative units (a = down, b = right, units of eye distance)
EXT = [(a, b) for a in (0.45, 0.9, 1.35) for b in (-1.1, -0.55, 0, 0.55, 1.1)] + \
      [(-2.2, -2.2), (-2.2, 2.2), (2.0, -2.2), (2.0, 2.2), (0, -1.6), (0, 1.6), (-1.0, -1.8), (-1.0, 1.8)]
BOTTOM = [(0.5, -0.85), (0.95, -0.62), (1.3, -0.3), (1.35, 0), (1.3, 0.3), (0.95, 0.62), (0.5, 0.85)]  # beard tip outline
LEFT_JAW = [13, 14, 15, 16, 2, 3, 4]
RIGHT_JAW = [18, 32, 31, 30, 29]
NOSE_BASE = [83, 85, 80, 79, 77]
OUTER_LIP = [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]
USE = list(range(106))

def frame_axes(lm):
    le, re = lm[33:43].mean(0), lm[87:97].mean(0)  # eye regions
    s = np.linalg.norm(re - le)
    chin = lm[0]
    down = chin - (le + re) / 2; down /= np.linalg.norm(down)
    right = np.array([-down[1], down[0]])
    if np.dot(right, re - le) < 0: right = -right
    return chin, down, right, s

def rel(lm, pts):
    chin, down, right, s = frame_axes(lm)
    return np.array([chin + down * a * s + right * b * s for a, b in pts], np.float32)

def control(lm):
    return np.vstack([lm[USE], rel(lm, EXT)]).astype(np.float32)

def mask_poly(lm):
    return np.vstack([lm[LEFT_JAW], rel(lm, BOTTOM), lm[RIGHT_JAW], lm[NOSE_BASE]]).astype(np.int32)

# ---- source beard ----
src = cv2.imread(src_path)
sf = max(app.get(src), key=lambda f: f.bbox[2] - f.bbox[0])
slm = sf.landmark_2d_106.astype(np.float32)
smask = np.zeros(src.shape[:2], np.float32)
cv2.fillPoly(smask, [mask_poly(slm)], 1)
lip = np.zeros_like(smask); cv2.fillPoly(lip, [slm[OUTER_LIP].astype(np.int32)], 1)
lip = cv2.dilate(lip, np.ones((9, 9), np.uint8))
smask *= 1 - lip
sgray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
# keep only beard-dark pixels (drops the white shirt below the beard tip)
_m = cv2.blur(sgray, (15, 15))
_sd = cv2.GaussianBlur(np.sqrt(np.maximum(cv2.blur(sgray * sgray, (15, 15)) - _m * _m, 0)), (0, 0), 4)
hair = ((_sd > 7) | (cv2.GaussianBlur(sgray, (0, 0), 4) < 70)).astype(np.uint8)
hair = cv2.morphologyEx(hair, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
hair = cv2.morphologyEx(hair, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
_hsv = cv2.cvtColor(cv2.GaussianBlur(src, (0, 0), 3), cv2.COLOR_BGR2HSV)
not_hair = (cv2.GaussianBlur(sgray, (0, 0), 3) > 150) | ((_hsv[..., 1] > 140) & (_hsv[..., 2] > 140))
hair &= (1 - cv2.dilate(not_hair.astype(np.uint8), np.ones((15, 15), np.uint8)))
hair &= (smask + lip > 0).astype(np.uint8)
_n, _lab, _st, _ = cv2.connectedComponentsWithStats(hair)
hair = (_lab == 1 + np.argmax(_st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
_ff = hair.copy(); cv2.floodFill(_ff, None, (0, 0), 1); hair |= (1 - _ff)  # fill holes
hair = cv2.erode(hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
smask *= cv2.GaussianBlur(hair.astype(np.float32), (0, 0), 5)
# reference skin brightness: cheeks above the beard line
cheek = np.zeros_like(smask)
cv2.fillPoly(cheek, [np.vstack([slm[[9, 10, 11]], slm[[76, 74]]]).astype(np.int32)], 1)
src_skin = sgray[cheek > 0].mean()
src_ctrl = control(slm)
tri = Delaunay(src_ctrl).simplices
if debug_frames is not None:
    dbg = src.copy(); cv2.polylines(dbg, [mask_poly(slm)], True, (0, 255, 0), 4)
    cv2.imwrite("dbg_srcmask.jpg", dbg)
    cv2.imwrite("dbg_srccut.jpg", (src * smask[..., None])[1100:2100, 150:900])

def warp(img, src_pts, dst_pts, shape):
    out = np.zeros(shape, np.float32)
    for t in tri:
        s, d = src_pts[t], dst_pts[t]
        r = cv2.boundingRect(d)
        x, y, w, h = r
        if w <= 0 or h <= 0 or x + w <= 0 or y + h <= 0 or x >= shape[1] or y >= shape[0]:
            continue
        M = cv2.getAffineTransform(s, (d - [x, y]).astype(np.float32))
        patch = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = np.zeros((h, w), np.float32)
        cv2.fillConvexPoly(m, np.int32(d - [x, y]), 1, cv2.LINE_AA)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, shape[1]), min(y + h, shape[0])
        p, mm = patch[y0 - y:y1 - y, x0 - x:x1 - x], m[y0 - y:y1 - y, x0 - x:x1 - x]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - mm) + p * mm
    return out

INNER = [54, 57, 60, 62, 65, 66, 69, 70]

# ---- pass 1: left-face landmarks from the ORIGINAL video (true mouth motion) ----
capo = cv2.VideoCapture(vorig)
W, H = int(capo.get(3)), int(capo.get(4)); fps = capo.get(cv2.CAP_PROP_FPS)
lms, widths = [], []
n = 0
while True:
    ok, fr = capo.read()
    if not ok or (debug_frames and n > max(debug_frames)): break
    fs = [f for f in app.get(fr) if f.det_score > 0.5 and (f.bbox[0] + f.bbox[2]) / 2 < W * 0.5]
    if fs:
        f = max(fs, key=lambda f: f.bbox[2] - f.bbox[0])
        lms.append(f.landmark_2d_106.astype(np.float32)); widths.append(f.bbox[2] - f.bbox[0])
    else:
        lms.append(None); widths.append(None)
    n += 1
print("landmarks collected", n, flush=True)

# zero-phase (centred) smoothing: no lag, breaks at detection gaps / big jumps
SIG = 1.2
smoothed = [None] * len(lms)
for t, lm in enumerate(lms):
    if lm is None: continue
    acc, wsum = np.zeros_like(lm), 0.0
    for d in range(-3, 4):
        u = t + d
        if 0 <= u < len(lms) and lms[u] is not None and np.abs(lms[u] - lm).mean() < 0.12 * widths[t]:
            w = np.exp(-d * d / (2 * SIG * SIG)); acc += w * lms[u]; wsum += w
    smoothed[t] = acc / wsum

# ---- pass 2 ----
cap = cv2.VideoCapture(vin); capo = cv2.VideoCapture(vorig)
writer = None if debug_frames else cv2.VideoWriter(vout, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
for n in range(len(lms)):
    ok, frame = cap.read(); ok2, orig = capo.read()
    if not ok or not ok2: break
    out = frame
    if lms[n] is not None:
        raw, sm, fwid = lms[n], smoothed[n], widths[n]
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        tcheek = np.zeros(g.shape, np.float32)
        cv2.fillPoly(tcheek, [np.vstack([sm[[9, 10, 11]], sm[[76, 74]]]).astype(np.int32)], 1)
        gain = (g[tcheek > 0].mean() / src_skin) if tcheek.sum() > 20 else 1.0
        dst_ctrl = control(sm)
        beard = cv2.GaussianBlur(warp(sgray, src_ctrl, dst_ctrl, g.shape) * gain, (0, 0), 0.6)
        m = warp(smask, src_ctrl, dst_ctrl, g.shape)
        # mouth hole cut from this frame's real lips, so the beard never covers them
        lipm = np.zeros(g.shape, np.float32)
        cv2.fillConvexPoly(lipm, cv2.convexHull(raw[OUTER_LIP].astype(np.int32)), 1)
        k = max(3, int(fwid * 0.04)) | 1
        lipm = cv2.dilate(lipm, np.ones((k, k), np.uint8))
        m *= 1 - lipm
        m = cv2.GaussianBlur(m, (0, 0), max(2.0, 0.035 * fwid))
        b3 = np.repeat(beard[..., None], 3, 2)
        out = frame * (1 - m[..., None]) + b3 * m[..., None]
        # restore the original lips + mouth interior (true lip-sync), feathered
        mouth = np.zeros(g.shape, np.float32)
        cv2.fillConvexPoly(mouth, cv2.convexHull(raw[OUTER_LIP].astype(np.int32)), 1)
        mouth = cv2.GaussianBlur(mouth, (0, 0), max(1.5, 0.025 * fwid))
        og = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # match original lips' brightness to the swapped face around them
        ring = (cv2.dilate((mouth > 0.5).astype(np.uint8), np.ones((k * 2 + 1,) * 2, np.uint8)) > 0) & (mouth < 0.1)
        if ring.sum() > 20:
            og = og + (np.repeat(out.mean(2)[..., None], 1, 2)[..., 0][ring].mean() - og[ring].mean()) * 0.5
        o3 = np.repeat(og[..., None], 3, 2)
        mm = (mouth * 0.9)[..., None]
        out = (out * (1 - mm) + o3 * mm).clip(0, 255).astype(np.uint8)
    if debug_frames:
        if n in debug_frames:
            cv2.imwrite(f"dbg_{n}.png", cv2.resize(out[300:700, 0:330], None, fx=2.5, fy=2.5))
    else:
        writer.write(out)
    if n % 50 == 0: print(n, flush=True)
if writer: writer.release()
print("done", len(lms))
