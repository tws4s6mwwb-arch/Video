import sys, os, cv2, numpy as np, onnxruntime as ort, insightface
from insightface.app import FaceAnalysis

HERE = os.path.dirname(os.path.abspath(__file__))
video, src_left, src_right, out_path = sys.argv[1:5]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
ENH_BLEND = 0.75  # how much GFPGAN restoration to apply (1 = full)

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                   allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(f"{HERE}/models/inswapper_128.onnx",
                                          providers=["CPUExecutionProvider"])
so = ort.SessionOptions(); so.intra_op_num_threads = 4
gfp = ort.InferenceSession(f"{HERE}/models/gfpgan_1.4.onnx", so, providers=["CPUExecutionProvider"])
gfp_in = gfp.get_inputs()[0].name

FFHQ = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                 [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float32) * 512

def biggest(img_path):
    faces = app.get(cv2.imread(img_path))
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

sources = [biggest(src_left), biggest(src_right)]

# soft elliptical mask in the 512 crop space
mask512 = np.zeros((512, 512), np.float32)
cv2.ellipse(mask512, (256, 270), (190, 230), 0, 0, 360, 1, -1)
mask512 = cv2.GaussianBlur(mask512, (0, 0), 25)

def enhance(frame, kps):
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), FFHQ, method=cv2.LMEDS)
    crop = cv2.warpAffine(frame, M, (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    x = crop[:, :, ::-1].astype(np.float32) / 255.0
    x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
    y = gfp.run(None, {gfp_in: x})[0][0]
    y = (np.clip(y.transpose(1, 2, 0), -1, 1) + 1) / 2 * 255
    y = y[:, :, ::-1].astype(np.float32)
    y = crop * (1 - ENH_BLEND) + y * ENH_BLEND
    Mi = cv2.invertAffineTransform(M)
    h, w = frame.shape[:2]
    back = cv2.warpAffine(y, Mi, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    m = cv2.warpAffine(mask512, Mi, (w, h))[..., None]
    return (frame * (1 - m) + back * m).clip(0, 255).astype(np.uint8)

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(3)), int(cap.get(4))
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

refs = None  # identity embeddings of the left / right people in the video
n = 0
while True:
    ok, frame = cap.read()
    if not ok or (limit and n >= limit):
        break
    faces = [f for f in app.get(frame) if f.det_score > 0.5]
    if refs is None and len(faces) >= 2:
        two = sorted(sorted(faces, key=lambda f: -f.det_score)[:2], key=lambda f: f.bbox[0])
        refs = [two[0].normed_embedding, two[1].normed_embedding]
    out = frame.copy()
    used = set()
    for f in sorted(faces, key=lambda f: -f.det_score):
        if refs is not None:
            sims = [float(np.dot(f.normed_embedding, r)) for r in refs]
            who = int(np.argmax(sims))
            if max(sims) < 0.15:  # weak identity match: fall back to screen side
                who = 0 if (f.bbox[0] + f.bbox[2]) / 2 < W / 2 else 1
        else:
            who = 0 if (f.bbox[0] + f.bbox[2]) / 2 < W / 2 else 1
        if who in used:
            continue
        used.add(who)
        out = swapper.get(out, f, sources[who], paste_back=True)
        out = enhance(out, f.kps)
    # the clip is black & white: desaturate swapped pixels wherever the original was grey
    sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    grey = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = np.where((sat < 40)[..., None], grey, out)
    writer.write(out)
    n += 1
    if n % 25 == 0:
        print(f"{n} frames", flush=True)
writer.release()
print("done", n)
