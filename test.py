import time
import random
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from transformers import AutoModelForImageClassification, pipeline
from timm.data.transforms_factory import create_transform
import torchvision

# ---------------------
# CONFIG
# ---------------------
COCO_ROOT = "coco"
ANN_FILE = f"{COCO_ROOT}/annotations/instances_val2017.json"
IMG_DIR = f"{COCO_ROOT}/val2017"

NUM_IMAGES = 100
RESOLUTION = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------
# LOAD COCO
# ---------------------
coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()
sample_ids = random.sample(img_ids, NUM_IMAGES)

cat_map = coco.loadCats(coco.getCatIds())
coco_id_to_name = {c["id"]: c["name"] for c in cat_map}

# ---------------------
# LOAD MODELS
# ---------------------
print("Loading NextViT...")
pipe = pipeline("image-classification", model="timm/nextvit_small.bd_in1k", device=0)

print("Loading MambaVision...")
model_name = "nvidia/MambaVision-B-21K"
model = AutoModelForImageClassification.from_pretrained(model_name, trust_remote_code=True)
model.eval().to(DEVICE)

print("✅ Models loaded")

# ---------------------
# TRANSFORM FOR MAMBA
# ---------------------
input_resolution = (3, RESOLUTION, RESOLUTION)

transform = create_transform(
    input_size=input_resolution,
    is_training=False,
    mean=model.config.mean,
    std=model.config.std,
    crop_mode=model.config.crop_mode,
    crop_pct=model.config.crop_pct
)

# ---------------------
# COCO → IMAGENET NAME MATCH
# ---------------------
imagenet_labels = model.config.id2label

def coco_to_imagenet_match(coco_names, pred_label):
    pred_label = pred_label.lower()
    for coco_name in coco_names:
        if coco_name.lower() in pred_label or pred_label in coco_name.lower():
            return True
    return False

# ---------------------
# BENCHMARK LOOP
# ---------------------
correct_mamba = 0
correct_nextvit = 0
total = 0

mamba_time = 0
nextvit_time = 0

print("Running COCO benchmark...")

# Warmup
for img_id in sample_ids[:5]:
    img_info = coco.loadImgs(img_id)[0]
    img_path = f"{IMG_DIR}/{img_info['file_name']}"
    img = Image.open(img_path).convert("RGB")
    inputs = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        _ = model(inputs)
        _ = pipe(img_path)

torch.cuda.synchronize()

# Run evaluation
for img_id in tqdm(sample_ids):

    img_info = coco.loadImgs(img_id)[0]
    img_path = f"{IMG_DIR}/{img_info['file_name']}"
    img = Image.open(img_path).convert("RGB")

    # GT labels from COCO
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    coco_labels = set()
    for ann in anns:
        coco_labels.add(coco_id_to_name[ann["category_id"]])

    if len(coco_labels) == 0:
        continue

    # ---------------- MAMBA ----------------
    inputs = transform(img).unsqueeze(0).cuda()

    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        outputs = model(inputs)

    torch.cuda.synchronize()
    mamba_time += time.time() - t0

    logits = outputs["logits"]
    mamba_idx = logits.argmax(-1).item()
    mamba_label = model.config.id2label[mamba_idx]

    # ---------------- NEXTVIT ----------------
    torch.cuda.synchronize()
    t0 = time.time()

    nextvit_out = pipe(img_path)

    torch.cuda.synchronize()
    nextvit_time += time.time() - t0

    nextvit_label = nextvit_out[0]["label"]

    # ---------------- ACCURACY ----------------
    if coco_to_imagenet_match(coco_labels, mamba_label):
        correct_mamba += 1

    if coco_to_imagenet_match(coco_labels, nextvit_label):
        correct_nextvit += 1

    total += 1


# ---------------------
# RESULTS
# ---------------------
mamba_acc = correct_mamba / max(total, 1)
nextvit_acc = correct_nextvit / max(total, 1)

mamba_tput = total / mamba_time
nextvit_tput = total / nextvit_time

print("\n================ COCO RESULTS ================")
print(f"Images evaluated: {total}")
print("----------------------------------------------")
print(f"MambaVision Top-1 Accuracy: {mamba_acc:.4f}")
print(f"NextViT   Top-1 Accuracy:   {nextvit_acc:.4f}")
print("----------------------------------------------")
print(f"MambaVision Throughput: {mamba_tput:.2f} img/sec")
print(f"NextViT   Throughput:   {nextvit_tput:.2f} img/sec")
print("==============================================")