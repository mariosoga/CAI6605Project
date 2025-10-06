# Smart Plant Care Assistant (Multi-Task: Species + Health + Disease)

This repo trains an EfficientNet backbone with **three heads**:

- **Species** — which plant (Tomato, Apple, …)
- **Health** — **0 = Sick**, **1 = Healthy**
- **Disease** — global disease label (includes `"healthy"`)

> Why: PlantVillage folders are named `Species___Disease`. A single-head classifier learns the combined label and can’t
> say “Tomato — Healthy”. Multitask fixes that.

---

## 0) Requirements

```bash
pip install torch torchvision timm scikit-learn pytorch-grad-cam
```

---

## 1) Dataset (PlantVillage)

Point `--data_dir` to your PlantVillage **color** root that looks like:

```
data/plantvillage dataset/color/
  Tomato___Late_blight/
  Tomato___healthy/
  Apple___Black_rot/
  ...
```

---

## 2) (Optional) Build a CSV index & vocab

This inventories your dataset and writes a CSV plus vocab files—handy for checks and later tooling.

```bash
python build_pv_labels.py
# writes to outputs/pv_mtl_labels/:
#   - labels.csv (path, species_id, disease_id, health, ...)
#   - species_list.txt
#   - disease_list.txt
#   - counts_*.txt
```

Health encoding throughout this project is **1 = Healthy, 0 = Sick**.

---

## 3) Train

### Single-task (legacy, combined disease classes)

```bash
python train_and_eval_model.py   --data_dir "C:\Users\y-pol\PyCharmMiscProject\plant_care_assistant\data\plantvillage dataset\color"   --out_dir outputs --epochs 20   --grad_cam --grad_cam_k 50
```

### Multitask (species + health + disease) — recommended

```bash
python scripts/train_and_eval_model.py   --data_dir "C:\Users\y-pol\PyCharmMiscProject\plant_care_assistant\data\plantvillage dataset\color"   --out_dir outputs --epochs 20 --multitask   --healthy_keyword healthy   --health_loss_weight 1.0   --save_hard_examples --hard_k 50   --save_confusion_pairs --pairs_m 5 --examples_per_pair 8   --grad_cam --grad_cam_k 50 --grad_cam_task disease
```

**Notes**

- Backbone: `--model efficientnet_b0` (default, via `timm`)
- Images are resized to `--img_size 224`, normalized to ImageNet stats by default.
- Scheduler: `--lr_scheduler plateau` on validation loss.
- Early stop: `--early_stop_patience 3`.

---

## 4) Grad-CAM

Enable with `--grad_cam`. By default we visualize **disease**; for the health head:

```bash
--grad_cam --grad_cam_task health
```

Other useful knobs:

```bash
--grad_cam_layer ""      # auto: EfficientNet conv_head
--norm_mean 0.485,0.456,0.406
--norm_std  0.229,0.224,0.225
--cam_alpha 0.45
```

**Outputs**

- Overconfident wrong CAMs: `outputs/hard_examples/overconfident_wrong_cam/`
- Uncertain correct CAMs:   `outputs/hard_examples/uncertain_correct_cam/`

---

## 5) Artifacts

Under `--out_dir`:

- **Checkpoint**: `best.pt`
    - Single-task: `class_names` (combined labels)
    - Multi-task: `species_list`, `disease_list`, and `class_names = disease_list` (for backward compat)
- **Disease report (multiclass)**
    - `classification_report.txt`
    - `confusion_matrix.png`
    - `val_predictions.csv`
- **Health report (binary, if multitask)**
    - `classification_report_health.txt`
    - `val_predictions_health.csv`
- **Hard examples**
    - Disease: `hard_examples/overconfident_wrong`, `hard_examples/uncertain_correct`
    - Health:  `hard_examples_binary/...` (if `--save_hard_examples`)
- **Top confusions**: `confusions/` (if `--save_confusion_pairs`)

---

## 5.2) test model

```bash
python scripts/test_model.py --ckpt outputs/best.pt --data_dir "data/plantvillage dataset/test"
```

---

## 6) Inference (pretty output)

#### Single image:

```bash
python test_infer.py --ckpt outputs/best.pt --image "data/test/grape_ecsa.jpg"

```
---

#### Folder of images (recursively):

```bash
python test_infer.py --ckpt outputs/best.pt --image_dir "data/plantvillage dataset/test"

```

---

## 7) Tips & Troubleshooting

- **Health labels**: Everywhere we assume **0=Sick, 1=Healthy**. Binary reports use labels `('Sick','Healthy')`.
- **Class imbalance**: Consider class weights for species/disease heads.
- **Generalization**: PlantVillage is studio-like; for field photos, use stronger augs or test on more realistic sets.
- **Determinism**: use `set_seed(...)` and `seed_worker` in DataLoaders for reproducibility (see `core/utils.py`).

## 9) License

MIT (or your choice).