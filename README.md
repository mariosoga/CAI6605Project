# Smart Plant Recognition Tool (SPROUT) (Multi-Task: Species + Health + Disease)

## Midterm Project:
---
This repo trains an EfficientNet backbone with **three heads**:

- **Species** — which plant (Tomato, Apple, …)
- **Health** — **0 = Sick**, **1 = Healthy**
- **Disease** — global disease label (includes `"healthy"`, `"sick"`)

This approach allows the AI system to recognize the species, disease, and whether the plant is healthy or sick (
this is a derivative classification from the disease head)

 ---

### Quickstart

Below is the main steps to use SPROUT:

```bash
pip install -r requirements.txt  # Install dependencies
python scripts/build_pv_labels.py # Build label index and vocab
python scripts/train_and_eval_model.py --data_dir path/to/data --multitask  # Train the model
python scripts/test_infer.py --ckpt outputs/best.pt --image_path "data/infer/apple_scap.jpg" # Run inference
```

---

### 1) Requirements

The code uses torch torchvision and timm to train, evaluate and test the system. It also uses scikit-learn for
reporting. Run the command below to install dependencies.

```bash
pip install -r requirements.txt
```

---

### 2) Dataset

This system utilizes a unified dataset derived from both PlantVillage and PlantDoc sources. The two datasets have been
merged into a single collection, organized according to the PlantVillage directory structure.

To specify the dataset location, use the `--data_dir` argument and point it to the root of your dataset. For instance,
the
current setup uses the color folder within the PlantVillage hierarchy as the root:

```
data/plantvillage dataset/color/
  ├── Tomato___Late_blight/
  ├── Tomato___healthy/
  ├── Apple___Black_rot/
  └── ...
```

---

### 3) Build a CSV index & vocab

This step scans your dataset and generates a set of helpful files for inspection and downstream tooling:

- A CSV index of all image paths and labels
- Vocabulary files for species and diseases
- Class distribution counts
  These files are essential if you add new classes to the dataset, as species_list.txt and disease_list.txt are used
  during both training and evaluation.
  Run the following script:

```bash
python scripts/build_pv_labels.py
```

This will create the following outputs in outputs/pv_mtl_labels/:

```
labels.csv           # Contains: path, species_id, disease_id, health, ...
species_list.txt     # List of all species
disease_list.txt     # List of all diseases
counts_*.txt         # Class distribution stats
```

Throughout this project, health status is encoded as:

- 0 → Healthy
- 1 → Sick

---

### 4) Train

#### Single-task (legacy, combined disease classes)

The system originally used a single-head classifier that predicted combined species_disease labels. While this coupled
approach has since evolved into a multi-head architecture with separate outputs for species, disease, and health status,
single-task mode remains supported for legacy workflows or simpler use cases.
To run training and evaluation in single-task mode, use the following command:

```bash
python scripts/train_and_eval_model.py --data_dir "data/dataset/color"   --out_dir outputs --epochs 20   --grad_cam --grad_cam_k 50
```

#### Multitask (species + health + disease) — recommended

This is the recommended mode for running the Smart Plant Care Assistant. It performs training and evaluation using a
multi-head architecture that predicts species, disease, and health status simultaneously.
The interface supports a wide range of configurable parameters to give users more control over the training
process. Below is a complete list of available command-line arguments:

| **Category**                   | **Parameter**            | **Type** | **Default**         | **Description**                                     |
|--------------------------------|--------------------------|----------|---------------------|-----------------------------------------------------|
| **General Setup**              | `--data_dir`             | `str`    | *data_dir*          | Path to dataset directory                           |
|                                | `--out_dir`              | `str`    | `outputs`           | Directory for model outputs and logs                |
|                                | `--model`                | `str`    | `efficientnet_b0`   | Backbone model architecture                         |
|                                | `--img_size`             | `int`    | `224`               | Image input size (square)                           |
|                                | `--batch_size`           | `int`    | `32`                | Number of samples per batch                         |
|                                | `--epochs`               | `int`    | `10`                | Number of training epochs                           |
|                                | `--lr`                   | `float`  | `3e-4`              | Learning rate                                       |
|                                | `--weight_decay`         | `float`  | `1e-4`              | Weight decay (L2 regularization)                    |
|                                | `--seed`                 | `int`    | `42`                | Random seed for reproducibility                     |
|                                | `--val_split`            | `float`  | `0.15`              | Fraction of data used for validation                |
| **Early Stopping & Scheduler** | `--early_stop_patience`  | `int`    | `3`                 | Epochs to wait before stopping if no improvement    |
|                                | `--early_stop_min_delta` | `float`  | `0.0`               | Minimum delta for improvement recognition           |
|                                | `--lr_scheduler`         | `str`    | `plateau`           | Learning rate scheduler (`plateau` or `none`)       |
|                                | `--lr_factor`            | `float`  | `0.5`               | LR reduction factor on plateau                      |
|                                | `--lr_patience`          | `int`    | `2`                 | Epochs to wait before LR reduction                  |
|                                | `--lr_min`               | `float`  | `1e-6`              | Minimum learning rate                               |
| **Multi-Task Learning**        | `--multitask`            | `flag`   | `False`             | Enable multi-head model ( species, health, disease) |
|                                | `--healthy_keyword`      | `str`    | `healthy`           | Keyword to identify healthy samples                 |
|                                | `--health_loss_weight`   | `float`  | `1.0`               | Weight for health classification loss               |
| **Analysis Options**           | `--save_hard_examples`   | `flag`   | `False`             | Save hardest examples for inspection                |
|                                | `--hard_k`               | `int`    | `50`                | Number of hardest examples to save                  |
|                                | `--save_confusion_pairs` | `flag`   | `False`             | Save top confusion pairs between classes            |
|                                | `--pairs_m`              | `int`    | `5`                 | Number of confusion pairs per category              |
|                                | `--examples_per_pair`    | `int`    | `8`                 | Number of examples per confusion pair               |
| **Grad-CAM Visualization**     | `--grad_cam`             | `flag`   | `False`             | Enable Grad-CAM overlay generation                  |
|                                | `--grad_cam_k`           | `int`    | `50`                | Top-K examples per bucket for Grad-CAM              |
|                                | `--grad_cam_task`        | `str`    | `disease`           | Task to visualize (`disease` or `health`)           |
|                                | `--grad_cam_layer`       | `str`    | `""`                | Layer to visualize (auto if empty)                  |
|                                | `--norm_mean`            | `str`    | `0.485,0.456,0.406` | Image normalization mean (CSV)                      |
|                                | `--norm_std`             | `str`    | `0.229,0.224,0.225` | Image normalization std (CSV)                       |
|                                | `--cam_alpha`            | `float`  | `0.45`              | Transparency for Grad-CAM overlay                   |

This is an example for running training and evaluation.
##python scripts/train_and_eval_model.py --data_dir "data/dataset/color"   --out_dir outputs/fsgmRun --epochs 20 --multitask   --healthy_keyword healthy   --health_loss_weight 1.0   --save_hard_examples --hard_k 50   --save_confusion_pairs --pairs_m 5 --examples_per_pair 8   --grad_cam --grad_cam_k 50 --grad_cam_task disease
##python scripts/train_and_eval_model.py --data_dir "data/dataset/augmented_dataset"   --out_dir outputs/fsgmRun --epochs 20 --multitask   --healthy_keyword healthy   --health_loss_weight 1.0   --save_hard_examples --hard_k 50   --save_confusion_pairs --pairs_m 5 --examples_per_pair 8   --grad_cam --grad_cam_k 50 --grad_cam_task disease

```bash
python scripts/train_and_eval_model.py --data_dir "data/dataset/augmented_datasets/augmented_eps_0.000"   --out_dir outputs/fsgmRun/eps_0.000 --epochs 20 --multitask   --healthy_keyword healthy   --health_loss_weight 1.0   --save_hard_examples --hard_k 50   --save_confusion_pairs --pairs_m 5 --examples_per_pair 8   --grad_cam --grad_cam_k 50 --grad_cam_task disease
python scripts/train_and_eval_model.py --data_dir "data/dataset/augmented_datasets/augmented_eps_0.050"   --out_dir outputs/fsgmRun/eps_0.050 --epochs 20 --multitask   --healthy_keyword healthy   --health_loss_weight 1.0   --save_hard_examples --hard_k 50   --save_confusion_pairs --pairs_m 5 --examples_per_pair 8   --grad_cam --grad_cam_k 50 --grad_cam_task disease
python scripts/train_and_eval_model.py --data_dir "data/dataset/augmented_datasets/augmented_eps_0.100"   --out_dir outputs/fsgmRun/eps_0.100 --epochs 20 --multitask   --healthy_keyword healthy   --health_loss_weight 1.0   --save_hard_examples --hard_k 50   --save_confusion_pairs --pairs_m 5 --examples_per_pair 8   --grad_cam --grad_cam_k 50 --grad_cam_task disease
python scripts/train_and_eval_model.py --data_dir "data/dataset/augmented_datasets/augmented_eps_0.150"   --out_dir outputs/fsgmRun/eps_0.150 --epochs 20 --multitask   --healthy_keyword healthy   --health_loss_weight 1.0   --save_hard_examples --hard_k 50   --save_confusion_pairs --pairs_m 5 --examples_per_pair 8   --grad_cam --grad_cam_k 50 --grad_cam_task disease
python scripts/train_and_eval_model.py --data_dir "data/dataset/augmented_datasets/augmented_eps_0.200"   --out_dir outputs/fsgmRun/eps_0.200 --epochs 20 --multitask   --healthy_keyword healthy   --health_loss_weight 1.0   --save_hard_examples --hard_k 50   --save_confusion_pairs --pairs_m 5 --examples_per_pair 8   --grad_cam --grad_cam_k 50 --grad_cam_task disease
python scripts/train_and_eval_model.py --data_dir "data/dataset/augmented_datasets/augmented_eps_0.250"   --out_dir outputs/fsgmRun/eps_0.250 --epochs 20 --multitask   --healthy_keyword healthy   --health_loss_weight 1.0   --save_hard_examples --hard_k 50   --save_confusion_pairs --pairs_m 5 --examples_per_pair 8   --grad_cam --grad_cam_k 50 --grad_cam_task disease
python scripts/train_and_eval_model.py --data_dir "data/dataset/augmented_datasets/augmented_eps_0.300"   --out_dir outputs/fsgmRun/eps_0.300 --epochs 20 --multitask   --healthy_keyword healthy   --health_loss_weight 1.0   --save_hard_examples --hard_k 50   --save_confusion_pairs --pairs_m 5 --examples_per_pair 8   --grad_cam --grad_cam_k 50 --grad_cam_task disease
```



---

### 5) Artifacts

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

### 6) Test

SPROUT does not include testing within the training-evaluation pipeline by design. Separating testing allows
the independent inspection and comparison of metrics from both the evaluation and testing stages.

To run testing, place your test images in a separate folder and point data_dir to that location, and ckpt to the
location of the trained model check point:

```bash
python scripts/test_model.py --ckpt outputs/best.pt --data_dir "data/dataset/test" --out_dir "outputs/fsgm_test"
python scripts/test_model.py --ckpt outputs/fsgmRun/best.pt --data_dir "data/dataset/segmented" --out_dir "outputs/after_at"
python scripts/test_model.py --ckpt outputs/best.pt --data_dir "data/dataset/segmented" --out_dir "outputs/before_at"
python scripts/test_model.py --ckpt outputs/fsgmRun/eps_0.000/best.pt --data_dir "data/dataset/augmented_datasets/augmented_eps_0.000" --out_dir "outputs/base_eps0"
python scripts/test_model.py --ckpt outputs/fsgmRun/eps_0.300/best.pt --data_dir "data/dataset/augmented_datasets/augmented_eps_0.300" --out_dir "outputs/base_eps0.300"
python scripts/test_model.py --ckpt outputs/best.pt --data_dir "data/dataset/augmented_datasets/augmented_eps_0.200" --out_dir "outputs/base_model/0_200_perturbation"

```

---

### 7) Inference

##### Single image:

You can run inference on individual images using the trained model checkpoint. This is useful for quick predictions or
testing the model on new samples.

Alternatively you can run inference on a folder of images by using `--image_dir` instead of `--image_path`

You may Use one of the following:

```bash
#python scripts/test_infer.py --ckpt outputs/best.pt --image_path "data/infer/apple_scap.jpg"
#or
#python scripts/test_infer.py --ckpt outputs/best.pt --image_path "data/infer/tomato_early_blight.jpg"
#or
#python scripts/test_infer.py --ckpt outputs/best.pt --image_path "data/infer/bacterial_spot_tomato.jpg"
python scripts/test_infer.py --ckpt outputs/best.pt --image_dir "data/infer"

```

---

### 8) Tips & Troubleshooting

- **Health labels**: Everywhere we assume **0=Sick, 1=Healthy**. Binary reports use labels `('Sick','Healthy')`.
- **Class imbalance**: Consider class weights for species/disease heads.
- **Determinism**: use `set_seed(...)` and `seed_worker` in DataLoaders for reproducibility (see `core/utils.py`).

```bash
python scripts/create_fgsm_datasets.py
```

---
## Final Project:

### 1) Transparency and Explainability: Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) provides visual explanations of what regions in the image most
influenced the model’s decision.
This is especially useful for interpreting disease localization and validating model trustworthiness.

Enable Grad-Cam with the ```--grad_cam``` flag:

By default, Grad-CAM visualizes activations for the disease classification head.
To visualize the health head instead, specify:

```--grad_cam --grad_cam_task health```

Other useful controls:

```bash
--grad_cam_layer ""      # Auto-selects the best conv layer (e.g., EfficientNet's conv_head)
--norm_mean 0.485,0.456,0.406   # Image normalization mean (CSV format)
--norm_std  0.229,0.224,0.225   # Image normalization std (CSV format)
--cam_alpha 0.45                # Overlay transparency (0 = image only, 1 = heatmap only)
```

**Outputs**

- Overconfident wrong CAMs: `outputs/hard_examples/overconfident_wrong_cam/`
- Uncertain correct CAMs:   `outputs/hard_examples/uncertain_correct_cam/`