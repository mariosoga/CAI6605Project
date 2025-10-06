"""
Step 1: Inventory PlantVillage into multi-task labels without moving files.

Outputs (under outputs/pv_mtl_labels):
  - species_list.txt
  - disease_list.txt
  - labels.csv  (path,species,species_id,disease,disease_id,health,orig_class)
  - counts_species.txt
  - counts_health.txt

Health encoding: 1 = Healthy, 0 = Sick
"""

from pathlib import Path
from collections import Counter
import csv

# --- EDIT THIS to your dataset root ---
DATA_ROOT = Path(r"C:\Users\y-pol\PyCharmMiscProject\plant_care_assistant\data\plantvillage dataset\color")

# Where to write artifacts
OUT_DIR = Path(r"C:\Users\y-pol\PyCharmMiscProject\plant_care_assistant\outputs\pv_mtl_labels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Image extensions to include
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_classname(name: str):
    """
    'Tomato___Late_blight' -> ('Tomato', 'Late_blight')
    If no '___', treat whole as species, disease='unknown'
    """
    if "___" in name:
        sp, dz = name.split("___", 1)
    else:
        sp, dz = name, "unknown"
    return sp, dz


def main():
    assert DATA_ROOT.is_dir(), f"DATA_ROOT not found: {DATA_ROOT}"

    # 1) Collect class folders (ImageFolder-style)
    class_dirs = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class subfolders found under {DATA_ROOT}")

    # 2) Build species/disease vocab from folder names
    species_set, disease_set = set(), set()
    class_to_parts = {}  # folder name -> (species, disease)
    for d in class_dirs:
        sp, dz = parse_classname(d.name)
        species_set.add(sp)
        disease_set.add(dz)
        class_to_parts[d.name] = (sp, dz)

    species_list = sorted(species_set)
    disease_list = sorted(disease_set)
    species_to_id = {s: i for i, s in enumerate(species_list)}
    disease_to_id = {d: i for i, d in enumerate(disease_list)}

    # 3) Walk all images, write rows
    labels_csv = OUT_DIR / "labels.csv"
    n_rows = 0
    counts_species = Counter()
    counts_health = Counter()

    with labels_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "species", "species_id", "disease", "disease_id", "health", "orig_class"])

        for d in class_dirs:
            sp, dz = class_to_parts[d.name]
            sp_id = species_to_id[sp]
            dz_id = disease_to_id[dz]
            health = 1 if dz.lower() == "healthy" else 0

            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in EXTS:
                    writer.writerow([str(p), sp, sp_id, dz, dz_id, health, d.name])
                    n_rows += 1
                    counts_species[sp] += 1
                    counts_health[health] += 1

    # 4) Save vocab files and simple counts
    (OUT_DIR / "species_list.txt").write_text("\n".join(species_list), encoding="utf-8")
    (OUT_DIR / "disease_list.txt").write_text("\n".join(disease_list), encoding="utf-8")

    with (OUT_DIR / "counts_species.txt").open("w", encoding="utf-8") as f:
        for sp in species_list:
            f.write(f"{sp}\t{counts_species[sp]}\n")

    with (OUT_DIR / "counts_health.txt").open("w", encoding="utf-8") as f:
        f.write(f"Healthy(1)\t{counts_health[1]}\n")
        f.write(f"Sick(0)\t{counts_health[0]}\n")

    print(f"[OK] Wrote: {labels_csv}  ({n_rows} rows)")
    print(f"[OK] Species: {len(species_list)} -> {OUT_DIR / 'species_list.txt'}")
    print(f"[OK] Diseases: {len(disease_list)} -> {OUT_DIR / 'disease_list.txt'}")
    print(f"[OK] Counts -> {OUT_DIR / 'counts_species.txt'}, {OUT_DIR / 'counts_health.txt'}")


if __name__ == "__main__":
    main()
