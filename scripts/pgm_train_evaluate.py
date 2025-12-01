import os
import shutil
from pathlib import Path

from tqdm import tqdm


def main():
    base_dir = Path('data/dataset/color')
    fgsm_dir = Path('outputs/fgsm_test')
    output_dir = Path('data/dataset/augmented_dataset')

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Copy base dataset structure and images
    print("📁 Copying base dataset...")
    for class_folder in tqdm(list(base_dir.iterdir()), desc="Base classes"):
        if class_folder.is_dir():
            dst_class_path = output_dir / class_folder.name
            dst_class_path.mkdir(parents=True, exist_ok=True)

            for img_file in class_folder.iterdir():
                if img_file.is_file():
                    shutil.copy(img_file, dst_class_path / img_file.name)

    # Step 2: Add FGSM images to each class folder
    print("⚡ Merging FGSM adversarial images...")
    for eps_folder in tqdm(list(fgsm_dir.iterdir()), desc="Epsilon levels"):
        if eps_folder.is_dir():
            for class_folder in eps_folder.iterdir():
                if class_folder.is_dir():
                    dst_class_path = output_dir / class_folder.name
                    dst_class_path.mkdir(parents=True, exist_ok=True)

                    for img_file in class_folder.iterdir():
                        if img_file.is_file():
                            new_name = f"{eps_folder.name}_{img_file.name}"
                            shutil.copy(img_file, dst_class_path / new_name)

    print("✅ Dataset creation complete.")


if __name__ == "__main__":
    main()

