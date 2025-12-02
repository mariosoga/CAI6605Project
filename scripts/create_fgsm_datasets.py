import shutil
from pathlib import Path
from tqdm import tqdm
from itertools import count


def short_copy(src_files, dst_class_path, prefix=None):
    """
    Copies files into dst_class_path using short, collision-free filenames.
    Example: 000001.jpg, 000002.jpg
    """
    ctr = count()
    for img_file in src_files:
        if img_file.is_file():
            idx = next(ctr)
            if prefix:
                new_name = f"{prefix}_{idx:06d}{img_file.suffix}"
            else:
                new_name = f"{idx:06d}{img_file.suffix}"

            shutil.copy(img_file, dst_class_path / new_name)


def main():
    base_dir = Path('data/dataset/color')
    fgsm_dir = Path('outputs/fgsm_test')
    output_root = Path('data/dataset/augmented_datasets')

    # Iterate over each epsilon folder
    for eps_folder in tqdm(list(fgsm_dir.iterdir()), desc="Epsilon levels"):
        if eps_folder.is_dir():
            # Create a distinct output dataset for this epsilon
            output_dir = output_root / f"augmented_{eps_folder.name}"
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n📁 Creating augmented dataset for {eps_folder.name} at {output_dir}")

            # Step 1 — Copy base dataset with short filenames
            for class_folder in tqdm(list(base_dir.iterdir()),
                                     desc="Copying base classes",
                                     leave=False):
                if class_folder.is_dir():
                    dst_class_path = output_dir / class_folder.name
                    dst_class_path.mkdir(parents=True, exist_ok=True)

                    # Use short filenames
                    short_copy(class_folder.iterdir(), dst_class_path)

            # Step 2 — Add FGSM images for this epsilon, short filenames with prefix
            eps_prefix = eps_folder.name.replace("eps_", "e")

            for class_folder in tqdm(list(eps_folder.iterdir()),
                                     desc=f"Adding {eps_folder.name} classes",
                                     leave=False):
                if class_folder.is_dir():
                    dst_class_path = output_dir / class_folder.name
                    if not dst_class_path.exists():
                        dst_class_path.mkdir(parents=True, exist_ok=True)
                        print(f"⚠️ Created new class folder '{class_folder.name}' for epsilon {eps_folder.name}")

                    # Use short filenames, but with epsilon prefix to distinguish
                    short_copy(class_folder.iterdir(), dst_class_path, prefix=eps_prefix)

            print(f"✅ Finished augmented dataset for {eps_folder.name}")


if __name__ == "__main__":
    main()
