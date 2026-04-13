import os
import random
import shutil
from PIL import Image

random.seed(42)

SOURCE_DIR = "data_raw/PetImages"
OUTPUT_DIR = "data"
MAX_PER_CLASS = 4000  # use 4000 cats and 4000 dogs

SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

classes = {
    "Cat": "cat",
    "Dog": "dog"
}

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def make_folders():
    for split in SPLITS:
        for cls_name in classes.values():
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls_name), exist_ok=True)

def collect_images(class_folder):
    folder = os.path.join(SOURCE_DIR, class_folder)
    files = []

    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and is_valid_image(path):
            files.append(path)

    random.shuffle(files)
    return files[:MAX_PER_CLASS]

def split_list(items):
    n = len(items)
    train_end = int(n * SPLITS["train"])
    val_end = train_end + int(n * SPLITS["val"])
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:]
    }

def copy_split(files_by_split, out_class_name):
    for split, files in files_by_split.items():
        for src_path in files:
            filename = os.path.basename(src_path)
            dst_path = os.path.join(OUTPUT_DIR, split, out_class_name, filename)
            shutil.copy2(src_path, dst_path)

def main():
    make_folders()

    for original_class, out_class in classes.items():
        files = collect_images(original_class)
        print(f"{original_class}: using {len(files)} valid images")
        split_files = split_list(files)
        for split, split_list_files in split_files.items():
            print(f"  {split}: {len(split_list_files)}")
        copy_split(split_files, out_class)

    print("Done.")

if __name__ == "__main__":
    main()