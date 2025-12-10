import os
import pickle

# =========================
# CONFIG
# =========================
TRAIN_DIR = "train"
TEST_DIR = "test"
PICKLE_DIR = "pickles"

# Stress folder names
STRESSED = ["angry", "fear", "sad"]

os.makedirs(PICKLE_DIR, exist_ok=True)


def create_pickle(dataset_path, output_file):
    image_paths = []
    labels = []

    print(f"Processing folder: {dataset_path}")

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        # 2-class labeling
        label = 1 if folder.lower() in STRESSED else 0

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            image_paths.append(img_path)
            labels.append(label)

    with open(output_file, "wb") as f:
        pickle.dump((image_paths, labels), f)

    print(f"Saved {output_file} with {len(labels)} samples.\n")


# Generate PKLs
create_pickle(TRAIN_DIR, os.path.join(PICKLE_DIR, "train.pkl"))
create_pickle(TEST_DIR, os.path.join(PICKLE_DIR, "test.pkl"))

print("Pickle files created successfully!")
