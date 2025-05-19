import os, shutil, random, csv

# === CONFIGURATION ===
SOURCE_ROOT = "C:/Users/dones/Downloads/archive (3)/COVID-19_Radiography_Dataset"
 # CHANGE THIS
DEST_ROOT   = "../1_data"  # output path relative to script location
IMG_DIR     = os.path.join(DEST_ROOT, "images")
TRAIN_CSV   = os.path.join(DEST_ROOT, "train.csv")
VAL_CSV     = os.path.join(DEST_ROOT, "val.csv")

NORMAL_DIR     = os.path.join(SOURCE_ROOT, "NORMAL", "images")
PNEUMONIA_DIR  = os.path.join(SOURCE_ROOT, "Viral Pneumonia", "images")

SPLIT_RATIO    = 0.8

# === MAKE DESTINATION FOLDERS ===
os.makedirs(IMG_DIR, exist_ok=True)

# === GATHER & LABEL ===
all_samples = []

def gather_images(source_dir, label):
    for fname in os.listdir(source_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            new_name = f"{label}_{fname}"
            shutil.copy(os.path.join(source_dir, fname), os.path.join(IMG_DIR, new_name))
            all_samples.append((new_name, label))

gather_images(NORMAL_DIR,    0)
gather_images(PNEUMONIA_DIR, 1)

print(f"Total images: {len(all_samples)}")
random.shuffle(all_samples)

# === SPLIT ===
split_idx = int(len(all_samples) * SPLIT_RATIO)
train_data = all_samples[:split_idx]
val_data   = all_samples[split_idx:]

# === WRITE CSVs ===
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

write_csv(TRAIN_CSV, train_data)
write_csv(VAL_CSV,   val_data)

print(f"Wrote {len(train_data)} train and {len(val_data)} val samples.")
