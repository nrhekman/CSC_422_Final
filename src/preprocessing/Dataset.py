import kagglehub
import os
import shutil
from tqdm import tqdm
import random
import numpy as np



# Download latest version
pest_path = kagglehub.dataset_download("rupankarmajumdar/crop-pests-dataset")
print(pest_path)

train_path = os.path.join(pest_path, "train", "images")
valid_path = os.path.join(pest_path, "valid", "images")
test_path = os.path.join(pest_path, "test", "images")

# --- INPUT FOLDERS ---
source_folders = [
    train_path,
    valid_path,
    test_path
]

# --- OUTPUT ROOT (everything goes here) ---
dest_path = pest_path
n = 4  # Number of levels to go up
for _ in range(n):
    dest_path = os.path.dirname(dest_path)

destination_root = os.path.join(dest_path, "Combined_DS", "images")

# All possible classes (case-insensitive)
CLASSES = [
    "Ants", "Bees", "Beetle", "Catterpillar", "Earthworms",
    "Earwig", "Grasshopper", "Moth", "Slug", "Snail",
    "Wasp", "Weevil", "Healthy"
]

# Make lookup in lowercase
CLASS_LOOKUP = {cls.lower(): cls for cls in CLASSES}

# Ensure destination folders exist
for cls in CLASSES:
    os.makedirs(os.path.join(destination_root, cls), exist_ok=True)

def get_class_from_filename(filename):
    """
    Extract class from filename like:
    ants-17-_jpg.rf.366ce3d542821626b2926e3142d1bb64
    Returns canonical class name (e.g. 'Ants')
    """
    prefix = filename.split("-")[0].lower()
    return CLASS_LOOKUP.get(prefix, None)

# Process all folders
for src in source_folders:
    if not os.path.isdir(src):
        print(f"Skipping missing folder: {src}")
        continue

    for fname in os.listdir(src):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        cls = get_class_from_filename(fname)
        src_path = os.path.join(src, fname)

        if cls is None:
            print(f"⚠️ Could not determine class for: {fname}")
            continue

        dst_path = os.path.join(destination_root, cls, fname)

        shutil.copy2(src_path, dst_path)   # or use move: shutil.move()
        print(f"Moved {fname} → {cls}/")

print("✨ Done!")

# Add Healthy class images

# path to your crop folders (Wheat, Corn, etc.)
healthy_path = kagglehub.dataset_download("mdwaquarazam/agricultural-crops-image-classification")
crop_root =  os.path.join(healthy_path, "Agricultural-crops")

# where your classification dataset is
output_class = os.path.join(destination_root, "Healthy")

for crop_folder in os.listdir(crop_root):
    crop_path = os.path.join(crop_root, crop_folder)
    
    if not os.path.isdir(crop_path):
        continue

    for img in tqdm(os.listdir(crop_path), desc=f"Merging {crop_folder}"):
        if img.lower().endswith((".jpg", ".png", ".jpeg")):
            src = os.path.join(crop_path, img)
            dst = os.path.join(output_class, f"{crop_folder}_{img}")
            shutil.copy(src, dst)

print("✔ All images collected into ../images/ by category.")



# --------------------------------------
# CONFIG
# --------------------------------------

SOURCE_ROOT = destination_root     # all images organized into class folders
DEST_ROOT   = os.path.join(dest_path, "Combined_DS", "dataset")     # final dataset with train/val/test

# train/val/test split ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# ensure repeatability (optional)
random.seed(42)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# --------------------------------------
# HELPERS
# --------------------------------------

def make_dirs():
    """Create train/val/test/class subfolders."""
    for split in ["train", "val", "test"]:
        for cls in os.listdir(SOURCE_ROOT):
            os.makedirs(os.path.join(DEST_ROOT, split, cls), exist_ok=True)


def split_data():
    """Split each class folder into 70/15/15 and copy."""
    for cls in os.listdir(SOURCE_ROOT):
        cls_path = os.path.join(SOURCE_ROOT, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [f for f in os.listdir(cls_path) if f.lower().endswith(IMG_EXTS)]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val   = int(n * VAL_SPLIT)
        # rest → test
        n_test  = n - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs   = images[n_train:n_train+n_val]
        test_imgs  = images[n_train+n_val:]

        # copy images
        for img in train_imgs:
            shutil.copy2(
                os.path.join(cls_path, img),
                os.path.join(DEST_ROOT, "train", cls, img)
            )
        for img in val_imgs:
            shutil.copy2(
                os.path.join(cls_path, img),
                os.path.join(DEST_ROOT, "val", cls, img)
            )
        for img in test_imgs:
            shutil.copy2(
                os.path.join(cls_path, img),
                os.path.join(DEST_ROOT, "test", cls, img)
            )

        print(f"✔ {cls}: {n_train} train, {n_val} val, {n_test} test")




# --------------------------------------
# MAIN
# --------------------------------------

make_dirs()
split_data()

print("\n🎉 Dataset successfully split into 70/15/15 train/val/test!")


# Define comprehensive data augmentation
print("🎨 Implementing Advanced Data Augmentation")
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # horizontal flip augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    
    # rotation augmentation  
    # HINT: Use transforms.RandomRotation(degrees=10) for ±10 degree rotation
    transforms.RandomRotation(degrees=10),   
    
    # translation augmentation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  #  
    
    # color jitter augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),   

    # Convert to tensor
    transforms.ToTensor(),   
    
    # random erasing augmentation 
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.25)),   
    
    
    # Normalize with Resnet 18 statistics
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    
])

# TODO: Create test transforms (no augmentation for consistent evaluation)
# HINT: Test set should only have ToTensor() and Normalize() - no augmentation!
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # TODO: Convert to tensor
    transforms.ToTensor(),   
    
    # TODO: Normalize with same statistics as training
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


try:
    # Create augmented datasets
    train_path = os.path.join(DEST_ROOT, "train")
    valid_path = os.path.join(DEST_ROOT, "val")
    test_path = os.path.join(DEST_ROOT, "test")
    
    train_ds = datasets.ImageFolder(root=train_path, transform=train_transform)

    valid_ds = datasets.ImageFolder(root=valid_path, transform=test_transform)

    test_ds = datasets.ImageFolder(root=test_path, transform=test_transform)
    
    # Test that transforms work
    sample_img, sample_label = train_ds[0]
    
    print(f"✅ SUCCESS! Augmented dataset created")
    print(f"✅ Training transforms: {len([t for t in train_transform.transforms if t is not None])} operations")
    print(f"✅ Test transforms: {len([t for t in test_transform.transforms if t is not None])} operations")
    print(f"✅ Sample image shape: {sample_img.shape}")
    print(f"✅ Sample image range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")


    print(f"Training samples: {len(train_ds):,}")
    print(f"Test samples: {len(test_ds):,}")
    print(f"Image shape: {train_ds[0][0].shape}")
    print(f"Classes: {CLASSES}")

    # Analyze class distribution
    train_labels = [train_ds[i][1] for i in range(len(train_ds))]
    class_counts = np.bincount(train_labels)

    print(f"\n📊 Class Distribution:")
    for i, (class_name, count) in enumerate(zip(CLASSES, class_counts)):
        print(f"   {i}: {class_name:13} - {count:,} samples")
        
    # Validation checks
    if sample_img.shape == (3, 224, 224):
        print("🎯 EXCELLENT! Sample has correct dimensions")
    else:
        print(f"❌ ERROR: Expected (3, 224, 224), got {sample_img.shape}")
        
except Exception as e:
    print(f"❌ Implementation incomplete: {e}")
    print("check path names")
