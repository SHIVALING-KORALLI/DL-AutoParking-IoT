import cv2
import os
from tqdm import tqdm
import albumentations as A

# Define individual augmentations (applied one by one)
transformations = {
    "horizontal_flip": A.HorizontalFlip(p=1),
    "vertical_flip": A.VerticalFlip(p=1),
    "rotation": A.Rotate(limit=15, p=1),
    "affine_scale_rotate_shear": A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), shear=(-10, 10), p=1),
    "hue_saturation_value": A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=30, p=1),
    "brightness_contrast": A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
    "gauss_noise": A.GaussNoise(var_limit=(20, 70), p=1),  # Increased noise variation
    "gaussian_blur": A.Sequential([
    A.GaussianBlur(blur_limit=(31, 51), p=1),
    A.GaussianBlur(blur_limit=(31, 51), p=1),
    A.GaussianBlur(blur_limit=(31, 51), p=1),
    A.GaussianBlur(blur_limit=(31, 51), p=1),
    A.GaussianBlur(blur_limit=(31, 51), p=1),
    A.GaussianBlur(blur_limit=(31, 51), p=1),], p=1),
    "random_shadow": A.RandomShadow(p=1),
    "clahe": A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
    "to_gray": A.ToGray(p=1),
    "posterize": A.Posterize(num_bits=2, p=1),  # Stronger effect
    "coarse_dropout": A.CoarseDropout(max_holes=8, max_height=0.1, max_width=0.1, p=1),
    "sharpen": A.Sharpen(alpha=(0.5, 0.8), lightness=(0.8, 1.2), p=1),
}

# Define paths
input_folder = r"C:\Users\OneDrive\Documents\Desktop\Major_Project_skk\detection_YOLO\paper_work\to_aug"
output_folder = r"C:\Users\OneDrive\Documents\Desktop\Major_Project_skk\detection_YOLO\paper_work\augmented"
os.makedirs(output_folder, exist_ok=True)

# Debug: Check input folder contents
print("Checking files in input folder:", os.listdir(input_folder))

# Get valid image files
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
print("Filtered image files:", image_files)

# Check if valid images exist
if not image_files:
    print("No valid image files found in input folder. Please check the folder path and file extensions.")
else:
    for img_name in tqdm(image_files, desc="Generating augmented images"):
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)

        # Skip unreadable images
        if image is None:
            print(f"Skipping file: {img_name}, unable to read or not a valid image.")
            continue

        for transformation_name, augmentation in transformations.items():
            augmented = augmentation(image=image)['image']
            aug_img_name = f"{os.path.splitext(img_name)[0]}_{transformation_name}.jpg"
            cv2.imwrite(os.path.join(output_folder, aug_img_name), augmented)

    print("Augmentation completed successfully!")
