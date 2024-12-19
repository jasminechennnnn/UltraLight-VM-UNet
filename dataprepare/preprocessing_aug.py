##### 用 training_dataset + testing_dataset #####
##### refactored
import argparse
import h5py
import numpy as np
import glob
import cv2
from scipy import ndimage
import random
from PIL import Image
import albumentations as A
import os
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess and augment image dataset')
    parser.add_argument('--num-augment', type=int, default=5,
                       help='Number of augmentations per image (default: 5)')
    parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256],
                       help='Image size (height, width) (default: 256 256)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--output-dir', type=str, default='data/',
                       help='Output directory for processed data (default: data)')
    return parser.parse_args()

def create_augmentations() -> A.Compose:
    """Create augmentation pipeline."""
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
            A.RandomGamma(p=1),
        ], p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=1),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
        ], p=0.4),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.OneOf([
            A.Sharpen(p=1),
            A.MotionBlur(p=1),
            A.MedianBlur(blur_limit=3, p=1)
        ], p=0.3),
    ])
    return transform

def augment_image(image: np.ndarray, mask: np.ndarray, transform: A.Compose, num_augment: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Augment a single image and its mask."""
    augmented_images = [image]
    augmented_masks = [mask]
    
    for _ in range(num_augment):
        augmented = transform(image=image, mask=mask)
        augmented_images.append(augmented['image'])
        augmented_masks.append(augmented['mask'])
    
    return augmented_images, augmented_masks

def load_and_preprocess_image(image_path: str, size: Tuple[int, int]) -> np.ndarray:
    """Load and preprocess a single image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    # return img.astype(np.float64)
    return img

def load_and_preprocess_mask(mask_path: str, size: Tuple[int, int]) -> np.ndarray:
    """Load and preprocess a single mask."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    # return mask.astype(np.float64)
    return mask

def process_dataset(image_list: List[str], mask_list: List[str], 
                   size: Tuple[int, int], transform: A.Compose, 
                   num_augment: int = 0, is_training: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Process a dataset (training/validation/testing)."""
    num_images = len(image_list)
    height, width = size
    augmented_size = num_images * (num_augment + 1) if is_training else num_images
    
    data = np.zeros([augmented_size, height, width, 3])
    labels = np.zeros([augmented_size, height, width])
    
    idx = 0
    desc = "Processing training set" if is_training else "Processing validation/test set"
    for _, (img_path, mask_path) in tqdm(enumerate(zip(image_list, mask_list)), 
                                        total=num_images, desc=desc):
        
        img = load_and_preprocess_image(img_path, (width, height))
        mask = load_and_preprocess_mask(mask_path, (width, height))
        
        if is_training and num_augment > 0:
            aug_images, aug_masks = augment_image(img, mask, transform, num_augment)
            for aug_img, aug_mask in zip(aug_images, aug_masks):
                data[idx] = aug_img.astype(np.float64)
                labels[idx] = aug_mask.astype(np.float64)
                idx += 1
        else:
            data[idx] = img.astype(np.float64)
            labels[idx] = mask.astype(np.float64)
            idx += 1
    
    return data, labels

def main():
    """Main function."""
    args = parse_arguments()
    height, width = args.image_size
    
    print("\n=== Processing Parameters ===")
    print(f"Image size: {width}x{height}")
    print(f"Augmentations per image: {args.num_augment}")
    print(f"Train/Val split ratio: {args.train_ratio:.2f}/{1-args.train_ratio:.2f}")
    print(f"Output directory: {args.output_dir}")

    # Prepare data paths
    tr_list = sorted(glob.glob("training/image/*.png"))
    ms_list = sorted(glob.glob("training/mask/*.png"))
    test_list = sorted(glob.glob("testing/image/*.[jJ][pP][gG]"))
    test_mask_list = sorted(glob.glob("testing/mask/*.png"))
    
    print("\n=== Dataset Statistics ===")
    print(f"Total training images found: {len(tr_list)}")
    print(f"Total test images found: {len(test_list)}")

    # Split training and validation
    train_number = int(len(tr_list) * args.train_ratio)
    train_images = tr_list[:train_number]
    train_masks = ms_list[:train_number]
    val_images = tr_list[train_number:]
    val_masks = ms_list[train_number:]
    
    print("\n=== Split Information ===")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Expected augmented training images: {len(train_images) * (args.num_augment + 1)}")

    # Create augmentation transform
    transform = create_augmentations()
    
    # Process datasets
    print(f'\nProcessing training set with {args.num_augment} augmentations per image...')
    data_train, mask_train = process_dataset(
        train_images, train_masks, 
        (height, width), transform, 
        args.num_augment, is_training=True
    )
    
    print('\nProcessing validation set...')
    data_val, mask_val = process_dataset(
        val_images, val_masks, 
        (height, width), transform
    )
    
    print('\nProcessing test set...')
    data_test, mask_test = process_dataset(
        test_list, test_mask_list, 
        (height, width), transform
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f'\nSaving processed data to {output_dir}...')
    np.save(output_dir / 'data_train.npy', data_train)
    np.save(output_dir / 'data_val.npy', data_val)
    np.save(output_dir / 'data_test.npy', data_test)
    np.save(output_dir / 'mask_train.npy', mask_train)
    np.save(output_dir / 'mask_val.npy', mask_val)
    np.save(output_dir / 'mask_test.npy', mask_test)
    
    print(f'\n=== Finished processing dataset ===')
    print(f'Training samples: {len(data_train)}')
    print(f'Validation samples: {len(data_val)}')
    print(f'Test samples: {len(data_test)}')

if __name__ == '__main__':
    main()


##### 用 training_dataset + testing_dataset #####
# import h5py
# import numpy as np
# import glob
# import cv2
# from scipy import ndimage
# import random
# from PIL import Image
# import albumentations as A
# import os


# def create_augmentations():

#     transform = A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.OneOf([
#             # Use ColorJitter to replace "RandomBrightness"
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
#             A.RandomGamma(p=1),
#         ], p=0.5),
#         A.OneOf([
#             A.Blur(blur_limit=3, p=1),
#             A.GaussNoise(var_limit=(10.0, 50.0), p=1),
#         ], p=0.4),
#         # 可以添加一些特別針對水體的增強
#         A.RandomBrightnessContrast(
#             brightness_limit=0.2,
#             contrast_limit=0.2,
#             p=0.5
#         ),
#         # 添加一些可能有助於水體識別的轉換
#         A.OneOf([
#             A.Sharpen(p=1),
#             A.MotionBlur(p=1),
#             A.MedianBlur(blur_limit=3, p=1)
#         ], p=0.3),
#     ])
#     return transform

# def augment_image(image, mask, transform, num_augment=5):
#     augmented_images = [image]
#     augmented_masks = [mask]
    
#     for _ in range(num_augment):
#         augmented = transform(image=image, mask=mask)
#         augmented_images.append(augmented['image'])
#         augmented_masks.append(augmented['mask'])
    
#     return augmented_images, augmented_masks

# ########## Parameters ##########
# height = 256
# width = 256
# channels = 3
# num_augment = 5  # 每張圖片增強的次數

# # Prepare your data set
# Tr_list = glob.glob("training/image/*.png")
# Ms_list = glob.glob("training/mask/*.png") 
# Test_list = glob.glob("testing/image/*.[jJ][pP][gG]")
# Test_mask_list = glob.glob("testing/mask/*.png")
# train_number = int(len(Tr_list) * 0.8)
# val_number = int(len(Tr_list) * 0.2)
# test_number = len(Test_list)

# print("Test_list", Test_list)

# # Calculate size and init. np.array
# augmented_train_size = train_number * (num_augment + 1)  # 原圖 + 增強圖
# augmented_val_size = val_number
# augmented_test_size = test_number
# Data_train_aug = np.zeros([augmented_train_size, height, width, channels])
# Label_train_aug = np.zeros([augmented_train_size, height, width])
# Data_val = np.zeros([augmented_val_size, height, width, channels])
# Label_val = np.zeros([augmented_val_size, height, width])
# Data_test = np.zeros([augmented_test_size, height, width, channels])
# Label_test = np.zeros([augmented_test_size, height, width])

# transform = create_augmentations()

# print(f'Start reading and augmenting {len(Tr_list)} images...')
# aug_idx = 0

# # Training set
# for idx in range(train_number):
#     print(f'Processing training image {idx+1}/{train_number}')

#     img = cv2.imread(Tr_list[idx])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (width, height))
    
#     mask = cv2.imread(Ms_list[idx], cv2.IMREAD_GRAYSCALE)
#     mask = cv2.resize(mask, (width, height))
    
#     aug_images, aug_masks = augment_image(img, mask, transform, num_augment)
    
#     for i, (aug_img, aug_mask) in enumerate(zip(aug_images, aug_masks)):
#         Data_train_aug[aug_idx] = aug_img.astype(np.float64)
#         Label_train_aug[aug_idx] = aug_mask.astype(np.float64)
#         aug_idx += 1

# print("Items in Data_train_aug:", aug_idx)

# # Validation set
# for idx in range(val_number):
#     val_idx = train_number + idx
#     print(f'Processing validation image {idx+1}/{val_number}')
    
#     img = cv2.imread(Tr_list[val_idx])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (width, height))
#     Data_val[idx] = img.astype(np.float64)
    
#     mask = cv2.imread(Ms_list[val_idx], cv2.IMREAD_GRAYSCALE)
#     mask = cv2.resize(mask, (width, height))
#     Label_val[idx] = mask.astype(np.float64)

# # Testing set
# for idx in range(test_number):
#     print(f'Processing test image {idx+1}/{test_number}')
    
#     img = cv2.imread(Test_list[idx])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (width, height))
#     Data_test[idx] = img.astype(np.float64)
    
#     mask = cv2.imread(Test_mask_list[idx], cv2.IMREAD_GRAYSCALE)
#     mask = cv2.resize(mask, (width, height))
#     Label_test[idx] = mask.astype(np.float64)

# print('\nAugmentation finished.')

# if not os.path.exists("data/"):
#     os.makedirs("data/")
# np.save('data/data_train', Data_train_aug)
# np.save('data/data_val', Data_val)
# np.save('data/data_test', Data_test)
# np.save('data/mask_train', Label_train_aug)
# np.save('data/mask_val', Label_val)
# np.save('data/mask_test', Label_test)

# print(f'Saved augmented dataset with {augmented_train_size} training samples.')


##### 僅用 training_dataset 切出 train/val/test set #####
# import h5py
# import numpy as np
# import glob
# import cv2
# from scipy import ndimage
# import random
# from PIL import Image
# import albumentations as A

# def create_augmentations():

#     transform = A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.OneOf([
#             # Use ColorJitter to replace "RandomBrightness"
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
#             A.RandomGamma(p=1),
#         ], p=0.5),
#         A.OneOf([
#             A.Blur(blur_limit=3, p=1),
#             A.GaussNoise(var_limit=(10.0, 50.0), p=1),
#         ], p=0.4),
#         # 可以添加一些特別針對水體的增強
#         A.RandomBrightnessContrast(
#             brightness_limit=0.2,
#             contrast_limit=0.2,
#             p=0.5
#         ),
#         # 添加一些可能有助於水體識別的轉換
#         A.OneOf([
#             A.Sharpen(p=1),
#             A.MotionBlur(p=1),
#             A.MedianBlur(blur_limit=3, p=1)
#         ], p=0.3),
#     ])
#     return transform

# def augment_image(image, mask, transform, num_augment=5):
#     augmented_images = [image]
#     augmented_masks = [mask]
    
#     for _ in range(num_augment):
#         augmented = transform(image=image, mask=mask)
#         augmented_images.append(augmented['image'])
#         augmented_masks.append(augmented['mask'])
    
#     return augmented_images, augmented_masks

# ########## Parameters ##########
# height = 256
# width = 256
# channels = 3
# num_augment = 5  # 每張圖片增強的次數

# # Prepare your data set
# Tr_list = glob.glob("training/image/*.png")
# Ms_list = glob.glob("training/mask/*.png")
# train_number = int(len(Tr_list) * 0.7)
# val_number = int(len(Tr_list) * 0.2)
# test_number = int(len(Tr_list) * 0.1)
# all = len(Tr_list)

# # 計算增強後的總數據量
# augmented_train_size = train_number * (num_augment + 1)  # 原圖 + 增強圖
# augmented_val_size = val_number
# # augmented_test_size = all
# augmented_test_size = test_number

# # 初始化數組以存儲增強後的數據
# Data_train_aug = np.zeros([augmented_train_size, height, width, channels])
# Label_train_aug = np.zeros([augmented_train_size, height, width])
# Data_val = np.zeros([augmented_val_size, height, width, channels])
# Label_val = np.zeros([augmented_val_size, height, width])
# Data_test = np.zeros([augmented_test_size, height, width, channels])
# Label_test = np.zeros([augmented_test_size, height, width])

# transform = create_augmentations()

# print(f'Start reading and augmenting {len(Tr_list)} images...')
# aug_idx = 0

# # Training set
# for idx in range(train_number):
#     print(f'Processing training image {idx+1}/{train_number}')

#     img = cv2.imread(Tr_list[idx])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (width, height))
    
#     mask = cv2.imread(Ms_list[idx], cv2.IMREAD_GRAYSCALE)
#     mask = cv2.resize(mask, (width, height))
    
#     aug_images, aug_masks = augment_image(img, mask, transform, num_augment)
    
#     for i, (aug_img, aug_mask) in enumerate(zip(aug_images, aug_masks)):
#         Data_train_aug[aug_idx] = aug_img.astype(np.float64)
#         Label_train_aug[aug_idx] = aug_mask.astype(np.float64)
#         aug_idx += 1

# # Validation set
# for idx in range(val_number):
#     val_idx = train_number + idx
#     print(f'Processing validation image {idx+1}/{val_number}')
    
#     img = cv2.imread(Tr_list[val_idx])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (width, height))
#     Data_val[idx] = img.astype(np.float64)
    
#     mask = cv2.imread(Ms_list[val_idx], cv2.IMREAD_GRAYSCALE)
#     mask = cv2.resize(mask, (width, height))
#     Label_val[idx] = mask.astype(np.float64)

# # Testing set
# for idx in range(test_number):
#     test_idx = train_number + val_number + idx
#     print(f'Processing test image {idx+1}/{test_number}')
    
#     img = cv2.imread(Tr_list[test_idx])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (width, height))
#     Data_test[idx] = img.astype(np.float64)
    
#     mask = cv2.imread(Ms_list[idx], cv2.IMREAD_GRAYSCALE)
#     mask = cv2.resize(mask, (width, height))
#     Label_test[idx] = mask.astype(np.float64)
# # for idx in range(all):
# #     print(f'Processing test image {idx+1}/{all}')
    
# #     img = cv2.imread(Tr_list[idx])
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     img = cv2.resize(img, (width, height))
# #     Data_test[idx] = img.astype(np.float64)
    
# #     mask = cv2.imread(Ms_list[idx], cv2.IMREAD_GRAYSCALE)
# #     mask = cv2.resize(mask, (width, height))
# #     Label_test[idx] = mask.astype(np.float64)


# print('\nAugmentation finished.')

# np.save('data_train', Data_train_aug)
# np.save('data_val', Data_val)
# np.save('data_test', Data_test)
# np.save('mask_train', Label_train_aug)
# np.save('mask_val', Label_val)
# np.save('mask_test', Label_test)

# print(f'Saved augmented dataset with {augmented_train_size} training samples.')