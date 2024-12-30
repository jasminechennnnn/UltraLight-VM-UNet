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
    parser.add_argument('--method', type=str, default='mild',
                       help='intensity of aug, {mild} or {strong}')
    parser.add_argument('--num', type=int, default=5,
                       help='Number of augmentations per image (default: 5)')
    parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256],
                       help='Image size (height, width) (default: 256 256)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--output-dir', type=str, default='data/',
                       help='Output directory for processed data (default: data)')
    return parser.parse_args()


#### version 1: mild
def create_augmentations_mild() -> A.Compose:
    """
    Create mild augmentations for water segmentation.
    Focuses on essential transforms with lower intensity.
    """
    transform = A.Compose([
        # 基礎裁切和縮放 - 降低裁切比例
        A.RandomResizedCrop(
            height=256,
            width=256,
            scale=(0.75, 1.0),     # 改為只裁切25%
            ratio=(0.85, 1.15),    # 更小的寬高比變化
            p=0.5                   # 降低裁切概率
        ),
        
        # 基礎翻轉 - 保留但降低概率
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.3),
        
        # 水波紋模擬 - 降低變形程度
        A.OneOf([
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.15,  # 從0.3降低到0.15
                p=1.0
            ),
            A.ElasticTransform(
                alpha=50,            # 從150降低到50
                sigma=5,             # 從10降低到5
                alpha_affine=5,      # 從10降低到5
                p=1.0
            ),
        ], p=0.2),                  # 從0.4降低到0.2
        
        # 光照變化 - 降低變化範圍
        A.RandomBrightnessContrast(
            brightness_limit=0.1,    # 從0.3降低到0.1
            contrast_limit=0.1,      # 從0.3降低到0.1
            p=0.3                    # 從0.5降低到0.3
        ),
        
        # 輕微的色彩調整
        A.HueSaturationValue(
            hue_shift_limit=5,       # 從10降低到5
            sat_shift_limit=10,      # 從20降低到10
            val_shift_limit=10,      # 從20降低到10
            p=0.2                    # 從0.4降低到0.2
        ),
    ])
    return transform


#### version 2: strong
def create_augmentations_strong() -> A.Compose:
    """
    Create specialized augmentations for water segmentation.
    Focus on preserving water features while increasing variation.
    """
    transform = A.Compose([
        
        A.RandomResizedCrop(
            height=256,  # 最終輸出大小
            width=256,
            scale=(0.5, 1.0),     # 裁切大小範圍：原圖50%~100%
            ratio=(0.75, 1.33),   # 寬高比範圍
            p=0.7
        ),
        
        # 1. 基礎翻轉
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),  # 水面的水平翻轉不會改變其本質特徵
        A.VerticalFlip(p=0.2),    # 部分水體場景（如河流）垂直翻轉也合理
        
        # 2. 水波紋模擬
        A.OneOf([
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=1.0
            ),  # 模擬水面波紋效果
            A.ElasticTransform(
                alpha=150,         # 較大的形變幅度
                sigma=10,          # 控制形變的平滑度
                alpha_affine=10,   # 整體形變程度
                p=1.0
            ),  # 模擬流動的水面
        ], p=0.4),
        
        # 3. 光照變化（水面反光效果）
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),  # 模擬陽光直射
            A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(4, 4),
                p=1.0
            ),  # 增強局部對比度，突出水面紋理
        ], p=0.5),
        
        # 4. 水體顏色變化
        A.OneOf([
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=1.0
            ),  # 模擬不同時間、天氣下的水體顏色
            A.HueSaturationValue(
                hue_shift_limit=10,  # 輕微的色調變化
                sat_shift_limit=20,  # 適度的飽和度變化
                val_shift_limit=20,  # 適度的明度變化
                p=1.0
            ),  # 模擬不同深度的水體
        ], p=0.4),
        
        # 5. 水質變化模擬
        A.OneOf([
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=1.0
            ),  # 模擬渾濁水體
            A.GaussNoise(
                var_limit=(10.0, 30.0),
                mean=0,
                per_channel=True,
                p=1.0
            ),  # 模擬水面雜質
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
    print(f"Augmentations per image: {args.num}")
    print(f"Train/Val split ratio: {args.train_ratio:.2f}/{1-args.train_ratio:.2f}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.output_dir = os.path.join(script_dir, args.output_dir)
    print(f"Dataset path: {script_dir}/training & {script_dir}/testing")
    print(f"Output directory: {args.output_dir}")
    
    # Prepare data paths
    tr_list = sorted(glob.glob(f"{script_dir}/training/image/*.png"))
    ms_list = sorted(glob.glob(f"{script_dir}/training/mask/*.png"))
    test_list = sorted(glob.glob(f"{script_dir}/testing/image/*.[jJ][pP][gG]"))
    test_mask_list = sorted(glob.glob(f"{script_dir}/testing/mask/*.png"))
    
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
    print(f"Expected augmented training images: {len(train_images) * (args.num + 1)}")

    # Create augmentation transform
    if args.method == 'mild':
        transform = create_augmentations_mild()
    elif args.method == 'strong':
        transform = create_augmentations_strong()
    
    # Process datasets
    print(f'\nProcessing training set with {args.num} augmentations per image...')
    data_train, mask_train = process_dataset(
        train_images, train_masks, 
        (height, width), transform, 
        args.num, is_training=True
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


# ##### 用 training_dataset + testing_dataset #####
# ##### refactored
# import argparse
# import h5py
# import numpy as np
# import glob
# import cv2
# from scipy import ndimage
# import random
# from PIL import Image
# import albumentations as A
# import os
# from typing import List, Tuple, Dict
# from pathlib import Path
# from tqdm import tqdm

# def parse_arguments() -> argparse.Namespace:
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description='Preprocess and augment image dataset')
#     parser.add_argument('--method', type=str, default='mild',
#                        help='intensity of aug, {mild} or {strong}')
#     parser.add_argument('--num', type=int, default=5,
#                        help='Number of augmentations per image (default: 5)')
#     parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256],
#                        help='Image size (height, width) (default: 256 256)')
#     parser.add_argument('--train-ratio', type=float, default=0.8,
#                        help='Training set ratio (default: 0.8)')
#     parser.add_argument('--output-dir', type=str, default='data/',
#                        help='Output directory for processed data (default: data)')
#     return parser.parse_args()

# def calculate_water_ratio(mask: np.ndarray) -> float:
#     """Calculate the ratio of water pixels in the mask."""
#     return np.sum(mask == 255) / mask.size

# def get_augmentation_params(water_ratio: float) -> Tuple[int, str]:
#     """
#     Determine augmentation parameters based on water ratio.
#     Returns number of augmentations and augmentation intensity.
#     """
#     if water_ratio < 0.3:
#         return 8, 'mild'  # 較少水體的圖片做基本擴增
#     elif 0.3 <= water_ratio < 0.5:
#         return 4, 'mild'  # 中等水體做更多基本擴增
#     else:
#         return 2, 'strong'  # 較多水體的圖片做強度更大的擴增

# def create_adaptive_augmentations(water_ratio: float) -> A.Compose:
#     """Create augmentation pipeline based on water ratio."""
#     # 基礎變換 - 所有圖片都會使用
#     base_transforms = [
#         A.RandomRotate90(p=0.5),
#         A.HorizontalFlip(p=0.5),
#     ]
    
#     # 根據水體比例添加額外的變換
#     if water_ratio < 0.2:
#         # 少量水體：主要進行基礎擴增和輕微的光照調整
#         additional_transforms = [
#             A.RandomBrightnessContrast(
#                 brightness_limit=0.1,
#                 contrast_limit=0.1,
#                 p=0.3
#             ),
#             A.HueSaturationValue(
#                 hue_shift_limit=5,
#                 sat_shift_limit=10,
#                 val_shift_limit=10,
#                 p=0.2
#             ),
#         ]
#     elif 0.2 <= water_ratio < 0.4:
#         # 中等水體：加入更多的變換
#         additional_transforms = [
#             A.RandomBrightnessContrast(
#                 brightness_limit=0.2,
#                 contrast_limit=0.2,
#                 p=0.4
#             ),
#             A.GridDistortion(
#                 num_steps=5,
#                 distort_limit=0.2,
#                 p=0.3
#             ),
#             A.HueSaturationValue(
#                 hue_shift_limit=10,
#                 sat_shift_limit=15,
#                 val_shift_limit=15,
#                 p=0.3
#             ),
#         ]
#     else:
#         # 大量水體：使用強力擴增
#         additional_transforms = [
#             A.RandomBrightnessContrast(
#                 brightness_limit=0.3,
#                 contrast_limit=0.3,
#                 p=0.5
#             ),
#             A.GridDistortion(
#                 num_steps=5,
#                 distort_limit=0.3,
#                 p=0.4
#             ),
#             A.ElasticTransform(
#                 alpha=150,
#                 sigma=10,
#                 alpha_affine=10,
#                 p=0.4
#             ),
#             A.HueSaturationValue(
#                 hue_shift_limit=15,
#                 sat_shift_limit=20,
#                 val_shift_limit=20,
#                 p=0.4
#             ),
#             A.GaussianBlur(
#                 blur_limit=(3, 5),
#                 p=0.3
#             ),
#         ]
    
#     # 合併所有變換
#     transforms = base_transforms + additional_transforms
    
#     return A.Compose(transforms)

# def augment_image(image: np.ndarray, mask: np.ndarray, num_augment: int, water_ratio: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
#     """Augment a single image and its mask based on water ratio."""
#     transform = create_adaptive_augmentations(water_ratio)
#     augmented_images = [image]
#     augmented_masks = [mask]
    
#     for _ in range(num_augment):
#         augmented = transform(image=image, mask=mask)
#         augmented_images.append(augmented['image'])
#         augmented_masks.append(augmented['mask'])
    
#     return augmented_images, augmented_masks

# def load_and_preprocess_image(image_path: str, size: Tuple[int, int]) -> np.ndarray:
#     """Load and preprocess a single image."""
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, size)
#     # return img.astype(np.float64)
#     return img

# def load_and_preprocess_mask(mask_path: str, size: Tuple[int, int]) -> np.ndarray:
#     """Load and preprocess a single mask."""
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
#     # return mask.astype(np.float64)
#     return mask

# def process_dataset(image_list: List[str], mask_list: List[str], 
#                    size: Tuple[int, int], is_training: bool = False) -> Tuple[np.ndarray, np.ndarray]:
#     """Process a dataset with adaptive augmentation strategy."""
#     height, width = size
#     processed_images = []
#     processed_masks = []
    
#     desc = "Processing training set" if is_training else "Processing validation/test set"
#     for img_path, mask_path in tqdm(zip(image_list, mask_list), 
#                                   total=len(image_list), desc=desc):
#         # 載入原始圖片和遮罩
#         img = load_and_preprocess_image(img_path, (width, height))
#         mask = load_and_preprocess_mask(mask_path, (width, height))
        
#         if is_training:
#             # 計算水體比例並決定擴增策略
#             water_ratio = calculate_water_ratio(mask)
#             num_augment, _ = get_augmentation_params(water_ratio)
            
#             # 進行擴增
#             aug_images, aug_masks = augment_image(img, mask, num_augment, water_ratio)
#             processed_images.extend(aug_images)
#             processed_masks.extend(aug_masks)
#         else:
#             processed_images.append(img)
#             processed_masks.append(mask)
    
#     # 轉換為 numpy array
#     return (np.array(processed_images, dtype=np.float64), 
#             np.array(processed_masks, dtype=np.float64))

# def main():
#     """Main function."""
#     args = parse_arguments()
#     height, width = args.image_size
    
#     print("\n=== Processing Parameters ===")
#     print(f"Image size: {width}x{height}")
#     print(f"Augmentations per image: {args.num}")
#     print(f"Train/Val split ratio: {args.train_ratio:.2f}/{1-args.train_ratio:.2f}")
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     args.output_dir = os.path.join(script_dir, args.output_dir)
#     print(f"Dataset path: {script_dir}/training & {script_dir}/testing")
#     print(f"Output directory: {args.output_dir}")
    
#     # Prepare data paths
#     tr_list = sorted(glob.glob(f"{script_dir}/training/image/*.png"))
#     ms_list = sorted(glob.glob(f"{script_dir}/training/mask/*.png"))
#     test_list = sorted(glob.glob(f"{script_dir}/testing/image/*.[jJ][pP][gG]"))
#     test_mask_list = sorted(glob.glob(f"{script_dir}/testing/mask/*.png"))
    
#     print("\n=== Dataset Statistics ===")
#     print(f"Total training images found: {len(tr_list)}")
#     print(f"Total test images found: {len(test_list)}")

#     # Split training and validation
#     train_number = int(len(tr_list) * args.train_ratio)
#     train_images = tr_list[:train_number]
#     train_masks = ms_list[:train_number]
#     val_images = tr_list[train_number:]
#     val_masks = ms_list[train_number:]
    
#     print("\n=== Split Information ===")
#     print(f"Training images: {len(train_images)}")
#     print(f"Validation images: {len(val_images)}")
#     print(f"Expected augmented training images: {len(train_images) * (args.num + 1)}")

#     # Process datasets with adaptive augmentation
#     print('\nProcessing training set with adaptive augmentation...')
#     data_train, mask_train = process_dataset(
#         train_images, train_masks, 
#         (height, width), is_training=True
#     )
    
#     print('\nProcessing validation set...')
#     data_val, mask_val = process_dataset(
#         val_images, val_masks, 
#         (height, width), is_training=False
#     )
    
#     print('\nProcessing test set...')
#     data_test, mask_test = process_dataset(
#         test_list, test_mask_list, 
#         (height, width), is_training=False
#     )
    
#     # Save results
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(exist_ok=True)
    
#     print(f'\nSaving processed data to {output_dir}...')
#     np.save(output_dir / 'data_train.npy', data_train)
#     np.save(output_dir / 'data_val.npy', data_val)
#     np.save(output_dir / 'data_test.npy', data_test)
#     np.save(output_dir / 'mask_train.npy', mask_train)
#     np.save(output_dir / 'mask_val.npy', mask_val)
#     np.save(output_dir / 'mask_test.npy', mask_test)
    
#     print(f'\n=== Finished processing dataset ===')
#     print(f'Training samples: {len(data_train)}')
#     print(f'Validation samples: {len(data_val)}')
#     print(f'Test samples: {len(data_test)}')

# if __name__ == '__main__':
#     main()
