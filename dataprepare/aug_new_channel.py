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
from natsort import natsorted
import albumentations as A
import os
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess and augment image dataset')
    parser.add_argument('--method', type=str, default='mild',
                       help='intensity of aug, {mild} or {strong} or mix')
    parser.add_argument('--num', type=int, default=5,
                       help='Number of augmentations per image (default: 5)')
    parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256],
                       help='Image size (height, width) (default: 256 256)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--output-dir', type=str, default='data/',
                       help='Output directory for processed data (default: data)')
    return parser.parse_args()

# feature 1: frequency
def extract_frequency_features(img: np.ndarray) -> np.ndarray:
    """Extract frequency domain features using FFT.
    
    Args:
        img: RGB image (H, W, 3)
        
    Returns:
        frequency features (H, W, 2) - magnitude and phase
    """
    # Convert to grayscale for frequency analysis
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    
    # Calculate magnitude spectrum and phase
    magnitude = np.log(np.abs(f_shift) + 1)  # Add 1 to avoid log(0)
    phase = np.angle(f_shift)
    
    # Normalize to 0-1 range
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    phase = (phase - phase.min()) / (phase.max() - phase.min())
    
    # Stack magnitude and phase as channels
    freq_features = np.stack([magnitude, phase], axis=-1)
    return freq_features

# feature 2: ndwi
def calculate_ndwi(img: np.ndarray) -> np.ndarray:
    """Calculate Normalized Difference Water Index.
    
    NDWI = (Green - NIR) / (Green + NIR)
    Since we don't have NIR band, we'll use Red channel as approximate NIR
    
    Args:
        img: RGB image (H, W, 3)
        
    Returns:
        NDWI feature (H, W, 1)
    """
    # Extract green and red channels
    green = img[:, :, 1].astype(float)
    red = img[:, :, 0].astype(float)  # Using red as NIR approximation
    
    # Calculate NDWI
    numerator = green - red
    denominator = green + red
    
    # Avoid division by zero
    ndwi = np.zeros_like(green)
    mask = denominator != 0
    ndwi[mask] = numerator[mask] / denominator[mask]
    
    # Normalize to 0-1 range
    ndwi = (ndwi - ndwi.min()) / (ndwi.max() - ndwi.min())
    
    return ndwi[..., np.newaxis]

# def calculate_texture_features(img: np.ndarray) -> np.ndarray:
#     """Calculate texture features using Gabor filters."""
#     # Ensure image is in uint8 format
#     if img.dtype != np.uint8:
#         img = (img * 255).astype(np.uint8)
        
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
#     orientations = [0, 45, 90, 135]
#     texture_features = []
    
#     for theta in orientations:
#         kernel = cv2.getGaborKernel(
#             ksize=(21, 21),
#             sigma=5.0,
#             theta=theta * np.pi / 180.0,
#             lambda_=10.0,
#             gamma=0.5,
#             psi=0,
#         )
#         filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
#         filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min() + 1e-8)
#         texture_features.append(filtered)
    
#     return np.stack(texture_features, axis=-1)


def extract_features(img: np.ndarray) -> np.ndarray:
    """Extract all features from an image."""
    # Normalize RGB to 0-1 range
    img_normalized = img.astype(np.float64) / 255.0
    
    # Extract additional features
    freq_features = extract_frequency_features(img)
    ndwi_feature = calculate_ndwi(img)
    # texture_features = calculate_texture_features(img)
    
    # Concatenate all features
    all_features = np.concatenate([
        img_normalized,    # RGB (3 channels)
        freq_features,     # Frequency domain (2 channels)
        ndwi_feature,      # NDWI (1 channel)
        # texture_features   # Texture (4 channels)
    ], axis=-1)
    
    return all_features

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

def augment_image(image: np.ndarray, mask: np.ndarray, transform: A.Compose, 
                 num_augment: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Augment image and mask, then extract features."""
    augmented_images = [image]
    augmented_masks = [mask]
    
    for _ in range(num_augment):
        augmented = transform(image=image, mask=mask)
        augmented_images.append(augmented['image'])
        augmented_masks.append(augmented['mask'])
    
    # Extract features for all augmented images
    augmented_features = [extract_features(img) for img in augmented_images]
    
    return augmented_features, augmented_masks

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
                   size: Tuple[int, int], transform: A.Compose = None, 
                   num_augment: int = 0, is_training: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Process dataset with feature extraction after augmentation."""
    num_images = len(image_list)
    height, width = size
    num_channels = 6  # RGB(3) + Freq(2) + NDWI(1) +  # Texture(4)
    augmented_size = num_images * (num_augment + 1) if is_training else num_images
    
    data = np.zeros([augmented_size, height, width, num_channels])
    labels = np.zeros([augmented_size, height, width])
    
    idx = 0
    desc = "Processing training set" if is_training else "Processing validation/test set"
    for _, (img_path, mask_path) in tqdm(enumerate(zip(image_list, mask_list)), 
                                        total=num_images, desc=desc):
        
        # Load original image and mask
        img = load_and_preprocess_image(img_path, (width, height))
        mask = load_and_preprocess_mask(mask_path, (width, height))
        
        if is_training and num_augment > 0:
            # First augment the original image
            aug_images, aug_masks = augment_image(img, mask, transform, num_augment)
            
            # Then extract features for all augmented images
            for aug_img, aug_mask in zip(aug_images, aug_masks):
                data[idx] = aug_img  # aug_img already contains all features
                labels[idx] = aug_mask
                idx += 1
        else:
            # For validation/test, just extract features without augmentation
            features = extract_features(img)
            data[idx] = features
            labels[idx] = mask
            idx += 1
    
    return data, labels

def get_augmentation_params(mask_ratio: float) -> Tuple[int, A.Compose]:
    if mask_ratio >= 0.4:
        num_aug = 3
        transform = create_augmentations_strong()
    elif 0.2 <= mask_ratio < 0.4:
        num_aug = 7
        transform = create_augmentations_mild()
    else:
        num_aug = 5
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        ])
        
    return num_aug, transform

def process_dataset_mix(image_list: List[str], mask_list: List[str], 
                       size: Tuple[int, int], is_training: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Process dataset with mixed augmentation strategy."""
    num_images = len(image_list)
    height, width = size
    
    processed_images = []
    processed_masks = []
    
    desc = "Processing training set" if is_training else "Processing validation/test set"
    for _, (img_path, mask_path) in tqdm(enumerate(zip(image_list, mask_list)), 
                                        total=num_images, desc=desc):
        
        # Load original image and mask
        img = load_and_preprocess_image(img_path, (width, height))
        mask = load_and_preprocess_mask(mask_path, (width, height))
        
        if is_training:
            # First determine augmentation strategy
            mask_ratio = np.mean(mask) / 255
            num_augment, transform = get_augmentation_params(mask_ratio)
            print(num_augment, end=' ')
            
            # Then augment and extract features
            aug_images, aug_masks = augment_image(img, mask, transform, num_augment)
            
            # Add to processed lists
            for aug_img, aug_mask in zip(aug_images, aug_masks):
                processed_images.append(aug_img)
                processed_masks.append(aug_mask)
        else:
            # For validation/test, just extract features
            features = extract_features(img)
            processed_images.append(features)
            processed_masks.append(mask)
    
    return np.array(processed_images), np.array(processed_masks)

def main():
    """Main function."""
    args = parse_arguments()
    height, width = args.image_size
    
    print("\n=== Processing Parameters ===")
    print(f'Method: {args.method}')
    print(f"Image size: {width}x{height}")
    print(f"Augmentations per image: {args.num}")
    print(f"Train/Val split ratio: {args.train_ratio:.2f}/{1-args.train_ratio:.2f}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.output_dir = os.path.join(script_dir, args.output_dir)
    print(f"Dataset path: {script_dir}/training & {script_dir}/testing")
    print(f"Output directory: {args.output_dir}")
    
    # Prepare data paths
    tr_list = natsorted(glob.glob(f"{script_dir}/training/image/*.png"))
    ms_list = natsorted(glob.glob(f"{script_dir}/training/mask/*.png"))
    test_list = natsorted(glob.glob(f"{script_dir}/testing/image/*.[jJ][pP][gG]"))
    test_mask_list = natsorted(glob.glob(f"{script_dir}/testing/mask/*.png"))
    
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
    # Process datasets
    if args.method == 'mild' or args.method == 'strong':
        transform = create_augmentations_mild()
        transform = create_augmentations_strong()
        print(f'\nProcessing training set with {args.num} augmentations per image...')
        data_train, mask_train = process_dataset(
            train_images, train_masks, 
            (height, width), transform, 
            args.num, is_training=True
        )

    elif args.method == 'mix':
        print(f'\nProcessing training set with mix augmentations...')
        data_train, mask_train = process_dataset_mix(
            train_images, train_masks, 
            (height, width), is_training=True
        )

    print('\nProcessing validation set...')
    data_val, mask_val = process_dataset(
        val_images, val_masks, 
        (height, width)
    )
    
    print('\nProcessing test set...')
    data_test, mask_test = process_dataset(
        test_list, test_mask_list, 
        (height, width)
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
