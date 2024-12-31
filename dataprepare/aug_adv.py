import argparse
import numpy as np
import glob
import cv2
import random
# from PIL import Image
import albumentations as A
import os
from typing import List, Tuple, Dict
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
from scipy.spatial.distance import cosine

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Preprocess and augment image dataset')
    parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256],
                       help='Image size (height, width) (default: 256 256)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--output-dir', type=str, default='data/',
                       help='Output directory for processed data (default: data)')
    parser.add_argument('--min-water-ratio', type=float, default=0.2,
                       help='Minimum water ratio threshold (default: 0.2)')
    parser.add_argument('--shuffle', type=int, default=0,
                       help='Shuffle before splitting?')
    parser.add_argument('--con-th', type=float, default=0.75,
                       help='Complexity threshold')
    return parser.parse_args()

def load_and_preprocess_image(image_path: str, size: Tuple[int, int]) -> np.ndarray:
    """Load and preprocess a single image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

def load_and_preprocess_mask(mask_path: str, size: Tuple[int, int]) -> np.ndarray:
    """Load and preprocess a single mask."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {mask_path}")
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return mask

def load_dataset(image_paths: List[str], mask_paths: List[str], size: Tuple[int, int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load and preprocess a list of images and masks."""
    images = []
    masks = []
    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), desc="Loading dataset"):
        try:
            img = load_and_preprocess_image(img_path, size)
            mask = load_and_preprocess_mask(mask_path, size)
            images.append(img)
            masks.append(mask)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    return images, masks

def analyze_image_complexity(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    分析圖片的複雜度
    
    Args:
        image: 輸入圖片 (H, W, C)
        mask: 遮罩 (H, W)
        
    Returns:
        包含各種複雜度指標的字典
    """
    metrics = {}
    
    # 1. 計算邊緣複雜度
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    metrics['edge_density'] = np.mean(edges > 0)
    
    # 2. 計算遮罩的邊緣複雜度
    mask_edges = cv2.Canny((mask > 127).astype(np.uint8) * 255, 100, 200)
    metrics['mask_edge_density'] = np.mean(mask_edges > 0)
    
    # 3. 計算水體區域的形狀複雜度
    water_region = mask > 127
    if np.any(water_region):
        # 計算水體區域的周長與面積比
        contours, _ = cv2.findContours(water_region.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        area = np.sum(water_region)
        perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
        metrics['shape_complexity'] = perimeter / (np.sqrt(area) + 1e-6)
    else:
        metrics['shape_complexity'] = 0
        
    # 4. 計算圖片的清晰度（使用方差作為度量）
    metrics['sharpness'] = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 5. 計算圖片的整體對比度
    metrics['contrast'] = np.std(gray)
    
    # 6. 計算雜訊水平（使用高斯差分）
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    noise = np.abs(gray.astype(np.float32) - blur.astype(np.float32))
    metrics['noise_level'] = np.mean(noise)
    
    return metrics

def filter_training_data(images: List[np.ndarray], 
                        masks: List[np.ndarray],
                        complexity_threshold: float = 0.75) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    根據複雜度過濾訓練資料
    
    Args:
        images: 訓練圖片列表
        masks: 訓練遮罩列表
        complexity_threshold: 複雜度閾值（百分比）
        
    Returns:
        過濾後的圖片和遮罩列表
    """
    print("\nAnalyzing training data complexity...")
    complexities = []
    
    # 計算每張圖片的複雜度指標
    for img, msk in tqdm(zip(images, masks)):
        metrics = analyze_image_complexity(img, msk)
        # 計算綜合複雜度分數
        complexity_score = (
            0.3 * metrics['edge_density'] +
            0.2 * metrics['mask_edge_density'] +
            0.2 * metrics['shape_complexity'] +
            0.1 * metrics['noise_level'] +
            0.1 * metrics['contrast'] +
            0.1 * metrics['sharpness']
        )
        complexities.append(complexity_score)
    
    # 計算複雜度閾值
    threshold = np.percentile(complexities, complexity_threshold * 100)
    
    # 過濾資料
    filtered_data = [(img, msk, score) for img, msk, score 
                     in zip(images, masks, complexities) 
                     if score <= threshold]
    
    # 分離圖片和遮罩
    filtered_images = [data[0] for data in filtered_data]
    filtered_masks = [data[1] for data in filtered_data]
    
    # 輸出統計信息
    print(f"\nFiltering results:")
    print(f"Original data count: {len(images)}")
    print(f"Filtered data count: {len(filtered_images)}")
    print(f"Removed {len(images) - len(filtered_images)} complex samples")
    
    # 可視化一些統計信息
    # plt.figure(figsize=(10, 5))
    # plt.hist(complexities, bins=50, alpha=0.75)
    # plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
    # plt.title('Distribution of Image Complexity Scores')
    # plt.xlabel('Complexity Score')
    # plt.ylabel('Count')
    # plt.legend()
    # plt.savefig('complexity_distribution.png')
    # plt.close()
    
    return filtered_images, filtered_masks

class ImageSimilarityAnalyzer:
    
    @staticmethod
    def extract_features(image: np.ndarray, mask: np.ndarray) -> Dict:
        features = {}
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # 1. color hist
        color_hist = []
        for i in range(3):  # RGB channels
            hist = cv2.calcHist([image], [i], None, [32], [0, 256])
            color_hist.extend(hist.flatten())
        features['color_hist'] = np.array(color_hist) / np.sum(color_hist)
        
        # 2. water_ratio
        water_region = mask > 127
        features['water_ratio'] = np.mean(water_region)
        
        # water loactation
        if np.any(water_region):
            y_indices, x_indices = np.where(water_region)
            features['water_center_y'] = np.mean(y_indices) / image.shape[0]
            features['water_center_x'] = np.mean(x_indices) / image.shape[1]
            features['water_spread'] = np.std(y_indices) / image.shape[0] + np.std(x_indices) / image.shape[1]
        else:
            features['water_center_y'] = 0
            features['water_center_x'] = 0
            features['water_spread'] = 0
        
        # 3. texture
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features['contrast'] = np.std(gray)
        features['brightness'] = np.mean(gray)
        
        # 4. edge
        edges = cv2.Canny(gray, 100, 200)
        features['edge_density'] = np.mean(edges > 0)
        
        return features

    @staticmethod
    def calculate_similarity(features1: Dict, features2: Dict) -> float:
        weights = {
            'color_hist': 0.3,
            'water_features': 0.4,
            'texture_features': 0.3
        }
        
        color_sim = 1 - cosine(features1['color_hist'], features2['color_hist'])
        
        water_features = np.array([
            abs(features1['water_ratio'] - features2['water_ratio']),
            abs(features1['water_center_x'] - features2['water_center_x']),
            abs(features1['water_center_y'] - features2['water_center_y']),
            abs(features1['water_spread'] - features2['water_spread'])
        ])
        water_sim = 1 - np.mean(water_features)
        
        texture_features = np.array([
            abs(features1['contrast'] - features2['contrast']) / max(features1['contrast'], features2['contrast']),
            abs(features1['brightness'] - features2['brightness']) / 255,
            abs(features1['edge_density'] - features2['edge_density'])
        ])
        texture_sim = 1 - np.mean(texture_features)
        
        similarity = (weights['color_hist'] * color_sim +
                     weights['water_features'] * water_sim +
                     weights['texture_features'] * texture_sim)
        
        return similarity

    def analyze_dataset_similarity(self, 
                                 train_images: List[np.ndarray],
                                 train_masks: List[np.ndarray],
                                 test_images: List[np.ndarray],
                                 test_masks: List[np.ndarray],
                                 top_k: int = 5) -> List[Dict]:
        results = []
        
        train_features = []
        for img, mask in tqdm(zip(train_images, train_masks)):
            features = self.extract_features(img, mask)
            train_features.append(features)
        
        for test_idx, (test_img, test_mask) in enumerate(tqdm(zip(test_images, test_masks))):
            test_features = self.extract_features(test_img, test_mask)
            
            similarities = []
            for train_idx, train_feat in enumerate(train_features):
                sim = self.calculate_similarity(test_features, train_feat)
                similarities.append((train_idx, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similar = similarities[:top_k]
            
            results.append({
                'test_index': test_idx,
                'similar_trains': [{'index': idx, 'similarity': sim} 
                                 for idx, sim in top_similar]
            })
        
        return results

def analyze_and_weight_augmentation(similarity_results: List[Dict], 
                                  min_weight: float = 1.0,
                                  max_weight: float = 3.0) -> Dict[int, float]:
    train_weights = {}
    
    for result in similarity_results:
        for similar in result['similar_trains']:
            train_idx = similar['index']
            sim_score = similar['similarity']
            
            if train_idx not in train_weights:
                train_weights[train_idx] = {'count': 0, 'total_sim': 0.0}
            
            train_weights[train_idx]['count'] += 1
            train_weights[train_idx]['total_sim'] += sim_score

    final_weights = {}
    max_count = max([w['count'] for w in train_weights.values()]) if train_weights else 1
    max_sim = max([w['total_sim'] for w in train_weights.values()]) if train_weights else 1
    
    for idx, weights in train_weights.items():
        normalized_count = weights['count'] / max_count
        normalized_sim = weights['total_sim'] / max_sim
        
        weight = min_weight + (max_weight - min_weight) * (normalized_count * 0.7 + normalized_sim * 0.3)
        final_weights[idx] = weight
    
    return final_weights

def calculate_water_ratio(mask: np.ndarray) -> float:
    return np.mean(mask > 127)

def create_base_transform() -> A.Compose:
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
    ])

def create_water_transform() -> A.Compose:
    return A.Compose([
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.05, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.4),
    ])

def create_mixed_sample(image1: np.ndarray, mask1: np.ndarray, 
                       image2: np.ndarray, mask2: np.ndarray,
                       alpha_range: Tuple[float, float] = (0.3, 0.7)) -> Tuple[np.ndarray, np.ndarray]:
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    mixed_image = cv2.addWeighted(image1, alpha, image2, 1-alpha, 0)
    mixed_mask = np.logical_or(mask1 > 127, mask2 > 127).astype(np.uint8) * 255
    return mixed_image, mixed_mask

def enhance_water_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    enhanced = image.copy()
    water_region = mask > 127
    
    # Blue channel enhance
    b, g, r = cv2.split(enhanced)
    b[water_region] = np.clip(b[water_region] * 1.2, 0, 255)
    enhanced = cv2.merge([b, g, r])
    
    # CLAHE
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    
    return enhanced

def process_single_image(image: np.ndarray, mask: np.ndarray, 
                        water_ratio: float,
                        base_transform: A.Compose,
                        water_transform: A.Compose,
                        aug_weight: float = 1.0,
                        is_training: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    augmented_images = []
    augmented_masks = []

    augmented_images.append(image)
    augmented_masks.append(mask)

    if not is_training:
        return augmented_images, augmented_masks

    total = 0
    def get_aug_count(base_count: int) -> int:
        count = max(1, int(base_count * aug_weight))
        return min(count, 7)
    
    base_count = 2
    if 0.2 <= water_ratio <= 0.4: # target
        base_count = 4
    
    for _ in range(get_aug_count(base_count)):
        augmented = base_transform(image=image, mask=mask)
        augmented_images.append(augmented['image'])
        augmented_masks.append(augmented['mask'])
        total += 1

    if water_ratio >= 0.2:
        water_count = 2
        if 0.2 <= water_ratio <= 0.4:
            water_count = 3
        
        for _ in range(get_aug_count(water_count)):
            augmented = water_transform(image=image, mask=mask)
            augmented_images.append(augmented['image'])
            augmented_masks.append(augmented['mask'])
            total += 1

        enhanced_count = max(1, int(aug_weight))
        for _ in range(enhanced_count):
            enhanced = enhance_water_features(image, mask)
            augmented_images.append(enhanced)
            augmented_masks.append(mask)
            total += 1
    
    print(total, end=" ")
    return augmented_images, augmented_masks

def process_dataset(images: List[np.ndarray],
                   masks: List[np.ndarray],
                   min_water_ratio: float,
                   aug_weights: Dict[int, float] = None,
                   is_training: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    
    base_transform = create_base_transform()
    water_transform = create_water_transform()
    
    processed_images = []
    processed_masks = []
    high_water_samples = []
    
    # Phase 1
    print("First pass: Processing all images...")
    for idx, (img, mask) in enumerate(tqdm(zip(images, masks), total=len(images))):
        try:
            water_ratio = calculate_water_ratio(mask)
            aug_weight = aug_weights.get(idx, 1.0) if aug_weights else 1.0
            
            aug_images, aug_masks = process_single_image(
                img, mask, water_ratio,
                base_transform, water_transform,
                aug_weight=aug_weight,
                is_training=is_training
            )
            processed_images.extend(aug_images)
            processed_masks.extend(aug_masks)
            
            if water_ratio >= min_water_ratio:
                high_water_samples.append((img, mask))
                
        except Exception as e:
            print(f"Error processing image at index {idx}: {str(e)}")
            continue
    
    # Phase 2
    if is_training and high_water_samples and len(high_water_samples) > 0:
        print("\nSecond pass: Creating mixed samples for low water ratio images...")
        for idx, (img, mask) in enumerate(tqdm(zip(images, masks), total=len(images))):
            try:
                water_ratio = calculate_water_ratio(mask)
                
                if water_ratio < min_water_ratio:
                    aug_weight = aug_weights.get(idx, 1.0) if aug_weights else 1.0
                    mix_count = max(1, int(2 * aug_weight))
                    
                    for _ in range(mix_count):
                        high_img, high_mask = random.choice(high_water_samples)
                        
                        mixed_img, mixed_mask = create_mixed_sample(
                            img, mask, high_img, high_mask,
                            alpha_range=(0.3, 0.7)
                        )
                        processed_images.append(mixed_img)
                        processed_masks.append(mixed_mask)
                        
            except Exception as e:
                print(f"Error processing image at index {idx} in second pass: {str(e)}")
                continue
    
    processed_images = np.array(processed_images)
    processed_masks = np.array(processed_masks)
    
    print(f"\nProcessed dataset statistics:")
    print(f"Total samples: {len(processed_images)}")
    print(f"High water ratio samples: {len(high_water_samples)}")
    print(f"Final dataset shape: {processed_images.shape}")
    
    return processed_images, processed_masks

def main():
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    args = parse_arguments()
    height, width = args.image_size
    
    print("\n=== Configuration ===")
    print(f"Image size: {width}x{height}")
    print(f"Train/Val split ratio: {args.train_ratio:.2f}/{1-args.train_ratio:.2f}")
    print(f"Minimum water ratio: {args.min_water_ratio:.2f}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.output_dir = os.path.join(script_dir, args.output_dir)
    print(f"Output directory: {args.output_dir}")
    
    train_image_paths = natsorted(glob.glob(os.path.join(script_dir, "training/image/*.png")))
    train_mask_paths = natsorted(glob.glob(os.path.join(script_dir, "training/mask/*.png")))
    test_image_paths = natsorted(glob.glob(os.path.join(script_dir, "testing/image/*.[jJ][pP][gG]")))
    test_mask_paths = natsorted(glob.glob(os.path.join(script_dir, "testing/mask/*.png")))
    
    print("\nLoading training dataset...")
    train_images, train_masks = load_dataset(train_image_paths, train_mask_paths, (width, height))
    print("\nLoading test dataset...")
    test_images, test_masks = load_dataset(test_image_paths, test_mask_paths, (width, height))

    train_images, train_masks = filter_training_data(train_images, train_masks, 
                                               complexity_threshold=args.con_th)

    if args.shuffle == 1:
        indices = np.arange(len(train_images))
        np.random.shuffle(indices)
        train_images = np.array(train_images)[indices]
        train_masks = np.array(train_masks)[indices]
    train_split = int(len(train_images) * args.train_ratio)
    train_img = train_images[:train_split]
    train_msk = train_masks[:train_split]
    val_img = train_images[train_split:]
    val_msk = train_masks[train_split:]
    
    print("\nCalculating augmentation weights...")
    analyzer = ImageSimilarityAnalyzer()
    similarity_results = analyzer.analyze_dataset_similarity(
        train_images, train_masks,
        test_images, test_masks,
        top_k=5
    )
    aug_weights = analyze_and_weight_augmentation(
        similarity_results,
        min_weight=1.0,
        max_weight=3.0
    )
    
    # print("\n=== Similarity Analysis Results ===")
    # for result in similarity_results:
    #     test_idx = result['test_index']
    #     print(f"\nTest image {test_idx}:")
    #     for similar in result['similar_trains']:
    #         train_idx = similar['index']
    #         sim_score = similar['similarity']
    #         weight = aug_weights.get(train_idx, 1.0)
    #         print(f"  Training image {train_idx}: "
    #               f"similarity = {sim_score:.3f}, "
    #               f"augmentation weight = {weight:.2f}")
    
    print("\nProcessing datasets with weighted augmentation...")
    data_train, mask_train = process_dataset(
        train_img, train_msk,
        min_water_ratio=0.2,
        aug_weights=aug_weights,
        is_training=True
    )

    print("\nProcessing validation set...")
    data_val, mask_val = process_dataset(
        val_img, val_msk,
        args.min_water_ratio,
        is_training=False
    )
    
    print("\nProcessing test set...")
    data_test, mask_test = process_dataset(
        test_images, test_masks,
        args.min_water_ratio,
        is_training=False
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f'\nSaving processed data to {output_dir}...')
    np.save(output_dir / 'data_train.npy', data_train)
    np.save(output_dir / 'data_val.npy', data_val)
    np.save(output_dir / 'data_test.npy', data_test)
    np.save(output_dir / 'mask_train.npy', mask_train)
    np.save(output_dir / 'mask_val.npy', mask_val)
    np.save(output_dir / 'mask_test.npy', mask_test)
    
    print('\n=== Final Dataset Statistics ===')
    print(f'Training samples: {len(data_train)}')
    print(f'Validation samples: {len(data_val)}')
    print(f'Test samples: {len(data_test)}')

if __name__ == '__main__':
    main()