import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2

def calculate_image_stats(img, mask):
    """計算單張圖片的統計量"""
    return {
        'mean': np.mean(img),
        'std': np.std(img),
        'max': np.max(img),
        'min': np.min(img),
        'mask_ratio': np.sum(mask == 255) / mask.size,
        'mask_unique_values': np.unique(mask)
    }

def calculate_dataset_stats(data, masks):
    """計算整個數據集的統計量"""
    stats = {
        'mean': [],
        'std': [],
        'max': [],
        'min': [],
        'mask_ratio': [],
        'mask_unique_values': set()
    }
    
    for img, mask in zip(data, masks):
        img_stats = calculate_image_stats(img, mask)
        for key in stats:
            if key == 'mask_unique_values':
                stats[key].update(img_stats[key])  # 使用 update 來添加新的獨特值
            else:
                stats[key].append(img_stats[key])

    stats['mask_unique_values'] = sorted(list(stats['mask_unique_values']))
    return stats

def plot_statistics(stats, dataset_name, save_path):
    """繪製統計圖表"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    
    # 均值和標準差
    axes[0].plot(stats['mean'], label='Mean', color='blue', alpha=0.7)
    axes[0].fill_between(range(len(stats['mean'])), 
                        np.array(stats['mean']) - np.array(stats['std']),
                        np.array(stats['mean']) + np.array(stats['std']),
                        alpha=0.3, color='blue')
    axes[0].set_title(f'Mean and Standard Deviation per Image ({dataset_name})')
    axes[0].set_xlabel('Image Index')
    axes[0].set_ylabel('Pixel Value')
    axes[0].legend()
    
    # 最大值和最小值
    axes[1].plot(stats['max'], label='Max', color='red')
    axes[1].plot(stats['min'], label='Min', color='green')
    axes[1].set_title(f'Max and Min Values per Image ({dataset_name})')
    axes[1].set_xlabel('Image Index')
    axes[1].set_ylabel('Pixel Value')
    axes[1].legend()
    
    # 水體比例
    axes[2].plot(stats['mask_ratio'], label='Water Ratio', color='purple')
    axes[2].set_title(f'Water Ratio per Image ({dataset_name})')
    axes[2].set_xlabel('Image Index')
    axes[2].set_ylabel('Ratio')
    axes[2].legend()
    
    # Unique Values 的直方圖
    unique_values = np.array(stats['mask_unique_values'])
    axes[3].hist(unique_values, bins='auto', color='teal', alpha=0.7)
    axes[3].set_title(f'Distribution of Mask Unique Values ({dataset_name})')
    axes[3].set_xlabel('Pixel Value')
    axes[3].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_samples(data, masks, n_samples, dataset_name, save_path):
    """可視化數據集的樣本"""
    indices = random.sample(range(len(data)), n_samples)
    
    fig, axes = plt.subplots(4, 8, figsize=(28, 14))
    for i, idx in enumerate(indices):
        row = i // 4
        col = (i % 4) * 2
        
        # 顯示原始圖片
        axes[row, col].imshow(data[idx].astype(np.uint8))
        axes[row, col].set_title(f'{dataset_name} Image {idx}')
        axes[row, col].axis('off')
        
        # 顯示遮罩
        axes[row, col + 1].imshow(masks[idx], cmap='gray')
        axes[row, col + 1].set_title(f'{dataset_name} Mask {idx}')
        axes[row, col + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return indices

def print_dataset_stats(stats, dataset_name):
    """打印數據集統計信息"""
    print(f"\n{dataset_name} Dataset Statistics:")
    print(f"Number of images: {len(stats['mean'])}")
    print(f"Average mean pixel value: {np.mean(stats['mean']):.2f}")
    print(f"Average std pixel value: {np.mean(stats['std']):.2f}")
    print(f"Max pixel value: {np.max(stats['max']):.2f}")
    print(f"Min pixel value: {np.min(stats['min']):.2f}")

    print("\nMask Unique Values Statistics:")
    print(f"Average water ratio: {np.mean(stats['mask_ratio']):.2%}")
    print(f"Number of unique values: {len(stats['mask_unique_values'])}")
    print(f"Min unique value: {min(stats['mask_unique_values'])}")
    print(f"Max unique value: {max(stats['mask_unique_values'])}")
    if len(stats['mask_unique_values']) <= 10:
        print(f"All unique values: {stats['mask_unique_values']}")
    else:
        print(f"First 5 values: {stats['mask_unique_values'][:5]}")
        print(f"Last 5 values: {stats['mask_unique_values'][-5:]}")

def main():
    # 讀取數據
    train_data = np.load('data/data_train.npy')
    train_masks = np.load('data/mask_train.npy')
    val_data = np.load('data/data_val.npy')
    val_masks = np.load('data/mask_val.npy')
    test_data = np.load('data/data_test.npy')
    test_masks = np.load('data/mask_test.npy')
    
    print('train_masks.shape = ', train_masks.shape)
    print('np.unique(train_masks) = ', np.unique(train_masks[0]))
    # 讀取原始 PNG
    original_mask = cv2.imread('training/mask/1.png', cv2.IMREAD_GRAYSCALE)

    # 開啟文件並寫入詳細資訊
    with open('mask_comparison.txt', 'w') as f:
        # 原始 PNG 資訊
        f.write("=== Original PNG Mask ===\n")
        f.write(f"Shape: {original_mask.shape}\n")
        f.write(f"Unique values: {np.unique(original_mask)}\n")
        f.write(f"Min value: {original_mask.min()}\n")
        f.write(f"Max value: {original_mask.max()}\n")
        f.write("\nFirst 10x10 values of original mask:\n")
        np.savetxt(f, original_mask, fmt='%3d')
        
        # 處理後的 mask 資訊
        f.write("\n\n=== Processed Mask ===\n")
        f.write(f"Shape: {train_masks[0].shape}\n")
        f.write(f"Unique values: {np.unique(train_masks[0])}\n")
        f.write(f"Min value: {train_masks[0].min():.6f}\n")
        f.write(f"Max value: {train_masks[0].max():.6f}\n")
        f.write("\nFirst 10x10 values of processed mask:\n")
        np.savetxt(f, train_masks[0], fmt='%.6f')
    exit()

    # 計算統計量
    train_stats = calculate_dataset_stats(train_data, train_masks)
    val_stats = calculate_dataset_stats(val_data, val_masks)
    test_stats = calculate_dataset_stats(test_data, test_masks)
    
    # 繪製統計圖
    plot_statistics(train_stats, "Training", "statistics_train.png")
    plot_statistics(val_stats, "Validation", "statistics_validation.png")
    
    # 可視化樣本
    train_indices = visualize_samples(train_data, train_masks, 16, "Train", "sample_train.png")
    val_indices = visualize_samples(val_data, val_masks, 15, "Val", "sample_validation.png")
    
    # 打印統計信息
    print_dataset_stats(train_stats, "Training")
    print_dataset_stats(val_stats, "Validation")
    print_dataset_stats(test_stats, "Test")
    
    print("\nSelected training image indices:", train_indices)
    print("Selected validation image indices:", val_indices)

if __name__ == "__main__":
    main()