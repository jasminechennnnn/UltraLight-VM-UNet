##### refactored
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
from typing import Dict, List, Set, Tuple, Union
from pathlib import Path

class ImageStats:
    """Class for handling image statistics calculations and visualization"""
    
    @staticmethod
    def calculate_single_image_stats(img: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Calculate statistics for a single image and its mask
        
        Args:
            img: Input image array
            mask: Corresponding mask array
        
        Returns:
            Dictionary containing calculated statistics
        """
        return {
            'mean': np.mean(img),
            'std': np.std(img),
            'max': np.max(img),
            'min': np.min(img),
            'mask_ratio': np.sum(mask == 255) / mask.size,
            'mask_unique_values': np.unique(mask)
        }

    @staticmethod
    def calculate_dataset_stats(data: np.ndarray, masks: np.ndarray) -> Dict:
        """
        Calculate statistics for the entire dataset
        
        Args:
            data: Array of images
            masks: Array of corresponding masks
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'mean': [],
            'std': [],
            'max': [],
            'min': [],
            'mask_ratio': [],
            'mask_unique_values': set()
        }
        
        for img, mask in zip(data, masks):
            img_stats = ImageStats.calculate_single_image_stats(img, mask)
            for key in stats:
                if key == 'mask_unique_values':
                    stats[key].update(img_stats[key])
                else:
                    stats[key].append(img_stats[key])

        stats['mask_unique_values'] = sorted(list(stats['mask_unique_values']))
        return stats

class Visualizer:
    """Class for handling data visualization tasks"""
    
    @staticmethod
    def plot_statistics(stats: Dict, dataset_name: str, save_path: str) -> None:
        """
        Create and save statistical plots
        
        Args:
            stats: Dictionary containing dataset statistics
            dataset_name: Name of the dataset for plot titles
            save_path: Path to save the resulting plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 20))
        
        # Mean and standard deviation plot
        axes[0].plot(stats['mean'], label='Mean', color='blue', alpha=0.7)
        axes[0].fill_between(range(len(stats['mean'])), 
                           np.array(stats['mean']) - np.array(stats['std']),
                           np.array(stats['mean']) + np.array(stats['std']),
                           alpha=0.3, color='blue')
        axes[0].set_title(f'Mean and Standard Deviation per Image ({dataset_name})')
        axes[0].set_xlabel('Image Index')
        axes[0].set_ylabel('Pixel Value')
        axes[0].legend()
        
        # Max and min values plot
        axes[1].plot(stats['max'], label='Max', color='red')
        axes[1].plot(stats['min'], label='Min', color='green')
        axes[1].set_title(f'Max and Min Values per Image ({dataset_name})')
        axes[1].set_xlabel('Image Index')
        axes[1].set_ylabel('Pixel Value')
        axes[1].legend()
        
        # Water ratio plot
        axes[2].plot(stats['mask_ratio'], label='Water Ratio', color='purple')
        axes[2].set_title(f'Water Ratio per Image ({dataset_name})')
        axes[2].set_xlabel('Image Index')
        axes[2].set_ylabel('Ratio')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def visualize_samples(data: np.ndarray, masks: np.ndarray, 
                         n_samples: int, dataset_name: str, 
                         save_path: str) -> List[int]:
        """
        Visualize random samples from the dataset
        
        Args:
            data: Array of images
            masks: Array of corresponding masks
            n_samples: Number of samples to visualize
            dataset_name: Name of the dataset
            save_path: Path to save the visualization
            
        Returns:
            List of selected sample indices
        """
        indices = random.sample(range(len(data)), n_samples)
        
        fig, axes = plt.subplots(4, 8, figsize=(28, 14))
        for i, idx in enumerate(indices):
            row = i // 4
            col = (i % 4) * 2
            
            axes[row, col].imshow(data[idx].astype(np.uint8))
            axes[row, col].set_title(f'{dataset_name} Image {idx}')
            axes[row, col].axis('off')
            
            axes[row, col + 1].imshow(masks[idx], cmap='gray')
            axes[row, col + 1].set_title(f'{dataset_name} Mask {idx}')
            axes[row, col + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return indices

class DataAnalyzer:
    """Class for analyzing and comparing mask data"""
    
    @staticmethod
    def compare_masks(original_mask_path: str, processed_mask: np.ndarray, 
                     output_file: str) -> None:
        """
        Compare original and processed masks and save analysis
        
        Args:
            original_mask_path: Path to original mask file
            processed_mask: Processed mask array
            output_file: Path to save comparison results
        """
        original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
        
        with open(output_file, 'w') as f:
            # Original mask information
            f.write("=== Original PNG Mask ===\n")
            f.write(f"Shape: {original_mask.shape}\n")
            f.write(f"Unique values: {np.unique(original_mask)}\n")
            f.write(f"Min value: {original_mask.min()}\n")
            f.write(f"Max value: {original_mask.max()}\n")
            f.write("\nFirst 10x10 values of original mask:\n")
            np.savetxt(f, original_mask[:10, :10], fmt='%3d')
            
            # Processed mask information
            f.write("\n\n=== Processed Mask ===\n")
            f.write(f"Shape: {processed_mask.shape}\n")
            f.write(f"Unique values: {np.unique(processed_mask)}\n")
            f.write(f"Min value: {processed_mask.min():.6f}\n")
            f.write(f"Max value: {processed_mask.max():.6f}\n")
            f.write("\nFirst 10x10 values of processed mask:\n")
            np.savetxt(f, processed_mask[:10, :10], fmt='%.6f')

    @staticmethod
    def print_dataset_stats(stats: Dict, dataset_name: str) -> None:
        """
        Print statistical summary of the dataset
        
        Args:
            stats: Dictionary containing dataset statistics
            dataset_name: Name of the dataset
        """
        print(f"\n=== {dataset_name} Dataset Statistics ===")
        print(f"Number of images: {len(stats['mean'])}")
        print(f"Average mean pixel value: {np.mean(stats['mean']):.2f}")
        print(f"Average std pixel value: {np.mean(stats['std']):.2f}")
        print(f"Max pixel value: {np.max(stats['max']):.2f}")
        print(f"Min pixel value: {np.min(stats['min']):.2f}")
        print("------------------------------")
        print("Mask Unique Values Statistics:")
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
    """Main function to run the analysis"""
    # Load data
    data_dir = Path('data')
    train_data = np.load(data_dir / 'data_train.npy')
    train_masks = np.load(data_dir / 'mask_train.npy')
    val_data = np.load(data_dir / 'data_val.npy')
    val_masks = np.load(data_dir / 'mask_val.npy')
    test_data = np.load(data_dir / 'data_test.npy')
    test_masks = np.load(data_dir / 'mask_test.npy')
    
    print(f'Train masks shape: {train_masks.shape}')
    
    # Compare original and processed masks
    DataAnalyzer.compare_masks(
        'training/mask/1.png',
        train_masks[0],
        'mask_comparison.txt'
    )
    
    # Calculate statistics
    train_stats = ImageStats.calculate_dataset_stats(train_data, train_masks)
    val_stats = ImageStats.calculate_dataset_stats(val_data, val_masks)
    test_stats = ImageStats.calculate_dataset_stats(test_data, test_masks)
    
    # Create visualizations
    Visualizer.plot_statistics(train_stats, "Training", "statistics_train.png")
    Visualizer.plot_statistics(val_stats, "Validation", "statistics_validation.png")
    
    train_indices = Visualizer.visualize_samples(
        train_data, train_masks, 16, "Train", "sample_train.png")
    val_indices = Visualizer.visualize_samples(
        val_data, val_masks, 16, "Val", "sample_validation.png")
    
    # Print statistics
    DataAnalyzer.print_dataset_stats(train_stats, "Training")
    DataAnalyzer.print_dataset_stats(val_stats, "Validation")
    DataAnalyzer.print_dataset_stats(test_stats, "Test")
    
    print(f"\nSelected training image indices: {train_indices}")
    print(f"Selected validation image indices: {val_indices}")

if __name__ == "__main__":
    main()