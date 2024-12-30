from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
import os
from PIL import Image
from einops.layers.torch import Rearrange
from scipy.ndimage.morphology import binary_dilation
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import ndimage
from utils import *
from typing import List, Tuple # , Dict
# from pathlib import Path


# ===== normalize over the dataset 
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / \
                              (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255

    # imgs_normalized = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
    return imgs_normalized


## Temporary
class isic_loader(Dataset):
    """ dataset class for Brats datasets
    """
    def __init__(self, path_Data, train = True, Test = False, logger = None):
        super(isic_loader, self)
        self.train = train
        log_info = logger.info if logger else print
        
        log_info("Start reading data from " + path_Data)

        if train:
            self.data = np.load(path_Data+'data_train.npy')
            self.mask = np.load(path_Data+'mask_train.npy')
            log_info("Training data statistics:")
            log_info(f"Range: {self.data.min():.3f} to {self.data.max():.3f}")
            log_info(f"Mean: {self.data.mean():.3f}")
            log_info(f"Std: {self.data.std():.3f}")                  
        
        else:
          if Test:
            self.data = np.load(path_Data+'data_test.npy')
            self.mask = np.load(path_Data+'mask_test.npy')
          else:
            self.data = np.load(path_Data+'data_val.npy')
            self.mask = np.load(path_Data+'mask_val.npy')          
        
        log_info(f"Mode train = {train}, shape = {self.data.shape}")
        print(f"Mode train = {train}, shape = {self.data.shape}")
        self.data = dataset_normalized(self.data)
        log_info(f"Image Range after normalization: {self.data.min():.3f} to {self.data.max():.3f}")
        
        self.mask = np.expand_dims(self.mask, axis=3)
        self.mask = self.mask/255.
        log_info(f"Mask Unique values: {np.unique(self.mask)}")

    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]

        if torch.isnan(torch.tensor(img)).any():
            print(f"NaN found in image {indx}")
        if torch.isnan(torch.tensor(seg)).any():
            print(f"NaN found in mask {indx}")

        if self.train:
            if random.random() > 0.5:
                img, seg = self.random_rot_flip(img, seg)
            if random.random() > 0.5:
                img, seg = self.random_rotate(img, seg)
        
        seg = torch.tensor(seg.copy())
        img = torch.tensor(img.copy())
        img = img.permute( 2, 0, 1)
        seg = seg.permute( 2, 0, 1)

        return img, seg
    
    def random_rot_flip(self,image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label
    
    def random_rotate(self,image, label):
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label


               
    def __len__(self):
        return len(self.data)
    
# def add_frequency_features(img):
#     """生成頻域相關的特徵"""
#     # 轉為灰度圖
#     if len(img.shape) == 3:
#         gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
#     else:
#         gray = (img * 255).astype(np.uint8)
    
#     # 1. FFT 變換
#     f = np.fft.fft2(gray)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = np.log1p(np.abs(fshift))
    
#     # 正規化頻譜
#     magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / \
#                         (magnitude_spectrum.max() - magnitude_spectrum.min())
    
#     # 2. 不同尺度的高斯濾波器響應
#     blurred_features = []
#     for sigma in [1, 3, 5]:
#         blur = cv2.GaussianBlur(gray, (0, 0), sigma)
#         blurred_features.append(blur)
    
#     # 3. 多尺度 DoG (Difference of Gaussian)
#     dog_features = []
#     sigma_pairs = [(1, 2), (2, 4), (4, 8)]
#     for sigma1, sigma2 in sigma_pairs:
#         g1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
#         g2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
#         dog = g1 - g2
#         dog_features.append(dog)
    
#     # 4. Gabor 濾波器響應
#     gabor_features = []
#     for theta in [0, 45, 90, 135]:
#         for freq in [0.1, 0.5]:
#             kernel = cv2.getGaborKernel((21, 21), sigma=5, theta=theta*np.pi/180,
#                                       lambd=1/freq, gamma=0.5, psi=0)
#             gabor = cv2.filter2D(gray, -1, kernel)
#             gabor_features.append(gabor)
    
#     return {
#         'magnitude_spectrum': magnitude_spectrum,
#         'blurred': blurred_features,
#         'dog': dog_features,
#         'gabor': gabor_features
#     }
# class WaterSegLoader(Dataset):    
#     def __init__(self, path_Data, train=True, Test=False, logger=None):
#         super(WaterSegLoader, self).__init__()
#         self.train = train
#         self.log_info = logger.info if logger else print
        
#         self.log_info("Start reading data from " + path_Data)

#         # 載入資料
#         if train:
#             self.data = np.load(path_Data+'data_train.npy')
#             self.mask = np.load(path_Data+'mask_train.npy')
#             self.log_info("Training data statistics:")
#             self.log_info(f"Range: {self.data.min():.3f} to {self.data.max():.3f}")
#             self.log_info(f"Mean: {self.data.mean():.3f}")
#             self.log_info(f"Std: {self.data.std():.3f}")
#         else:
#             if Test:
#                 self.data = np.load(path_Data+'data_test.npy')
#                 self.mask = np.load(path_Data+'mask_test.npy')
#             else:
#                 self.data = np.load(path_Data+'data_val.npy')
#                 self.mask = np.load(path_Data+'mask_val.npy')
        
#         self.log_info(f"Mode train = {train}, shape = {self.data.shape}")
        
#         # 原始圖片正規化
#         self.data = dataset_normalized(self.data)
#         self.log_info(f"Image Range after normalization: {self.data.min():.3f} to {self.data.max():.3f}")
        
#         # 處理遮罩
#         self.mask = np.expand_dims(self.mask, axis=3)
#         self.mask = self.mask/255.
#         self.log_info(f"Mask Unique values: {np.unique(self.mask)}")
        
#         # 選擇要使用的特徵通道
#         self.use_features = {
#             'frequency': True,    # 頻譜特徵
#             'dog': False,         # DoG 特徵
#             'gabor': False,       # Gabor 特徵
#             'ndwi': True         # 原始的 NDWI
#         }
        
#         # 計算總通道數
#         self.n_extra_channels = sum([
#             1 if self.use_features['frequency'] else 0,  # 頻譜
#             3 if self.use_features['dog'] else 0,        # DoG (3個尺度)
#             3 if self.use_features['gabor'] else 0,      # Gabor (選擇3個主要方向)
#             1 if self.use_features['ndwi'] else 0        # NDWI
#         ])
        
#         self.extra_channels = self._generate_extra_channels()  # 呼叫產生額外特徵通道
#         self.log_info(f"Extra channels shape: {self.extra_channels.shape}")

#     def _generate_extra_channels(self):
#         """生成額外的特徵通道"""
#         n_samples = self.data.shape[0]
#         h, w = self.data.shape[1:3]
#         extra_channels = np.zeros((n_samples, h, w, self.n_extra_channels))
        
#         for i in range(n_samples):
#             img = self.data[i]
#             channel_idx = 0
            
#             # 1. 頻域特徵
#             freq_features = add_frequency_features(img)
            
#             if self.use_features['frequency']:
#                 extra_channels[i, :, :, channel_idx] = freq_features['magnitude_spectrum']
#                 channel_idx += 1
            
#             if self.use_features['dog']:
#                 for dog in freq_features['dog'][:3]:  # 使用前3個 DoG 特徵
#                     dog_norm = (dog - dog.min()) / (dog.max() - dog.min())
#                     extra_channels[i, :, :, channel_idx] = dog_norm
#                     channel_idx += 1
            
#             if self.use_features['gabor']:
#                 # 選擇3個最具代表性的 Gabor 響應
#                 selected_gabor = freq_features['gabor'][:3]
#                 for gabor in selected_gabor:
#                     gabor_norm = (gabor - gabor.min()) / (gabor.max() - gabor.min())
#                     extra_channels[i, :, :, channel_idx] = gabor_norm
#                     channel_idx += 1
            
#             if self.use_features['ndwi']:
#                 # 添加 NDWI
#                 R, G = img[:,:,0], img[:,:,1]
#                 ndwi = (G - R) / (G + R + 1e-6)
#                 ndwi_norm = (ndwi - ndwi.min()) / (ndwi.max() - ndwi.min())
#                 extra_channels[i, :, :, channel_idx] = ndwi_norm
        
#         return extra_channels

#     def __getitem__(self, indx):
#         # 獲取原始圖片和額外特徵
#         img = self.data[indx]
#         extra = self.extra_channels[indx]
#         seg = self.mask[indx]
        
#         # 合併原始圖片和額外特徵
#         combined_img = np.concatenate([img, extra], axis=2)  # shape: (H, W, 6)

#         # 檢查 NaN
#         if torch.isnan(torch.tensor(combined_img)).any():
#             print(f"NaN found in image {indx}")
#         if torch.isnan(torch.tensor(seg)).any():
#             print(f"NaN found in mask {indx}")

#         # 資料增強
#         if self.train:
#             if random.random() > 0.5:
#                 combined_img, seg = self.random_rot_flip(combined_img, seg)
#             if random.random() > 0.5:
#                 combined_img, seg = self.random_rotate(combined_img, seg)
        
#         # 轉換為 tensor
#         seg = torch.tensor(seg.copy())
#         combined_img = torch.tensor(combined_img.copy())
        
#         # 調整通道順序
#         combined_img = combined_img.permute(2, 0, 1)  # (6, H, W)
#         seg = seg.permute(2, 0, 1)      # (1, H, W)

#         return combined_img, seg
    
#     def random_rot_flip(self, image, label):
#         k = np.random.randint(0, 4)
#         image = np.rot90(image, k)
#         label = np.rot90(label, k)
#         axis = np.random.randint(0, 2)
#         image = np.flip(image, axis=axis).copy()
#         label = np.flip(label, axis=axis).copy()
#         return image, label
    
#     def random_rotate(self, image, label):
#         angle = np.random.randint(20, 80)
#         image = ndimage.rotate(image, angle, order=0, reshape=False)
#         label = ndimage.rotate(label, angle, order=0, reshape=False)
#         return image, label
    
#     def __len__(self):
#         return len(self.data)
    

class WaterSegLoader(Dataset):
    def __init__(self, 
                 path_Data: str,
                 train: bool = True,
                 Test: bool = False,
                 img_size: Tuple[int, int] = (256, 256),
                 extra_features: List[str] = ['ndwi', 'fft'],
                 logger=None):
        """
        Args:
            path_Data: 資料路徑
            train: 是否為訓練模式
            test: 是否為測試模式
            img_size: 圖片大小
            extra_features: 要使用的額外特徵列表
                可選項: ['ndwi', 'fft', 'gabor', 'dog', 'gray']
            logger: 紀錄器
        """
        super().__init__()
        self.train = train
        self.img_size = img_size
        self.log_info = logger.info if logger else print
        self.extra_features = extra_features
        
        # 載入資料
        self.log_info(f"Loading data from {path_Data}")
        if train:
            self.data = np.load(path_Data+'data_train.npy')
            self.mask = np.load(path_Data+'mask_train.npy')
        else:
            if Test:
                self.data = np.load(path_Data+'data_test.npy')
                self.mask = np.load(path_Data+'mask_test.npy')
            else:
                self.data = np.load(path_Data+'data_val.npy')
                self.mask = np.load(path_Data+'mask_val.npy')
        
        # 資料正規化和前處理
        self._preprocess_data()
        
        # 生成額外特徵
        if extra_features:
            self.extra_channels = self._generate_extra_channels()
            self.log_info(f"Generated {self.extra_channels.shape[-1]} extra channels")
        else:
            self.extra_channels = None
            
    def _preprocess_data(self):
        """資料預處理"""
        # 正規化原始圖片
        self.data = self._normalize_data(self.data)
        self.log_info(f"Image shape: {self.data.shape}")
        
        # 處理遮罩
        self.mask = np.expand_dims(self.mask, axis=3)
        self.mask = self.mask/255.
        
    def _normalize_data(self, imgs):
        """正規化方法"""
        imgs_normalized = np.empty(imgs.shape)
        imgs_std = np.std(imgs)
        imgs_mean = np.mean(imgs)
        imgs_normalized = (imgs-imgs_mean)/imgs_std
        for i in range(imgs.shape[0]):
            imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / \
                                (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
        return imgs_normalized

    def _normalize_channel(self, channel):
        """標準化單一通道"""
        # Z-score 標準化
        std = np.std(channel)
        mean = np.mean(channel)
        if std != 0:
            channel = (channel - mean) / std
        
        # Min-max 縮放到 [0, 1]
        min_val = np.min(channel)
        max_val = np.max(channel)
        if max_val != min_val:
            channel = (channel - min_val) / (max_val - min_val)
        
        return channel

    def _generate_extra_channels(self):
        """生成額外特徵通道"""
        n_samples = len(self.data)
        extra_channels_list = []
        
        for img in self.data:
            current_channels = []
            
            # 轉換為灰度圖供後續使用
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # 根據設定生成不同特徵
            for feature in self.extra_features:
                if feature == 'ndwi':
                    # 水體指數
                    R, G = img[:,:,0], img[:,:,1]
                    ndwi = (G - R) / (G + R + 1e-6)
                    ndwi = (ndwi - ndwi.min()) / (ndwi.max() - ndwi.min() + 1e-6)
                    current_channels.append(ndwi)
                    
                elif feature == 'fft':
                    # FFT 頻譜特徵
                    f = np.fft.fft2(gray)
                    fshift = np.fft.fftshift(f)
                    magnitude_spectrum = np.log1p(np.abs(fshift))
                    magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / \
                                      (magnitude_spectrum.max() - magnitude_spectrum.min())
                    current_channels.append(magnitude_spectrum)
                    
                elif feature == 'gabor':
                    # Gabor 紋理特徵
                    theta = 45  # 選擇一個主要方向
                    kernel = cv2.getGaborKernel((21, 21), sigma=5, theta=theta*np.pi/180,
                                              lambd=10, gamma=0.5, psi=0)
                    gabor = cv2.filter2D(gray, -1, kernel)
                    gabor = (gabor - gabor.min()) / (gabor.max() - gabor.min())
                    current_channels.append(gabor)
                    
                elif feature == 'dog':
                    # Difference of Gaussian
                    g1 = cv2.GaussianBlur(gray, (0, 0), 1)
                    g2 = cv2.GaussianBlur(gray, (0, 0), 3)
                    dog = (g1 - g2)
                    dog = (dog - dog.min()) / (dog.max() - dog.min())
                    current_channels.append(dog)
                    
                elif feature == 'gray':
                    # 灰度圖
                    gray_norm = gray / 255.0
                    current_channels.append(gray_norm)
            
            # 堆疊當前圖片的所有特徵通道
            extra_channels_list.append(np.stack(current_channels, axis=-1))
        
        # 整體標準化
        extra_channels = np.stack(extra_channels_list, axis=0)
        
        # 在整個資料集上進行標準化
        for c in range(extra_channels.shape[-1]):
            channel_data = extra_channels[..., c]
            extra_channels[..., c] = self._normalize_channel(channel_data)
        
        self.log_info("Extra channels statistics after normalization:")
        for i, feature in enumerate(self.extra_features):
            channel_data = extra_channels[..., i]
            self.log_info(f"{feature}: mean={channel_data.mean():.3f}, std={channel_data.std():.3f}, "
                        f"min={channel_data.min():.3f}, max={channel_data.max():.3f}")
        
        return extra_channels

    def __getitem__(self, idx):
        img = self.data[idx]
        mask = self.mask[idx]
        
        # 合併原始圖片和額外特徵（如果有的話）
        if self.extra_channels is not None:
            img = np.concatenate([img, self.extra_channels[idx]], axis=-1)
        
        # 資料增強（僅用於訓練）
        if self.train:
            if random.random() > 0.5:
                img, mask = self.random_rot_flip(img, mask)
            if random.random() > 0.5:
                img, mask = self.random_rotate(img, mask)
        
        # 轉換為 tensor
        img = torch.from_numpy(img.copy()).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask.copy()).float().permute(2, 0, 1)
        
        return img, mask
    
    def random_rot_flip(self, img, mask):
        k = np.random.randint(0, 4)
        img = np.rot90(img, k)
        mask = np.rot90(mask, k)
        axis = np.random.randint(0, 2)
        img = np.flip(img, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()
        return img, mask

    def random_rotate(self, img, mask):
        angle = np.random.randint(20, 80)
        img = ndimage.rotate(img, angle, order=0, reshape=False)
        mask = ndimage.rotate(mask, angle, order=0, reshape=False)
        return img, mask

    def __len__(self):
        return len(self.data)