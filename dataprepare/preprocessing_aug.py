import h5py
import numpy as np
import glob
import cv2
from scipy import ndimage
import random
from PIL import Image
import albumentations as A
import os


def create_augmentations():

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            # Use ColorJitter to replace "RandomBrightness"
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
            A.RandomGamma(p=1),
        ], p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=1),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
        ], p=0.4),
        # 可以添加一些特別針對水體的增強
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        # 添加一些可能有助於水體識別的轉換
        A.OneOf([
            A.Sharpen(p=1),
            A.MotionBlur(p=1),
            A.MedianBlur(blur_limit=3, p=1)
        ], p=0.3),
    ])
    return transform

def augment_image(image, mask, transform, num_augment=5):
    augmented_images = [image]
    augmented_masks = [mask]
    
    for _ in range(num_augment):
        augmented = transform(image=image, mask=mask)
        augmented_images.append(augmented['image'])
        augmented_masks.append(augmented['mask'])
    
    return augmented_images, augmented_masks

########## Parameters ##########
height = 256
width = 256
channels = 3
num_augment = 5  # 每張圖片增強的次數

# Prepare your data set
Tr_list = glob.glob("training/image/*.png")
Ms_list = glob.glob("training/mask/*.png") 
Test_list = glob.glob("testing/image/*.[jJ][pP][gG]")
Test_mask_list = glob.glob("testing/mask/*.png")
train_number = int(len(Tr_list) * 0.8)
val_number = int(len(Tr_list) * 0.2)
test_number = len(Test_list)

print("Test_list", Test_list)

# Calculate size and init. np.array
augmented_train_size = train_number * (num_augment + 1)  # 原圖 + 增強圖
augmented_val_size = val_number
augmented_test_size = test_number
Data_train_aug = np.zeros([augmented_train_size, height, width, channels])
Label_train_aug = np.zeros([augmented_train_size, height, width])
Data_val = np.zeros([augmented_val_size, height, width, channels])
Label_val = np.zeros([augmented_val_size, height, width])
Data_test = np.zeros([augmented_test_size, height, width, channels])
Label_test = np.zeros([augmented_test_size, height, width])

transform = create_augmentations()

print(f'Start reading and augmenting {len(Tr_list)} images...')
aug_idx = 0

# Training set
for idx in range(train_number):
    print(f'Processing training image {idx+1}/{train_number}')

    img = cv2.imread(Tr_list[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    
    mask = cv2.imread(Ms_list[idx], cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height))
    
    aug_images, aug_masks = augment_image(img, mask, transform, num_augment)
    
    for i, (aug_img, aug_mask) in enumerate(zip(aug_images, aug_masks)):
        Data_train_aug[aug_idx] = aug_img.astype(np.float64)
        Label_train_aug[aug_idx] = aug_mask.astype(np.float64)
        aug_idx += 1

print("Items in Data_train_aug:", aug_idx)

# Validation set
for idx in range(val_number):
    val_idx = train_number + idx
    print(f'Processing validation image {idx+1}/{val_number}')
    
    img = cv2.imread(Tr_list[val_idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    Data_val[idx] = img.astype(np.float64)
    
    mask = cv2.imread(Ms_list[val_idx], cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height))
    Label_val[idx] = mask.astype(np.float64)

# Testing set
for idx in range(test_number):
    print(f'Processing test image {idx+1}/{test_number}')
    
    img = cv2.imread(Test_list[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    Data_test[idx] = img.astype(np.float64)
    
    mask = cv2.imread(Test_mask_list[idx], cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height))
    Label_test[idx] = mask.astype(np.float64)

print('\nAugmentation finished.')

if not os.path.exists("data/"):
    os.makedirs("data/")
np.save('data/data_train', Data_train_aug)
np.save('data/data_val', Data_val)
np.save('data/data_test', Data_test)
np.save('data/mask_train', Label_train_aug)
np.save('data/mask_val', Label_val)
np.save('data/mask_test', Label_test)

print(f'Saved augmented dataset with {augmented_train_size} training samples.')



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