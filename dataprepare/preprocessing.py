# -- coding: utf-8 --
import h5py
import numpy as np
import glob
import cv2

# Parameters
height = 256  # Enter the image size of the model.
width = 256   # Enter the image size of the model.
channels = 3  # Number of image channels

# Prepare your data set
Tr_list = glob.glob("training/image/*.png")   # Images storage folder
Ms_list = glob.glob("training/mask/*.png")   # Masks storage folder
train_number = int(len(Tr_list) * 0.8)  # Randomly assign the number of images for generating the training set.
val_number = int(len(Tr_list) * 0.2)     # Randomly assign the number of images for generating the validation set.
test_number = int(len(Tr_list) * 0.1)    # Randomly assign the number of images for generating the test set.
all = len(Tr_list)

Data_train_2018 = np.zeros([all, height, width, channels])
Label_train_2018 = np.zeros([all, height, width])

print(f'Start reading {len(Tr_list)} images...')
for idx in range(len(Tr_list)):
    print(idx+1, end = ' ')
    # 讀取並調整圖片大小
    img = cv2.imread(Tr_list[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換為 RGB
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    Data_train_2018[idx] = img.astype(np.float64)  # 轉換為 double

    # 讀取遮罩
    # b = Tr_list[idx]
    # b = b[len(b)-8: len(b)-4]
    # add = f"masks/{b}.png"  # Masks storage folder
    # img2 = cv2.imread(add, cv2.IMREAD_GRAYSCALE)  # 以灰階方式讀取遮罩
    img2 = cv2.imread(Ms_list[idx], cv2.IMREAD_GRAYSCALE)  # 以灰階方式讀取遮罩
    img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_LINEAR)
    Label_train_2018[idx] = img2.astype(np.float64)  # 轉換為 double

print('\nReading dataset finished.')

# Make the training, validation and test sets
Train_img = Data_train_2018[0:train_number]
Validation_img = Data_train_2018[train_number:train_number+val_number]
# Test_img = Data_train_2018[train_number+val_number:all]
Test_img = Data_train_2018[0:all]

Train_mask = Label_train_2018[0:train_number]
Validation_mask = Label_train_2018[train_number:train_number+val_number]
# Test_mask = Label_train_2018[train_number+val_number:all]
Test_mask = Label_train_2018[0:all]

# 儲存資料
np.save('data/data_train', Train_img)
np.save('data/data_test', Test_img)
np.save('data/data_val', Validation_img)

np.save('data/mask_train', Train_mask)
np.save('data/mask_test', Test_mask)
np.save('data/mask_val', Validation_mask)