<div align="center">
<h1>UltraLight VM-UNet </h1>
<h3>Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation</h3>

Renkai Wu<sup>1</sup>, Yinghao Liu<sup>2</sup>, Pengchen Liang<sup>1</sup>\*, Qing Chang<sup>1</sup>\*

<sup>1</sup>  Shanghai University, <sup>2</sup>  University of Shanghai for Science and Technology


ArXiv Preprint ([arXiv:2403.20035](https://arxiv.org/abs/2403.20035))


</div>


## 0. Environment
The environment can be set up following [VM-UNet](https://github.com/JCruan519/VM-UNet)'s installation procedure, or using the steps below (Python 3.8):
```bash
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0

# Note: For the following packages, please ensure you have the correct wheel files
pip install causal_conv1d==1.0.0  # requires: causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1      # requires: mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs h5py opencv-python scipy pillow albumentations tqdm
```

## 1. Data Preprocessing
1. Unzip the dataset:
```bash
chmod +x prepare_data.sh
./prepare_data.sh
```

2. Preprocess and augment data:
```bash
cd dataprepare/
python preprocessing_aug.py --num-augment <number>  # Replace <number> with desired augmentation count
cd ../
```

## 2. Training
Train the UltraLight VM-UNet:
```bash
python train.py
```
The training outputs will be saved in the `results/` directory.

## 3. Testing
1. Configure the model checkpoint:
   - Open `test.py`
   - Update the `resume_model` parameter with your checkpoint path

2. Run the test:
```bash
python test.py
```
Test results will be saved in the `results/` directory.

---

## 🔥🔥Highlights🔥🔥
### *1.The UltraLight VM-UNet has only 0.049M parameters, 0.060 GFLOPs, and a model weight file of only 229.1 KB.*</br>
### *2.Parallel Vision Mamba (or Mamba) is a winner for lightweight models.*</br>

## Additional information
PVM Layer can be very simply embedded into any model to reduce the overall parameters of the model. Please refer to [issue 7](https://github.com/wurenkai/UltraLight-VM-UNet/issues/7) for the methodology of calculating model parameters and GFLOPs. In addition to the above operations, the exact GFLOPs calculation still requires the addition of the SSM values due to the specific nature of SSM. Refer to [here](https://github.com/state-spaces/mamba/issues/110#issuecomment-1919470069) for details. However, due to the small number of UltraLight VM-UNet channels, the addition of all the SSM values has almost no effect on the results of the GFLOPs obtained through the operations described above (3 valid digits).

### Abstract
Traditionally for improving the segmentation performance of models, most approaches prefer to use adding more complex modules. And this is not suitable for the medical field, especially for mobile medical devices, where computationally loaded models are not suitable for real clinical environments due to computational resource constraints. Recently, state-space models (SSMs), represented by Mamba, have become a strong competitor to traditional CNNs and Transformers. In this paper, we deeply explore the key elements of parameter influence in Mamba and propose an UltraLight Vision Mamba UNet (UltraLight VM-UNet) based on this. Specifically, we propose a method for processing features in parallel Vision Mamba, named PVM Layer, which achieves excellent performance with the lowest computational load while keeping the overall number of processing channels constant. We conducted comparisons and ablation experiments with several state-of-the-art lightweight models on three skin lesion public datasets and demonstrated that the UltraLight VM-UNet exhibits the same strong performance competitiveness with parameters of only 0.049M and GFLOPs of 0.060. In addition, this study deeply explores the key elements of parameter influence in Mamba, which will lay a theoretical foundation for Mamba to possibly become a new mainstream module for lightweighting in the future.

### Different Parallel Vision Mamba （PVM Layer） settings:
| Setting | Briefly | Params | GFLOPs | DSC |
| --- | --- | --- | --- | --- |
| 1 | No paralleling ( Channel number ```C```) | 0.136M | 0.060 | 0.9069 |
| 2 | Double parallel ( Channel number ```(C/2)+(C/2)```) | 0.070M | 0.060 |  0.9073 |
| 3 | Quadruple parallel ( Channel number ```(C/4)+(C/4)+(C/4)+(C/4)```) | 0.049M | 0.060 | 0.9091 |


### Citation
If you find this repository helpful, please consider citing: </br>
```
@article{wu2024ultralight,
  title={UltraLight VM-UNet: Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation},
  author={Wu, Renkai and Liu, Yinghao and Liang, Pengchen and Chang, Qing},
  journal={arXiv preprint arXiv:2403.20035},
  year={2024}
}
```

### Acknowledgement
Thanks to [Vim](https://github.com/hustvl/Vim), [VMamba](https://github.com/MzeroMiko/VMamba), [VM-UNet](https://github.com/JCruan519/VM-UNet) and [LightM-UNet](https://github.com/MrBlankness/LightM-UNet) for their outstanding work.
