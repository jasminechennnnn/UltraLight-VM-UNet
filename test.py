import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

from models.UltraLight_VM_UNet import UltraLight_VM_UNet
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    resume_model = 'results/20241230_143745_UltraLight_VM_UNet_DIP/checkpoints/best-epoch80-loss0.3487.pth' # case 1
    # resume_model = 'results/20241230_172541_UltraLight_VM_UNet_DIP/checkpoints/best-epoch86-loss0.8725.pth' # case 2
    # resume_model = 'results/20241230_181723_UltraLight_VM_UNet_DIP/checkpoints/best-epoch110-loss1.1885.pth' # case 3
    
    config.work_dir = os.path.dirname(os.path.dirname(resume_model))
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    # checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    # resume_model = os.path.join('')
    outputs = os.path.join(config.work_dir, 'outputs')
    print(f"output path: {outputs}\n")
    demo = os.path.join(config.work_dir, 'demo')
    print(f"demo path: {demo}\n")
    
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)
    if not os.path.exists(demo):
        os.makedirs(demo)

    global logger
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)

    logger.info('='*50)
    logger.info('Testing Configuration:')
    logger.info(f'Resume Model Path: {resume_model}')
    logger.info(f'Testing with checkpoint: {resume_model.split("/")[-1]}')
    logger.info('='*50)


    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()
    


    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config    
    model = UltraLight_VM_UNet(num_classes=model_cfg['num_classes'], 
                               input_channels=model_cfg['input_channels'], 
                               c_list=model_cfg['c_list'], 
                               split_att=model_cfg['split_att'], 
                               bridge=model_cfg['bridge'],)
    
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])


    print('#----------Preparing dataset----------#')
    test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
    # test_dataset = WaterSegLoader(path_Data = config.data_path, train = False, Test = True, logger = logger)
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()




    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    print('#----------Testing----------#')
    best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
    model.module.load_state_dict(best_weight)
    threshold_list = [config.threshold]
    for thresh in threshold_list:
        print(f"\nTesting with threshold = {thresh}")
        config.threshold = thresh
        loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
            visualize=False
        )

if __name__ == '__main__':
    config = setting_config
    main(config)