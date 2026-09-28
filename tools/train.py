# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import csv
import json
import os
import pprint
import random

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import train, validate
from utils.utils import create_logger, FullModel


def seed_worker(worker_id):
    """Seed Python and NumPy RNGs inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def should_validate_epoch(epoch, real_end, interval, force=False):
    """Return True after each complete validation interval and on the final epoch."""
    validate_every = max(1, int(interval))
    return force or ((epoch + 1) % validate_every == 0) or (epoch == real_end - 1)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def estimate_flops(module, input_shape, device):
    total_flops = {'value': 0}
    handles = []
    was_training = module.training

    def conv_hook(layer, inputs, output):
        out = output[0] if isinstance(output, (list, tuple)) else output
        batch_size, out_channels, out_h, out_w = out.shape
        kernel_h, kernel_w = layer.kernel_size
        in_channels = layer.in_channels
        groups = layer.groups
        kernel_mul = kernel_h * kernel_w * (in_channels // groups)
        bias_ops = 1 if layer.bias is not None else 0
        total_flops['value'] += batch_size * out_h * out_w * out_channels * (kernel_mul + bias_ops)

    def linear_hook(layer, inputs, output):
        in_tensor = inputs[0]
        batch_size = in_tensor.shape[0] if in_tensor.dim() > 1 else 1
        bias_ops = layer.out_features if layer.bias is not None else 0
        total_flops['value'] += batch_size * (layer.in_features * layer.out_features + bias_ops)

    def bn_hook(layer, inputs, output):
        in_tensor = inputs[0]
        total_flops['value'] += in_tensor.numel() * 2

    def relu_hook(layer, inputs, output):
        in_tensor = inputs[0]
        total_flops['value'] += in_tensor.numel()

    def pool_hook(layer, inputs, output):
        out = output[0] if isinstance(output, (list, tuple)) else output
        if isinstance(layer.kernel_size, tuple):
            kernel_h, kernel_w = layer.kernel_size
        else:
            kernel_h = kernel_w = layer.kernel_size
        total_flops['value'] += out.numel() * kernel_h * kernel_w

    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, nn.Linear):
            handles.append(layer.register_forward_hook(linear_hook))
        elif isinstance(layer, nn.BatchNorm2d):
            handles.append(layer.register_forward_hook(bn_hook))
        elif isinstance(layer, nn.ReLU):
            handles.append(layer.register_forward_hook(relu_hook))
        elif isinstance(layer, (nn.AvgPool2d, nn.MaxPool2d)):
            handles.append(layer.register_forward_hook(pool_hook))

    module.eval()
    with torch.no_grad():
        dummy = torch.randn(*input_shape, device=device)
        module(dummy)

    for handle in handles:
        handle.remove()

    if was_training:
        module.train()

    return total_flops['value']


def update_experiment_index(run_dir, metrics):
    """Insert or replace one run in the dataset-level experiment index."""
    index_path = os.path.join(os.path.dirname(run_dir), 'experiments.csv')
    fields = [
        'run_id', 'model', 'dataset', 'seed', 'learning_rate', 'epochs',
        'best_epoch', 'best_val_miou', 'final_val_miou', 'test_miou',
        'status',
    ]
    rows = []
    if os.path.isfile(index_path):
        with open(index_path, 'r', encoding='utf-8-sig', newline='') as source:
            rows = list(csv.DictReader(source))
    row = {field: metrics.get(field) for field in fields}
    rows = [item for item in rows if item.get('run_id') != metrics['run_id']]
    rows.append(row)
    temp_path = index_path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8-sig', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, index_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # Previous RGB-only default kept for reference:
                        # default=r"f:\PIDnet_RT\PIDNet_TensorRT\configs\yaml\pidnet_s_sunrgbd15_paper_4227_1057.yaml",
                        default=r"f:\PIDnet_RT\PIDNet_TensorRT\configs\yaml\pidnet_s_sunrgbd15_rgbd_4227_1057.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('--run-name', type=str, default=None,
                        help='Optional concise run name; timestamp is appended automatically')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Existing run directory used for resume')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    if args.run_dir and not config.TRAIN.RESUME:
        raise ValueError('--run-dir is reserved for TRAIN.RESUME=true runs')
    if config.TRAIN.RESUME and not args.run_dir:
        raise ValueError('TRAIN.RESUME=true requires --run-dir to identify the run')

    if args.seed > 0:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train', run_name=args.run_name,
        seed=args.seed, run_dir=args.run_dir)

    with open(os.path.join(final_output_dir, 'config.yaml'), 'w',
              encoding='utf-8') as config_file:
        config_file.write(config.dump())

    logger.info(pprint.pformat(args))
    logger.info(config)
    logger.info('Checkpoint directory: %s', final_output_dir)
    logger.info('Training log directory: %s', tb_log_dir)
    logger.info(
        'Model initialization: %s',
        config.MODEL.PRETRAINED if config.MODEL.PRETRAINED else 'random initialization',
    )
    logger.info('Model input channels: %d', config.MODEL.IN_CHANNELS)
    logger.info('DAD enabled: %s; channels: %d', config.MODEL.DAD_ENABLED, config.MODEL.DAD_CHANNELS)
    logger.info(
        'Validation schedule: every %d epoch(s)',
        max(1, int(config.TEST.VAL_INTERVAL)),
    )

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0
    
    imgnet = 'imagenet' in config.MODEL.PRETRAINED
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)
 
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    test_generator = torch.Generator()
    test_generator.manual_seed(args.seed)
    dataloader_kwargs = {}
    if config.WORKERS > 0:
        dataloader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        scale_factor=config.TRAIN.SCALE_FACTOR)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
        **dataloader_kwargs)


    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=test_generator,
        **dataloader_kwargs)

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)

    bd_criterion = BondaryLoss() if config.LOSS.USE_BOUNDARY else None
    
    model = FullModel(model, sem_criterion, bd_criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    if len(trainloader) != epoch_iters:
        raise ValueError('Training loader length changed the fixed update budget')
        
    best_mIoU = 0
    best_epoch = None
    final_valid_loss = None
    final_mean_IoU = None
    final_IoU_array = None
    last_epoch = 0
    flag_rm = config.TRAIN.RESUME
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'last.ckpt')
        if not os.path.isfile(model_state_file):
            model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            best_epoch = checkpoint.get('best_epoch')
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    
    for epoch in range(last_epoch, real_end):

        train(config, epoch, config.TRAIN.END_EPOCH, 
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict)

        did_validate = False
        should_validate = should_validate_epoch(
            epoch,
            real_end,
            config.TEST.VAL_INTERVAL,
            force=flag_rm == 1,
        )
        if should_validate:
            valid_loss, mean_IoU, IoU_array = validate(config, 
                        testloader, model, writer_dict)
            did_validate = True
            final_valid_loss = valid_loss
            final_mean_IoU = mean_IoU
            final_IoU_array = IoU_array
        if flag_rm == 1:
            flag_rm = 0

        if did_validate and mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            best_epoch = epoch + 1

        logger.info('=> saving checkpoint to {}'.format(
            os.path.join(final_output_dir, 'last.ckpt')))
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'best_epoch': best_epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir, 'last.ckpt'))
        if did_validate and mean_IoU >= best_mIoU:
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best_miou.pth'))
        if did_validate:
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                        valid_loss, mean_IoU, best_mIoU)
            logging.info(msg)
            logging.info(IoU_array)



    torch.save(model.module.state_dict(),
            os.path.join(final_output_dir, 'final.pth'))

    stats_model = model.module.model
    stats_input_shape = (
        1,
        # RGB-only channel count kept for reference:
        # 3,
        config.MODEL.IN_CHANNELS,
        config.TEST.IMAGE_SIZE[1],
        config.TEST.IMAGE_SIZE[0],
    )
    param_count = count_parameters(stats_model)
    flops_count = estimate_flops(stats_model, stats_input_shape, torch.device('cuda'))

    if final_mean_IoU is not None:
        summary_msg = (
            'Final Summary | '
            'mIoU: {:.4f} | '
            'valid_loss: {:.4f} | '
            'Params: {:.2f} M | '
            'Approx FLOPs @ {}x{}: {:.2f} GFLOPs'
        ).format(
            final_mean_IoU,
            final_valid_loss,
            param_count / 1e6,
            config.TEST.IMAGE_SIZE[0],
            config.TEST.IMAGE_SIZE[1],
            flops_count / 1e9,
        )
        print(summary_msg)
        logger.info(summary_msg)
        logger.info(final_IoU_array)

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end-start)/3600))

    metrics = {
        'run_id': os.path.basename(os.path.normpath(final_output_dir)),
        'status': 'completed',
        'model': config.MODEL.NAME,
        'input_channels': int(config.MODEL.IN_CHANNELS),
        'dad_enabled': bool(config.MODEL.DAD_ENABLED),
        'dad_channels': int(config.MODEL.DAD_CHANNELS),
        'dataset': config.DATASET.DATASET,
        'train_set': config.DATASET.TRAIN_SET,
        'validation_set': config.DATASET.TEST_SET,
        'seed': args.seed,
        'learning_rate': float(config.TRAIN.LR),
        'epochs': int(real_end),
        'best_epoch': best_epoch,
        'best_val_miou': float(best_mIoU),
        'final_val_miou': (
            float(final_mean_IoU) if final_mean_IoU is not None else None),
        'final_valid_loss': (
            float(final_valid_loss) if final_valid_loss is not None else None),
        'test_miou': None,
        'best_checkpoint': 'best_miou.pth',
        'resume_checkpoint': 'last.ckpt',
    }
    with open(os.path.join(final_output_dir, 'metrics.json'), 'w',
              encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
        metrics_file.write('\n')

    final_text = ('{:.4%}'.format(final_mean_IoU)
                  if final_mean_IoU is not None else 'not available')
    with open(os.path.join(final_output_dir, 'summary.txt'), 'w',
              encoding='utf-8') as summary_file:
        summary_file.write(
            'Run: {run_id}\nStatus: completed\nBest epoch: {best_epoch}\n'
            'Best val mIoU: {best:.4%}\nFinal val mIoU: {final}\n'
            'Independent test: not evaluated\n'.format(
                run_id=metrics['run_id'], best_epoch=best_epoch,
                best=best_mIoU, final=final_text))
    update_experiment_index(final_output_dir, metrics)
    logger.info('Done')

if __name__ == '__main__':
    main()
