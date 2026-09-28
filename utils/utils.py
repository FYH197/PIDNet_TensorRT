# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import re
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config

class FullModel(nn.Module):

  def __init__(self, model, sem_loss, bd_loss):
    super(FullModel, self).__init__()
    self.model = model
    self.sem_loss = sem_loss
    self.bd_loss = bd_loss

  def pixel_acc(self, pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

  def forward(self, inputs, labels, bd_gt, *args, **kwargs):
    
    outputs = self.model(inputs, *args, **kwargs)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    
    h, w = labels.size(1), labels.size(2)
    ph, pw = outputs[0].size(2), outputs[0].size(3)
    if ph != h or pw != w:
        for i in range(len(outputs)):
            outputs[i] = F.interpolate(outputs[i], size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

    sem_outputs = outputs[:-1] if len(outputs) > 1 else outputs
    acc = self.pixel_acc(sem_outputs[-1], labels)
    loss_s = self.sem_loss(sem_outputs, labels)

    if self.bd_loss is None:
        zero = loss_s * 0.0
        return torch.unsqueeze(loss_s, 0), sem_outputs, acc, [loss_s, zero]

    loss_b = self.bd_loss(outputs[-1], bd_gt)

    filler = torch.ones_like(labels) * config.TRAIN.IGNORE_LABEL
    bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:])>0.8, labels, filler)
    loss_sb = self.sem_loss(outputs[-2], bd_label)
    loss = loss_s + loss_b + loss_sb

    return torch.unsqueeze(loss,0), sem_outputs, acc, [loss_s, loss_b]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def _slug(value):
    value = re.sub(r'[^a-zA-Z0-9._-]+', '-', str(value).strip())
    return value.strip('-_.').lower()


def _format_learning_rate(value):
    return ('{:g}'.format(float(value))).replace('.', 'p').replace('-', 'm')


def build_run_name(cfg, cfg_name, seed=None):
    """Build a concise run name from the experiment's stable identity."""
    model = _slug(cfg.MODEL.NAME).replace('_', '-')
    cfg_stem = _slug(Path(cfg_name).stem)
    dataset = 'sun15' if 'sunrgbd15' in cfg_stem else _slug(cfg.DATASET.DATASET)
    input_suffix = '_rgbd' if getattr(cfg.MODEL, 'IN_CHANNELS', 3) == 4 else ''
    initialization = 'img' if getattr(cfg.MODEL, 'PRETRAINED', '') else 'scratch'
    seed_part = 's{}'.format(seed) if seed is not None and seed > 0 else 'sna'
    lr_part = 'lr{}'.format(_format_learning_rate(cfg.TRAIN.LR))
    # RGB-only run name kept for reference:
    # return '{}_{}_{}_{}_{}'.format(
    #     model, dataset, initialization, seed_part, lr_part)
    return '{}_{}{}_{}_{}_{}'.format(
        model, dataset, input_suffix, initialization, seed_part, lr_part)


def _unique_run_dir(parent, run_name, timestamp):
    candidate = parent / '{}_{}'.format(run_name, timestamp)
    suffix = 2
    while candidate.exists():
        candidate = parent / '{}_{}-{:02d}'.format(run_name, timestamp, suffix)
        suffix += 1
    return candidate


def create_logger(cfg, cfg_name, phase='train', run_name=None, seed=None,
                  run_dir=None):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    root_log_dir = Path(cfg.LOG_DIR)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    root_log_dir.mkdir(parents=True, exist_ok=True)

    dataset = cfg.DATASET.DATASET
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    data_root = str(getattr(cfg.DATASET, 'ROOT', '')).lower()
    is_datamix = (
        ('indoor_mix' in cfg_name.lower())
        or ('datamix' in cfg_name.lower())
        or ('indoor_mix' in data_root)
        or ('datamix' in data_root)
    )

    time_str = time.strftime('%Y%m%d-%H%M%S')
    if run_dir:
        final_output_dir = Path(run_dir).expanduser()
        if not final_output_dir.is_absolute():
            final_output_dir = root_output_dir / final_output_dir
    elif phase == 'train':
        run_name = _slug(run_name) if run_name else build_run_name(
            cfg, cfg_name, seed)
        category = 'datamix' if is_datamix else dataset
        final_output_dir = _unique_run_dir(
            root_output_dir / category, run_name, time_str)
    else:
        # Keep evaluation compatible with legacy output locations. New runs
        # normally pass TEST.MODEL_FILE explicitly.
        final_output_dir = root_output_dir / dataset / cfg_name

    final_output_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir = final_output_dir / 'logs'
    run_log_dir.mkdir(parents=True, exist_ok=True)

    final_log_file = run_log_dir / '{}.log'.format(phase)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head,
                        force=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger, str(final_output_dir), str(run_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.int64)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int64)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = seg_gt * num_class + seg_pred
    confusion_matrix = np.bincount(
        index,
        minlength=num_class * num_class
    ).reshape(num_class, num_class)
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr
