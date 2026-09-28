# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
import logging

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


def _load_checkpoint_file(checkpoint_path):
    try:
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # For older torch versions without the weights_only argument.
        return torch.load(checkpoint_path, map_location='cpu')


def _build_torchvision_resnet18():
    try:
        from torchvision.models import ResNet18_Weights, resnet18
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        from torchvision.models import resnet18
        return resnet18(pretrained=True)


def _copy_prefix_weights(src_state, dst_state, src_prefix, dst_prefix):
    loaded = {}
    for key, value in src_state.items():
        if not key.startswith(src_prefix):
            continue
        suffix = key[len(src_prefix):]
        target_key = dst_prefix + suffix
        if target_key in dst_state and dst_state[target_key].shape == value.shape:
            loaded[target_key] = value
    return loaded


def _normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ('module.', 'model.'):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        normalized[new_key] = value
    return normalized


def _adapt_stem_input_weight(weight, target_shape):
    """Adapt a three-channel pretrained stem to a four-channel RGB-D stem."""
    if tuple(weight.shape) == tuple(target_shape):
        return weight
    if (
        weight.ndim == 4
        and len(target_shape) == 4
        and weight.shape[0] == target_shape[0]
        and weight.shape[1] == 3
        and target_shape[1] == 4
        and tuple(weight.shape[2:]) == tuple(target_shape[2:])
    ):
        adapted = weight.new_zeros(target_shape)
        adapted[:, :3] = weight
        adapted[:, 3:4] = weight.mean(dim=1, keepdim=True)
        return adapted
    return weight


def load_imagenet_backbone(model):
    resnet = _build_torchvision_resnet18()
    src_state = resnet.state_dict()
    dst_state = model.state_dict()
    loaded = {}

    # PIDNet-S uses a lighter two-3x3 stem, so only the first conv/bn can be partially initialized.
    stem_conv_key = 'conv1.0.weight'
    if stem_conv_key in dst_state and 'conv1.weight' in src_state:
        dst_shape = dst_state[stem_conv_key].shape
        src_weight = src_state['conv1.weight']
        center = src_weight[:, :, 2:5, 2:5]
        stem_weight = center[:dst_shape[0], :, :, :].contiguous()
        stem_weight = _adapt_stem_input_weight(stem_weight, dst_shape)
        if stem_weight.shape == dst_shape:
            loaded[stem_conv_key] = stem_weight

    for stat_name in ['weight', 'bias', 'running_mean', 'running_var']:
        dst_bn_key = f'conv1.1.{stat_name}'
        src_bn_key = f'bn1.{stat_name}'
        if dst_bn_key in dst_state and src_bn_key in src_state:
            loaded[dst_bn_key] = src_state[src_bn_key][:dst_state[dst_bn_key].shape[0]].contiguous()

    # Map the closest matching ResNet stages into the compatible PIDNet backbone stages.
    stage_pairs = [
        ('layer1.', 'layer3_.'),
        ('layer2.', 'layer3.'),
        ('layer3.', 'layer4.'),
        ('layer1.', 'layer4_.'),
    ]
    for src_prefix, dst_prefix in stage_pairs:
        loaded.update(_copy_prefix_weights(src_state, dst_state, src_prefix, dst_prefix))

    dst_state.update(loaded)
    model.load_state_dict(dst_state, strict=False)

    logging.info('Attention!!!')
    logging.info('Loaded {} ImageNet backbone tensors from torchvision ResNet18.'.format(len(loaded)))
    logging.info('Over!!!')
    return model



class PIDNet(nn.Module):

    # RGB-only signature kept for reference:
    # def __init__(self, m=2, n=3, num_classes=19, planes=64,
    #              ppm_planes=96, head_planes=128, augment=True):
    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96,
                 head_planes=128, augment=True, in_channels=3):
        super(PIDNet, self).__init__()
        self.augment = augment
        
        # I Branch
        self.conv1 =  nn.Sequential(
                          # RGB-only stem kept for reference:
                          # nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          nn.Conv2d(in_channels,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
        
        # P Branch
        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
                                          BatchNorm2d(planes * 2, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
                                          BatchNorm2d(planes * 2, momentum=bn_mom),
                                          )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        
        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                                        BatchNorm2d(planes, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                                        # BatchNorm2d(planes * 2, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                    #  BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)
            
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        
        # Prediction Head
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)           

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)
    
    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)
        
        return layer

    def forward(self, x):
        torch.cuda.nvtx.range_push("stage1")
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("stage2")
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)
        
        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(
                        self.diff3(x),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("stage3")
        if self.augment:
            temp_p = x_
        
        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))
        
        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(
                        self.diff4(x),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("stage4")
        if self.augment:
            temp_d = x_d
            
        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("stage5")
        x = F.interpolate(
                        self.spp(self.layer5(x)),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)

        x_ = self.final_layer(self.dfm(x_, x, x_d))
        torch.cuda.nvtx.range_pop()
        if self.augment: 
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_      

def get_seg_model(cfg, imgnet_pretrained):
    in_channels = getattr(cfg.MODEL, 'IN_CHANNELS', 3)
    if 's' in cfg.MODEL.NAME:
        # RGB-only construction kept for reference:
        # model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, ppm_planes=96, head_planes=128, augment=True)
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, ppm_planes=96, head_planes=128, augment=True, in_channels=in_channels)
    elif 'm' in cfg.MODEL.NAME:
        # RGB-only construction kept for reference:
        # model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=96, head_planes=128, augment=True)
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=96, head_planes=128, augment=True, in_channels=in_channels)
    else:
        # RGB-only construction kept for reference:
        # model = PIDNet(m=3, n=4, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=112, head_planes=256, augment=True)
        model = PIDNet(m=3, n=4, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=112, head_planes=256, augment=True, in_channels=in_channels)

    if cfg.MODEL.PRETRAINED == 'torchvision://resnet18':
        return load_imagenet_backbone(model)

    if not cfg.MODEL.PRETRAINED or not os.path.isfile(cfg.MODEL.PRETRAINED):
        logging.info('Attention!!!')
        logging.info('No pretrained weights found, training will start from random initialization.')
        logging.info('Over!!!')
        return model

    checkpoint = _load_checkpoint_file(cfg.MODEL.PRETRAINED)

    if imgnet_pretrained:
        pretrained_state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model_dict = model.state_dict()
        stem_key = 'conv1.0.weight'
        if stem_key in pretrained_state and stem_key in model_dict:
            pretrained_state[stem_key] = _adapt_stem_input_weight(
                pretrained_state[stem_key], model_dict[stem_key].shape
            )
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model.load_state_dict(model_dict, strict = False)
    else:
        pretrained_dict = checkpoint
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        pretrained_dict = _normalize_state_dict_keys(pretrained_dict)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict = False)
    
    return model

# RGB-only signature kept for reference:
# def get_pred_model(name, num_classes):
def get_pred_model(name, num_classes, in_channels=3):
    # RGB-only calls can omit in_channels; RGB-D callers pass in_channels=4.
    if 's' in name:
        # model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False, in_channels=in_channels)
    elif 'm' in name:
        # model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False)
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False, in_channels=in_channels)
    else:
        # model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False)
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False, in_channels=in_channels)
    
    return model

if __name__ == '__main__':
    
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    model = get_pred_model(name='pidnet_s', num_classes=19)
    model.eval()
    model.to(device)
    iterations = None
    
    input = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
    
    
    


