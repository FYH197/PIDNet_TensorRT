"""Reproducible DAD integration checks, not accuracy evaluation.

Run from repo root: python -B tests/verify_dad.py --output <directory>
Use --benchmark for short FP32 batch-44 synthetic training timings.
"""
import argparse
import copy
import gc
import json
import statistics
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from configs import config
from models.pidnet import PIDNet, get_seg_model, get_pred_model
from export_indoor_onnx import load_checkpoint_weights
from utils.criterion import OhemCrossEntropy, BondaryLoss
from utils.utils import FullModel


def make(enabled, augment=True, channels=16):
    return PIDNet(m=2, n=3, num_classes=15, planes=32, ppm_planes=96,
                  head_planes=128, augment=augment, in_channels=4,
                  dad_enabled=enabled, dad_channels=channels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    result = {}
    base_cfg = config.clone()
    base_cfg.merge_from_file(str(ROOT / 'configs/yaml/pidnet_s_sunrgbd15_rgbd_4227_1057.yaml'))
    config.merge_from_file(str(ROOT / 'configs/yaml/pidnet_s_sunrgbd15_rgbd_dad_4227_1057.yaml'))
    compare = config.clone()
    compare.MODEL.DAD_ENABLED = False
    assert compare == base_cfg, 'Protocol differs beyond DAD switch'
    result['config_protocol_equal'] = True
    torch.manual_seed(304)
    baseline = make(False).eval()
    base_rng = torch.get_rng_state()
    torch.manual_seed(304)
    dad = make(True).eval()
    assert torch.equal(base_rng, torch.get_rng_state())
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, dad.state_dict()[key]), key
    result['shared_initialization_and_rng_equal'] = True
    x = torch.randn(1, 4, 480, 640)
    with torch.no_grad():
        a, b = baseline(x), dad(x)
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    result['zero_residual_outputs_exact_equal'] = True
    result['output_shapes'] = [list(v.shape) for v in b]
    result['parameters'] = {'baseline': sum(p.numel() for p in baseline.parameters()),
                            'dad': sum(p.numel() for p in dad.parameters())}
    dad.train()
    optimizer = torch.optim.SGD(dad.parameters(), lr=0.01, momentum=0.9)
    xx = torch.randn(2, 4, 64, 64)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        edge = dad(xx)[-1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(edge, torch.zeros_like(edge))
        loss.backward()
        assert torch.isfinite(loss)
        for name, p in dad.dad.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            if step == 1:
                assert p.grad.abs().sum() > 0, name
        optimizer.step()
    result['boundary_only_two_step_gradients'] = True
    saved = {'state_dict': {'model.' + k: v for k, v in dad.state_dict().items()},
             'optimizer': copy.deepcopy(optimizer.state_dict())}
    checkpoint = args.output / 'dad_smoke.pth'
    torch.save(saved, checkpoint)
    resumed = make(True)
    resumed.load_state_dict(dad.state_dict(), strict=True)
    restored_optimizer = torch.optim.SGD(resumed.parameters(), lr=0.01, momentum=0.9)
    restored_optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    for model, opt in [(dad, optimizer), (resumed, restored_optimizer)]:
        model.train()
        opt.zero_grad(set_to_none=True)
        model(xx)[-1].square().mean().backward()
        opt.step()
    for key, value in dad.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key
    result['strict_resume_next_step_equal'] = True
    pred = get_pred_model('pidnet-s', 15, in_channels=4, dad_enabled=True)
    load_checkpoint_weights(pred, checkpoint)
    for enabled, channels in [(False, 16), (True, 8)]:
        try:
            load_checkpoint_weights(make(enabled, False, channels), checkpoint)
        except RuntimeError as exc:
            assert 'DAD checkpoint mismatch' in str(exc)
        else:
            raise AssertionError('Mismatched checkpoint was accepted')
    base_checkpoint = args.output / 'baseline_smoke.pth'
    torch.save(baseline.state_dict(), base_checkpoint)
    try:
        load_checkpoint_weights(make(True, False), base_checkpoint)
    except RuntimeError as exc:
        assert 'DAD checkpoint mismatch' in str(exc)
    else:
        raise AssertionError('Baseline checkpoint accepted for DAD inference')
    result['checkpoint_mismatch_rejected'] = True
    incomplete = dict(saved['state_dict'])
    incomplete.pop('model.conv1.0.weight')
    partial_path = args.output / 'incomplete_dad_smoke.pth'
    torch.save(incomplete, partial_path)
    try:
        load_checkpoint_weights(make(True, False), partial_path)
    except RuntimeError as exc:
        assert 'Incomplete DAD inference checkpoint' in str(exc)
    else:
        raise AssertionError('Incomplete DAD checkpoint accepted')
    result['incomplete_checkpoint_rejected'] = True
    torch.manual_seed(304)
    pretrained_base = get_seg_model(base_cfg, True)
    torch.manual_seed(304)
    pretrained_dad = get_seg_model(config, True)
    for key, value in pretrained_base.state_dict().items():
        assert torch.equal(value, pretrained_dad.state_dict()[key]), key
    result['pretrained_shared_weights_equal'] = True
    pred.eval()
    with torch.no_grad():
        assert pred(x).shape == (1, 15, 60, 80)
    (args.output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if args.benchmark:
        assert torch.cuda.is_available()
        result['gpu'] = torch.cuda.get_device_name()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        result['benchmark'] = []
        for enabled in [False, True, True, False]:
            torch.manual_seed(304)
            net = get_seg_model(config if enabled else base_cfg, True).cuda()
            full = FullModel(net, OhemCrossEntropy(ignore_label=255, thres=0.9,
                             min_kept=131072), BondaryLoss()).cuda().train()
            opt = torch.optim.SGD(full.parameters(), lr=0.01, momentum=0.9,
                                   weight_decay=0.0005)
            inputs = torch.randn(44, 4, 480, 640, device='cuda')
            labels = torch.randint(0, 15, (44, 480, 640), device='cuda')
            edges = (torch.rand(44, 480, 640, device='cuda') > 0.9).float()
            times = []
            torch.cuda.reset_peak_memory_stats()
            for step in range(args.warmup + args.steps):
                torch.cuda.synchronize()
                start = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss = full(inputs, labels, edges)[0].mean()
                loss.backward()
                opt.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                assert torch.isfinite(loss)
                if step >= args.warmup:
                    times.append(elapsed)
            item = {'dad': enabled, 'batch': 44, 'precision': 'FP32',
                    'warmup_steps': args.warmup, 'measured_steps': args.steps,
                    'seconds': times, 'median_seconds': statistics.median(times),
                    'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                    'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
            result['benchmark'].append(item)
            print(json.dumps(item), flush=True)
            (args.output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            del full, net, opt, inputs, labels, edges, loss
            gc.collect()
            torch.cuda.empty_cache()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
