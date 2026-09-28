import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import build_run_name, create_logger  # noqa: E402


def make_config(root):
    return SimpleNamespace(
        OUTPUT_DIR=str(root / 'output'),
        LOG_DIR=str(root / 'log'),
        MODEL=SimpleNamespace(
            NAME='pidnet_s',
            PRETRAINED='pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar',
        ),
        DATASET=SimpleNamespace(
            DATASET='indoor',
            ROOT='data/indoor',
        ),
        TRAIN=SimpleNamespace(LR=0.01),
    )


class RunOutputTests(unittest.TestCase):
    def test_generated_name_is_short_and_informative(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = make_config(Path(temp_dir))
            name = build_run_name(
                cfg, 'pidnet_s_sunrgbd15_official_strong.yaml', seed=304)
            self.assertEqual(name, 'pidnet-s_sun15_img_s304_lr0p01')

    def test_each_training_run_gets_an_isolated_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                cfg = make_config(Path(temp_dir))
                first, first_logs, _ = self._create(cfg)
                second, second_logs, _ = self._create(cfg)

                self.assertNotEqual(first, second)
                self.assertEqual(Path(first_logs), Path(first) / 'logs')
                self.assertEqual(Path(second_logs), Path(second) / 'logs')
                self.assertTrue((Path(first) / 'logs' / 'train.log').exists())
                self.assertTrue((Path(second) / 'logs' / 'train.log').exists())
            finally:
                logging.shutdown()

    @staticmethod
    def _create(cfg):
        logger, output_dir, log_dir = create_logger(
            cfg,
            'pidnet_s_sunrgbd15_official_strong.yaml',
            phase='train',
            seed=304,
        )
        for handler in logger.handlers:
            handler.flush()
        return output_dir, log_dir, logger


if __name__ == '__main__':
    unittest.main()
