# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as BaseExp
from yolox.data import get_yolox_datadir


class Exp(BaseExp):
    def __init__(self, config):
        super().__init__()

        # Model config
        self.num_classes = config.num_classes
        self.depth = config.depth
        self.width = config.width
        self.act = config.act

        # Dataloader configuration
        self.data_num_workers = config.data_num_workers
        self.input_size = config.input_size
        self.multiscale_range = config.multiscale_range
        self.data_dir = config.data_dir
        self.train_ann = config.train_ann
        self.val_ann = config.val_ann
        self.test_ann = config.test_ann
        if hasattr(config, "random_size"):
            self.random_size = config.random_size

        # Transform config
        self.mosaic_prob = config.mosaic_prob
        self.mixup_prob = config.mixup_prob
        self.hsv_prob = config.hsv_prob
        self.flip_prob = config.flip_prob
        self.degrees = config.degrees
        self.translate = config.translate
        self.mosaic_scale = config.mosaic_scale
        self.mixup_scale = config.mixup_scale
        self.shear = config.shear
        self.enable_mixup = config.enable_mixup
        self.legacy = config.legacy
        if self.legacy:
            self.rgb_means = config.rgb_means
            self.std = config.std
        else:
            self.rgb_means = None
            self.std = None

        # Training config
        self.warmup_epochs = config.warmup_epochs
        self.max_epoch = config.max_epoch
        self.warmup_lr = config.warmup_lr
        self.basic_lr_per_img = config.basic_lr_per_img
        self.min_lr_ratio = config.min_lr_ratio
        self.scheduler = config.scheduler
        self.no_aug_epochs = config.no_aug_epochs
        self.ema = config.ema

        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.print_interval = config.print_interval
        self.eval_interval = config.eval_interval
        self.save_history_ckpt = config.save_history_ckpt
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Test config
        self.test_size = config.test_size
        self.test_conf = config.test_conf
        self.nmsthre = config.nmsthre
        

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            MOTDataset,
            MOTTrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = MOTDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name='images',
                img_size=self.input_size,
                preproc=MOTTrainTransform(
                    max_labels=100,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                    rgb_means=self.rgb_means,
                    std=self.std,
                    legacy=self.legacy
                ),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=MOTTrainTransform(
                max_labels=200,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                rgb_means=self.rgb_means,
                std=self.std,
                legacy=self.legacy
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name='images',
            img_size=self.test_size,
            preproc=ValTransform(
                legacy=self.legacy,
                rgb_means=self.rgb_means,
                std=self.std,
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator