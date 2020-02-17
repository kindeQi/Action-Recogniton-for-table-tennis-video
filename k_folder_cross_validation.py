from spatial_transform import (
    Normalize,
    MultiScaleRandomCrop,
    MultiScaleCornerCrop,
    Compose,
    RandomHorizontalFlip,
    ToTensor,
)
from temporal_transform import TemporalRandomCrop
from target_transform import ClassLabel
from dataset import pingpong_dataset
from utils import Logger

import json
import random
import torch
import os
import opt

# from main import get_mean_std
from copy import deepcopy


class k_folder_cross_validation(object):
    def __init__(self, k, opt, dataset="pingpong_dataset"):
        """
        :param k: how many fold we would like to define
        :param opt: option argument
        :param dataset: just indicate what dataset is using, useless
        """
        print("init {} folder cross validation, the dataset is: {}".format(k, dataset))
        assert opt.train_crop in ["random", "corner", "center"]
        norm_method = Normalize(opt.mean, opt.std)
        if opt.train_crop == "random":
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == "corner":
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == "center":
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=["c"]
            )
        spatial_transform = Compose(
            [crop_method, RandomHorizontalFlip(), ToTensor(opt.norm_value), norm_method]
        )
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()

        assert opt.dataset == "pingpong_dataset"
        self.dataset = pingpong_dataset(
            r"/data2/pingpang_dataset/videos/",
            r"/home/zhangrui/3dresnet/meta_data",
            "training",
            spatial_transform,
            temporal_transform,
            target_transform,
        )
        self.k = k
        self.n_th_call = 0  # 0 ~ k - 1, record this is the No n to call this class
        self.opt = opt

        # shuffle the original dataset
        random.seed(0)  # set the seed value to make the experiment reproductive
        random.shuffle(self.dataset.data)

        print("write down action.json")
        with open("../action.json", "w") as f:
            json.dump(self.dataset.idx_to_class, f)

    def writedown_split(self, path, data, catagory):
        print("write down split: {}".format(catagory))
        data = deepcopy(data)
        for i in range(len(data)):
            data[i]["frame_indices"] = [
                data[i]["frame_indices"][0],
                data[i]["frame_indices"][-1],
            ]
            data[i]["type"] = catagory
        with open(path, mode="w") as f:
            json.dump(data, f)

    def __call__(self):
        start_index, end_index = (
            int(len(self.dataset) / self.k * self.n_th_call),
            int(len(self.dataset) / self.k * (self.n_th_call + 1)),
        )  # the two pointer to define the validation set
        validation_data = deepcopy(self.dataset)
        validation_data.data = validation_data.data[start_index:end_index]

        training_data = deepcopy(self.dataset)
        training_data.data = (
            training_data.data[:start_index] + training_data.data[end_index:]
        )

        self.n_th_call = (
            self.n_th_call + 1
        ) / self.k  # this is the n + 1 th to call this class

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=self.opt.n_threads,
            pin_memory=True,
        )

        train_logger = Logger(
            os.path.join(self.opt.result_path, "train.log"),
            ["epoch", "loss", "acc", "lr"],
        )
        train_batch_logger = Logger(
            os.path.join(self.opt.result_path, "train_batch.log"),
            ["epoch", "batch", "iter", "loss", "acc", "lr"],
        )

        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.n_threads,
            pin_memory=True,
        )
        val_logger = Logger(
            os.path.join(self.opt.result_path, "val.log"), ["epoch", "loss", "acc"]
        )

        self.writedown_split(
            path="../pingpong_dataset_train.json",
            data=training_data.data,
            catagory="train",
        )
        self.writedown_split(
            path="../pingpong_dataset_validation.json",
            data=validation_data.data,
            catagory="validation",
        )
        return train_loader, train_logger, train_batch_logger, val_loader, val_logger


if __name__ == "__main__":
    opt = opt.parse_opts()
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = "{}-{}".format(opt.model, opt.model_depth)
    # opt.mean, opt.std = get_mean_std(opt)

    cross_validation = k_folder_cross_validation(5, opt, dataset="pingpong_dataset")
    train_loader, train_logger, train_batch_logger, val_loader, val_logger = (
        cross_validation()
    )
    print("success")
