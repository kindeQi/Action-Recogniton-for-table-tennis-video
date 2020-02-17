import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import random
from copy import deepcopy

import target_transform, spatial_transform, temporal_transform
from spatial_transform import *

from utils import load_value_file


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def accimage_loader(path):
    try:
        import accimage

        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader
    else:
        return pil_loader


def video_loader(
    video_dir_path, frame_indices, image_loader, dataset="pingpong_dataset"
):
    video = []
    for i in frame_indices:
        if dataset == "UCF101":
            image_path = os.path.join(video_dir_path, "image_{:05d}.jpg".format(i))
        if dataset == "pingpong_dataset":
            image_path = os.path.join(video_dir_path, "{}.png".format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, "r") as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data["labels"]:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data["database"].items():
        this_subset = value["subset"]
        if this_subset == subset:
            label = value["annotations"]["label"]
            video_names.append("{}/{}".format(label, key))
            annotations.append(value["annotations"])

    return video_names, annotations


def make_dataset(
    root_path, annotation_path, subset, n_samples_for_each_video, sample_duration
):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print("dataset loading [{}/{}]".format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            print("video path not exists: ", video_path)
            continue

        n_frames_file_path = os.path.join(video_path, "n_frames")
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            "video": video_path,
            "segment": [begin_t, end_t],
            "n_frames": n_frames,
            "video_id": video_names[i].split("/")[1],
        }
        if len(annotations) != 0:
            sample["label"] = class_to_idx[annotations[i]["label"]]
        else:
            sample["label"] = -1

        if n_samples_for_each_video == 1:
            sample["frame_indices"] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(
                    1,
                    math.ceil(
                        (n_frames - 1 - sample_duration)
                        / (n_samples_for_each_video - 1)
                    ),
                )
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j["frame_indices"] = list(
                    range(j, min(n_frames + 1, j + sample_duration))
                )
                dataset.append(sample_j)

    return dataset, idx_to_class


def data_reinforcement(data, video_length, threshold=0.7, duration=16):

    random.seed(10)
    data_supplement = []

    for item in data:
        video_index = int(item["video_dir"].split(r"/")[-1])
        start_time = item["frame_indices"][0]
        end_time = item["frame_indices"][-1]
        sample_length = min(duration, len(item["frame_indices"]))

        offset = int(duration * (1 - threshold) / (1 + threshold))

        possible_start_time = max(0, start_time - offset)
        possible_end_time = min(video_length[video_index], end_time + offset)

        start_time1 = random.randrange(possible_start_time, start_time)
        end_time1 = start_time1 + sample_length - 1

        end_time2 = random.randrange(possible_end_time, end_time, -1)
        start_time2 = end_time2 + 1 - sample_length

        if end_time1 - start_time1 > 6:
            item1 = deepcopy(item)
            item1["frame_indices"] = list(range(start_time1, end_time1 + 1))
            data_supplement.append(item1)

        if end_time2 - start_time2 > 6:
            item2 = deepcopy(item)
            item2["frame_indices"] = list(range(start_time2, end_time2 + 1))
            data_supplement.append(item2)

    return data_supplement + data


class UCF101(data.Dataset):
    def __init__(
        self,
        root_path,
        annotation_path,
        subset,
        spatial_transform=None,
        temporal_transform=None,
        target_transform=None,
        n_samples_for_each_video=1,
        sample_duration=16,
        get_loader=get_default_video_loader,
    ):
        self.data, self.class_names = make_dataset(
            root_path,
            annotation_path,
            subset,
            n_samples_for_each_video,
            sample_duration,
        )

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        path = self.data[index]["video"]

        frame_indices = self.data[index]["frame_indices"]
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)


class pingpong_dataset(data.Dataset):
    def __init__(
        self,
        root_path,
        annotation_path,
        subset=None,
        spatial_transform=None,
        temporal_transform=None,
        target_transform=None,
        n_samples_for_each_video=1,
        sample_duration=16,
        get_loader=get_default_video_loader,
    ):
        self.data = []
        self.idx_to_class = dict()
        self.class_to_idx = dict()
        self.video_length = dict()

        for item in os.listdir(root_path):
            try:
                self.video_length[int(item)] = len(os.listdir(root_path + item))
            except:
                pass

        file_path = [
            os.path.join(annotation_path, item) for item in os.listdir(annotation_path)
        ]

        for file in file_path:
            with open(file, mode="r") as f:
                meta_data = json.load(f)
                for video_key in meta_data[0].keys():
                    for index, clip in enumerate(meta_data[0][video_key]):
                        if len(clip["action_catagory"]) == 3 and int(
                            clip["end_time"]
                        ) > int(clip["start_time"]):
                            clip["video_dir"] = os.path.join(root_path, video_key)
                            clip["action_catagory"] = clip["action_catagory"][-1]
                            self.data.append(clip)
            # print('load file--{} finished'.format(file))

        for index, item in enumerate(
            set([item["action_catagory"] for item in self.data])
        ):
            self.idx_to_class[index] = item
            self.class_to_idx[item] = index

        for index, item in enumerate(self.data):
            self.data[index]["frame_indices"] = list(
                range(int(item["start_time"]), int(item["end_time"]) + 1)
            )
            self.data[index].pop("start_time", None)
            self.data[index].pop("end_time", None)
            self.data[index]["action_catagory"] = self.class_to_idx[
                item["action_catagory"]
            ]

        # self.data = data_reinforcement(self.data, self.video_length)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]["video_dir"]

        frame_indices = self.data[index]["frame_indices"]

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            # target = self.target_transform(target)
            target = target["action_catagory"]
        return clip, target


class demo_dataset(data.Dataset):
    def __init__(
        self,
        root_path,
        annotation_path,
        subset=None,
        spatial_transform=None,
        temporal_transform=None,
        target_transform=None,
        n_samples_for_each_video=1,
        sample_duration=16,
        get_loader=get_default_video_loader,
    ):
        self.data = []
        self.idx_to_class = dict()
        self.class_to_idx = dict()
        self.video_length = dict()

        for item in os.listdir(root_path):
            try:
                self.video_length[int(item)] = len(os.listdir(root_path + item))
            except:
                pass

        file_path = [
            os.path.join(annotation_path, item) for item in os.listdir(annotation_path)
        ]

        for file in file_path:
            with open(file, mode="r") as f:
                meta_data = json.load(f)
                for video_key in meta_data[0].keys():
                    for index, clip in enumerate(meta_data[0][video_key]):
                        if len(clip["action_catagory"]) == 3 and int(
                            clip["end_time"]
                        ) > int(clip["start_time"]):
                            clip["video_dir"] = os.path.join(root_path, video_key)
                            clip["action_catagory"] = clip["action_catagory"][-1]
                            self.data.append(clip)
            print("load file--{} finished".format(file))

        for index, item in enumerate(
            set([item["action_catagory"] for item in self.data])
        ):
            self.idx_to_class[index] = item
            self.class_to_idx[item] = index

        for index, item in enumerate(self.data):
            self.data[index]["frame_indices"] = list(
                range(int(item["start_time"]), int(item["end_time"]) + 1)
            )
            self.data[index].pop("start_time", None)
            self.data[index].pop("end_time", None)
            self.data[index]["action_catagory"] = self.class_to_idx[
                item["action_catagory"]
            ]

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]["video_dir"]

        frame_indices = self.data[index]["frame_indices"]

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            # target = self.target_transform(target)
            target = target["action_catagory"]

        meta_data = "{};{};{};{}".format(
            path,
            frame_indices[0],
            frame_indices[-1],
            self.data[index]["action_catagory"],
        )
        return clip, target, meta_data


if __name__ == "__main__":

    print("success")
