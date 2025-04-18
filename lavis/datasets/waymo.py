import logging
import os
import glob
import pandas as pd
import numpy as np
import ast
from PIL import Image
from io import BytesIO
from torch.utils.data import ConcatDataset

import warnings
import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from lavis.common import utils
from lavis.processors.processor import BaseProcessor, Blip2ImageTrainProcessor, BlipCaptionProcessor
from omegaconf import OmegaConf

# from lavis.datasets.datasets.waymo_datasets import WaymoCameraDataset, WaymoLidarDataset

from lavis.datasets.datasets.caption_datasets import CaptionDataset

from omegaconf import ListConfig


# -----------------------------
# Custom Dataset Classes
# -----------------------------


class WaymoCameraDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, camera_root, ann_paths, **kwargs):
        super().__init__(vis_processor, text_processor, camera_root, ann_paths)
        parquet_files = sorted(glob.glob(os.path.join(camera_root, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {camera_root}")
        self.camera_df = pd.read_parquet(parquet_files[0])

    def __getitem__(self, index):
        ann = self.annotation[index]
        try:
            image_bytes = self.camera_df['[CameraImageComponent].image'].iloc[ann["camera_index"]]
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print(f"[CameraImage] Error loading image at index {ann['camera_index']}: {e}")
            return None
        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])
        return {
            "image": image,
            "text_input": caption,
            "image_id": ann.get("image_id", str(ann["camera_index"]))
        }


class WaymoLidarDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, lidar_root, ann_paths, **kwargs):
        super().__init__(vis_processor, text_processor, lidar_root, ann_paths)
        parquet_files = sorted(glob.glob(os.path.join(lidar_root, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {lidar_root}")
        self.lidar_df = pd.read_parquet(parquet_files[0])

    def __getitem__(self, index):
        ann = self.annotation[index]
        try:
            row = self.lidar_df.iloc[ann["lidar_index"]]
            raw_values = row['[LiDARComponent].range_image_return1.values']
            raw_shape = row['[LiDARComponent].range_image_return1.shape']
            values = np.array(ast.literal_eval(raw_values) if isinstance(raw_values, str) else raw_values)
            shape = ast.literal_eval(raw_shape) if isinstance(raw_shape, str) else raw_shape
            range_image = values.reshape(shape)
            bev_image = Image.fromarray((range_image[:, :, 0] * 255).astype(np.uint8)).convert("RGB")
            image = self.vis_processor(bev_image)
        except Exception as e:
            print(f"[LiDAR] Error loading data at index {ann['lidar_index']}: {e}")
            return None
        caption = self.text_processor(ann["caption"])
        return {
            "image": image,
            "text_input": caption,
            "image_id": ann.get("image_id", str(ann["lidar_index"]))
        }


from torch.utils.data import ConcatDataset

class WrappedConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.datasets = datasets  # store original list

    def collater(self, samples):
        # Delegate to the first dataset's collater
        return self.datasets[0].collater(samples)


class WaymoDatasetBuilder:
    train_dataset_cls = None
    eval_dataset_cls = None

    def __init__(self, cfg):
        self.config = cfg

        if isinstance(cfg.data_type, (ListConfig, list)):
            if len(cfg.data_type) == 1 and isinstance(cfg.data_type[0], list):
                self.data_type = cfg.data_type[0]
            else:
                self.data_type = list(cfg.data_type)
        elif isinstance(cfg.data_type, str):
            self.data_type = [cfg.data_type]
        else:
            raise TypeError(f"Unexpected data_type format: {cfg.data_type}")

        print("[DEBUG] Final data_type:", self.data_type)
        self.text_processors = {}
        self.processors = {}
        print("[DEBUG] Data types:", self.data_type)
        print("[DEBUG] build_info keys:", self.config.build_info.keys())

    def build_datasets(self):
        if is_main_process():
            self._download_data()
        if is_dist_avail_and_initialized():
            dist.barrier()
        return self.build()

    def _download_data(self):
        for modality in self.data_type:
            path = utils.get_cache_path(self.config.build_info.get(modality).storage)
            if not os.path.exists(path):
                warnings.warn(f"The specified path {path} for {modality} inputs does not exist.")

    def _build_processor(self, cfg_name):
        cfg = self.config.get(cfg_name)
        return {
            split: self._build_proc_from_cfg(cfg.get(split)) if cfg else None
            for split in ["train", "eval"]
        }

    @staticmethod
    def _build_proc_from_cfg(cfg):
        if cfg is None:
            return BaseProcessor()
        name = cfg.name
        if name == "blip2_image_train":
            return Blip2ImageTrainProcessor.from_config(cfg)
        if name == "blip_caption":
            return BlipCaptionProcessor.from_config(cfg)
        raise ValueError(f"Unknown processor: {name}")

    def build_processors(self):
        self.text_processors = self._build_processor("text_processor")
        self.processors = {
            split: {
                modality: self._build_proc_from_cfg(
                    self.config.get("vis_processor", {}).get(split)
                ) for modality in self.data_type
            } for split in ["train", "eval"]
        }

    # def build(self):
    #     self.build_processors()
    #     build_info = self.config.build_info
    #     datasets = {}

    #     for split in ["train"]:
    #         is_train = split == "train"
    #         datasets[split] = {}

    #         for modality in self.data_type:
    #             dataset_cls = WaymoCameraDataset if modality == "camera" else WaymoLidarDataset

    #             dataset_args = {
    #                 "vis_processor": self.processors[split].get(modality, BaseProcessor()),
    #                 "text_processor": self.text_processors[split],
    #                 f"{modality}_root": build_info[modality].storage,
    #                 "ann_paths": [build_info.annotations[modality][split].storage],
    #                 "modalities": self.data_type,
    #             }

    #             datasets[split][modality] = dataset_cls(**dataset_args)

    #     print(f"[DEBUG] Final dataset structure: { {k: list(v.keys()) for k, v in datasets.items()} }")
    #     return datasets

    # def build(self):
    #     self.build_processors()
    #     build_info = self.config.build_info
    #     datasets = {}

    #     for split in ["train"]:
    #         is_train = split == "train"
    #         datasets[split] = []

    #         for modality in self.data_type:
    #             dataset_cls = WaymoCameraDataset if modality == "camera" else WaymoLidarDataset

    #             dataset_args = {
    #                 "vis_processor": self.processors[split].get(modality, BaseProcessor()),
    #                 "text_processor": self.text_processors[split],
    #                 f"{modality}_root": build_info[modality].storage,
    #                 "ann_paths": [build_info.annotations[modality][split].storage],
    #                 "modalities": self.data_type
    #             }

    #             dataset = dataset_cls(**dataset_args)
    #             datasets[split].append(dataset)

    #     return datasets

    # def build(self):
    #     self.build_processors()
    #     build_info = self.config.build_info
    #     datasets = {}

    #     for split in ["train"]:
    #         is_train = split == "train"
    #         split_datasets = []

    #         for modality in self.data_type:
    #             dataset_cls = WaymoCameraDataset if modality == "camera" else WaymoLidarDataset

    #             dataset_args = {
    #                 "vis_processor": self.processors[split].get(modality, BaseProcessor()),
    #                 "text_processor": self.text_processors[split],
    #                 f"{modality}_root": build_info[modality].storage,
    #                 "ann_paths": [build_info.annotations[modality][split].storage],
    #                 "modalities": self.data_type
    #             }

    #             dataset = dataset_cls(**dataset_args)
    #             split_datasets.append(dataset)

    #         datasets[split] = ConcatDataset(split_datasets)

    #     # return datasets
    # #     return {
    # #     "waymo": {
    # #         "train": ConcatDataset(split_datasets)
    # #     }
    # # }
    #     return {"train": WrappedConcatDataset([datasets["train_camera"], datasets["train_lidar"]])}


    def build(self):
        self.build_processors()
        build_info = self.config.build_info
        datasets = []
        
        for modality in self.data_type:
            dataset_cls = WaymoCameraDataset if modality == "camera" else WaymoLidarDataset

            dataset_args = {
                "vis_processor": self.processors["train"].get(modality, BaseProcessor()),
                "text_processor": self.text_processors["train"],
                f"{modality}_root": build_info[modality].storage,
                "ann_paths": [build_info.annotations[modality]["train"].storage],
                "modalities": self.data_type,
            }

            dataset = dataset_cls(**dataset_args)
            datasets.append(dataset)

        # Wrap with collater-compatible dataset
        # return {"train": WrappedConcatDataset(datasets)}
        return {
        "waymo": {
            "train": WrappedConcatDataset([datasets["train_camera"], datasets["train_lidar"]])
        }
    }




if __name__ == "__main__":
    import json

    camera_ann_path = "lavis/my_datasets/waymo/waymo_camera_annotations.json"
    camera_data_path = "lavis/my_datasets/waymo/train/camera_image"
    lidar_ann_path = "lavis/my_datasets/waymo/waymo_lidar_annotations.json"
    lidar_data_path = "lavis/my_datasets/waymo/train/lidar"

    with open(camera_ann_path, "r") as f:
        camera_annotations = json.load(f)
    with open(lidar_ann_path, "r") as f:
        lidar_annotations = json.load(f)

    camera_processor = Blip2ImageTrainProcessor(image_size=224)
    text_processor = BlipCaptionProcessor()

    print("\n[Testing Camera Dataset]")
    camera_dataset = WaymoCameraDataset(
        vis_processor=camera_processor,
        text_processor=text_processor,
        camera_root=camera_data_path,
        ann_paths=[camera_ann_path],
        modalities=["camera"]
    )
    sample = camera_dataset[0]
    print(f"Sample keys: {sample.keys()} | Image ID: {sample['image_id']}")

    print("\n[Testing LiDAR Dataset]")
    lidar_dataset = WaymoLidarDataset(
        vis_processor=camera_processor,  # using same processor for now
        text_processor=text_processor,
        lidar_root=lidar_data_path,
        ann_paths=[lidar_ann_path],
        modalities=["lidar"]
    )
    sample = lidar_dataset[0]
    print(f"Sample keys: {sample.keys()} | Image ID: {sample['image_id']}")





# class WaymoDatasetBuilder:
#     train_dataset_cls = None  # Will be assigned dynamically based on modality
#     eval_dataset_cls = None

#     # def __init__(self, cfg):
#     #     self.config = cfg
#     #     self.data_type = cfg.data_type if isinstance(cfg.data_type, list) else [cfg.data_type]
#     #     self.text_processors = {}
#     #     self.processors = {}
#         # print("[DEBUG] Data types:", self.data_type)
#         # print("[DEBUG] build_info keys:", self.config.build_info.keys())

#     def __init__(self, cfg):
#         self.config = cfg
#         # ✅ Correct flattening
#         if isinstance(cfg.data_type, (ListConfig, list)):
#             if len(cfg.data_type) == 1 and isinstance(cfg.data_type[0], list):
#                 self.data_type = cfg.data_type[0]  # unwrap nested list
#             else:
#                 self.data_type = list(cfg.data_type)
#         elif isinstance(cfg.data_type, str):
#             self.data_type = [cfg.data_type]
#         else:
#             raise TypeError(f"Unexpected data_type format: {cfg.data_type}")
        
#         print("[DEBUG] Final data_type:", self.data_type)
#         self.text_processors = {}
#         self.processors = {}
#         print("[DEBUG] Data types:", self.data_type)
#         print("[DEBUG] build_info keys:", self.config.build_info.keys())

#     def build_datasets(self):
#         if is_main_process():
#             self._download_data()
#         if is_dist_avail_and_initialized():
#             dist.barrier()
#         return self.build()

#     def _download_data(self):
#         for modality in self.data_type:
#             path = utils.get_cache_path(self.config.build_info.get(modality).storage)
#             if not os.path.exists(path):
#                 warnings.warn(f"The specified path {path} for {modality} inputs does not exist.")

#     def _build_processor(self, cfg_name):
#         cfg = self.config.get(cfg_name)
#         return {
#             split: self._build_proc_from_cfg(cfg.get(split)) if cfg else None
#             for split in ["train", "eval"]
#         }

#     @staticmethod
#     def _build_proc_from_cfg(cfg):
#         if cfg is None:
#             return BaseProcessor()
#         name = cfg.name
#         if name == "blip2_image_train":
#             return Blip2ImageTrainProcessor.from_config(cfg)
#         if name == "blip_caption":
#             return BlipCaptionProcessor.from_config(cfg)
#         raise ValueError(f"Unknown processor: {name}")

#     def build_processors(self):
#         self.text_processors = self._build_processor("text_processor")
#         self.processors = {
#             split: {
#                 modality: self._build_proc_from_cfg(
#                     self.config.get(f"vis_processor", {}).get(split)
#                 ) for modality in self.data_type
#             } for split in ["train", "eval"]
#         }

#     def build(self):
#         self.build_processors()
#         build_info = self.config.build_info
#         datasets = {}

#         for split in ["train"]:
#             is_train = split == "train"

#             for modality in self.data_type:
#                 dataset_cls = WaymoCameraDataset if modality == "camera" else WaymoLidarDataset

#                 dataset_args = {
#                     "vis_processor": self.processors[split].get(modality, BaseProcessor()),
#                     "text_processor": self.text_processors[split],
#                     # f"{modality}_root": utils.get_cache_path(build_info[modality].storage),
#                     f"{modality}_root": (build_info[modality].storage),
#                     "ann_paths": [build_info.annotations[modality][split].storage],
#                     "modalities": self.data_type
#                 }

#                 datasets[split] = dataset_cls(**dataset_args)

#         return datasets





# if __name__ == "__main__":
#     import json
#     from lavis.processors.processor import Blip2ImageTrainProcessor, BlipCaptionProcessor

#     camera_ann_path = "lavis/my_datasets/waymo/waymo_camera_annotations.json"
#     camera_data_path = "lavis/my_datasets/waymo/train/camera_image"
#     lidar_ann_path = "lavis/my_datasets/waymo/waymo_lidar_annotations.json"
#     lidar_data_path = "lavis/my_datasets/waymo/train/lidar"

#     with open(camera_ann_path, "r") as f:
#         camera_annotations = json.load(f)
#     with open(lidar_ann_path, "r") as f:
#         lidar_annotations = json.load(f)

#     camera_processor = Blip2ImageTrainProcessor(image_size=224)
#     text_processor = BlipCaptionProcessor()

#     print("\n[Testing Camera Dataset]")
#     camera_dataset = WaymoCameraDataset(
#         vis_processor=camera_processor,
#         text_processor=text_processor,
#         camera_root=camera_data_path,
#         ann_paths=[camera_ann_path],
#         modalities=["camera"]
#     )
#     sample = camera_dataset[0]
#     print(f"Sample keys: {sample.keys()} | Image ID: {sample['image_id']}")

#     print("\n[Testing LiDAR Dataset]")
#     lidar_dataset = WaymoLidarDataset(
#         vis_processor=camera_processor,  # using same processor for now
#         text_processor=text_processor,
#         lidar_root=lidar_data_path,
#         ann_paths=[lidar_ann_path],
#         modalities=["lidar"]
#     )
#     sample = lidar_dataset[0]
#     print(f"Sample keys: {sample.keys()} | Image ID: {sample['image_id']}")









































# import os
# import json
# import glob
# import logging
# import warnings
# import shutil
# import ast
# from io import BytesIO

# import numpy as np
# import pandas as pd
# from PIL import Image, ImageFile
# import torch

# from lavis.processors.processor import Blip2ImageTrainProcessor, BlipCaptionProcessor, BaseProcessor
# from lavis.datasets.datasets.caption_datasets import CaptionDataset
# import lavis.common.utils as utils
# from lavis.common.dist_utils import is_main_process, is_dist_avail_and_initialized
# from omegaconf import OmegaConf

# ImageFile.LOAD_TRUNCATED_IMAGES = True


# # -----------------------------
# # Custom Dataset Classes
# # -----------------------------

# class WaymoCameraDataset(CaptionDataset):
#     def __init__(self, vis_processor, text_processor, camera_root, ann_paths, **kwargs):
#         super().__init__(vis_processor, text_processor, camera_root, ann_paths)
#         parquet_files = sorted(glob.glob(os.path.join(camera_root, "*.parquet")))
#         if not parquet_files:
#             raise FileNotFoundError(f"No parquet files found in {camera_root}")
#         self.camera_df = pd.read_parquet(parquet_files[0])

#     def __getitem__(self, index):
#         ann = self.annotation[index]
#         try:
#             image_bytes = self.camera_df['[CameraImageComponent].image'].iloc[ann["camera_index"]]
#             image = Image.open(BytesIO(image_bytes)).convert("RGB")
#         except Exception as e:
#             print(f"[CameraImage] Error loading image at index {ann['camera_index']}: {e}")
#             return None
#         image = self.vis_processor(image)
#         caption = self.text_processor(ann["caption"])
#         return {
#             "image": image,
#             "text_input": caption,
#             "image_id": ann.get("image_id", str(ann["camera_index"]))
#         }


# class WaymoLidarDataset(CaptionDataset):
#     def __init__(self, vis_processor, text_processor, lidar_root, ann_paths, **kwargs):
#         super().__init__(vis_processor, text_processor, lidar_root, ann_paths)
#         parquet_files = sorted(glob.glob(os.path.join(lidar_root, "*.parquet")))
#         if not parquet_files:
#             raise FileNotFoundError(f"No parquet files found in {lidar_root}")
#         self.lidar_df = pd.read_parquet(parquet_files[0])

#     def __getitem__(self, index):
#         ann = self.annotation[index]
#         try:
#             row = self.lidar_df.iloc[ann["lidar_index"]]
#             raw_values = row['[LiDARComponent].range_image_return1.values']
#             raw_shape = row['[LiDARComponent].range_image_return1.shape']
#             values = np.array(ast.literal_eval(raw_values) if isinstance(raw_values, str) else raw_values)
#             shape = ast.literal_eval(raw_shape) if isinstance(raw_shape, str) else raw_shape
#             range_image = values.reshape(shape)
#             bev_image = Image.fromarray((range_image[:, :, 0] * 255).astype(np.uint8)).convert("RGB")
#             image = self.vis_processor(bev_image)
#         except Exception as e:
#             print(f"[LiDAR] Error loading data at index {ann['lidar_index']}: {e}")
#             return None
#         caption = self.text_processor(ann["caption"])
#         return {
#             "image": image,
#             "text_input": caption,
#             "image_id": ann.get("image_id", str(ann["lidar_index"]))
#         }


# # -----------------------------
# # Custom Builder
# # -----------------------------

# class WaymoDatasetBuilder:
#     train_dataset_cls = None
#     eval_dataset_cls = None

#     def __init__(self, cfg):
#         self.config = cfg
#         # self.data_type = cfg.data_type if isinstance(cfg.data_type, list) else [cfg.data_type]
#         self.data_type = list(cfg.data_type)

#         self.text_processors = {}
#         self.processors = {}

#     def build_datasets(self):
#         if is_main_process():
#             self._download_data()
#         if is_dist_avail_and_initialized():
#             torch.distributed.barrier()
#         return self.build()

#     def build_processors(self):
#         self.text_processors = self._build_processor("text_processor")
#         self.processors = {
#             split: {
#                 modality: self._build_proc_from_cfg(
#                     self.config.get(f"{modality}_processor", {}).get(split)
#                 ) for modality in self.data_type
#             } for split in ["train", "eval"]
#         }

#     def _build_processor(self, cfg_name):
#         cfg = self.config.get(cfg_name)
#         return {
#             split: self._build_proc_from_cfg(cfg.get(split)) if cfg else None
#             for split in ["train", "eval"]
#         }

#     def _download_data(self):
#         for modality in self.data_type:
#             path = utils.get_cache_path(self.config.build_info.get(modality).storage)
#             if not os.path.exists(path):
#                 warnings.warn(f"The specified path {path} for {modality} inputs does not exist.")

#     def _build_proc_from_cfg(self, cfg):
#         if cfg is None:
#             return BaseProcessor()
#         name = cfg.name
#         if name == "blip2_image_train":
#             return Blip2ImageTrainProcessor.from_config(cfg)
#         if name == "blip_caption":
#             return BlipCaptionProcessor.from_config(cfg)
#         raise ValueError(f"Unknown processor: {name}")

#     def build(self):
#         self.build_processors()
#         build_info = self.config.build_info
#         datasets = {}

#         for split in ["train", "val", "test"]:
#             is_train = split == "train"
#             dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls

#             dataset_args = {
#                 f"{modality}_root": utils.get_cache_path(build_info[modality].storage)
#                 for modality in self.data_type
#             }
#             dataset_args.update({
#                 f"{modality}_processor": self.processors[split][modality]
#                 for modality in self.data_type
#             })
#             dataset_args["text_processor"] = self.text_processors[split]
#             dataset_args["ann_paths"] = ["dummy_ann.json"]  # Modify as needed
#             dataset_args["modalities"] = self.data_type

#             datasets[split] = dataset_cls(**dataset_args)

#         return datasets


# def load_dataset_config(cfg_path):
#     cfg = OmegaConf.load(cfg_path).datasets
#     return next(iter(cfg.values()))


# # -----------------------------
# # Specific Builder Instances
# # -----------------------------

# class WaymoCameraBuilder(WaymoDatasetBuilder):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.train_dataset_cls = WaymoCameraDataset
#         self.eval_dataset_cls = WaymoCameraDataset


# class WaymoLidarBuilder(WaymoDatasetBuilder):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.train_dataset_cls = WaymoLidarDataset
#         self.eval_dataset_cls = WaymoLidarDataset



# # --------------------
# # Testing Main Block
# # --------------------
# from omegaconf import OmegaConf
# if __name__ == "__main__":
    

#     cfg_path = "lavis/configs/datasets/waymo/defaults.yaml"
#     cfg = OmegaConf.load(cfg_path).datasets.waymo

#     builder = WaymoDatasetBuilder(cfg)
#     datasets = builder.build_datasets()

#     print("\n[Testing WaymoDatasetBuilder - Camera Split]")
#     camera_dataset = datasets.get("train")
#     if camera_dataset:
#         sample = camera_dataset[0]
#         print(f"Sample keys: {sample.keys()} | Image ID: {sample['image_id']}")
#     else:
#         print("Camera dataset not loaded.")



# # import os
# # import ast
# # import torch
# # import pandas as pd
# # import numpy as np
# # from PIL import Image
# # from PIL import ImageFile
# # from io import BytesIO
# # import glob

# # from lavis.datasets.datasets.caption_datasets import CaptionDataset
# # from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
# # from lavis.processors.processor import Blip2ImageTrainProcessor, BlipCaptionProcessor

# # ImageFile.LOAD_TRUNCATED_IMAGES = True

# # # --------------------
# # # Dataset Definitions
# # # --------------------

# # class WaymoCameraDataset(CaptionDataset):
# #     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
# #         super().__init__(vis_processor, text_processor, vis_root, ann_paths)
# #         parquet_files = sorted(glob.glob(os.path.join(vis_root, "*.parquet")))
# #         if not parquet_files:
# #             raise FileNotFoundError(f"No .parquet files found in {vis_root}")
# #         self.camera_df = pd.read_parquet(parquet_files[0])

# #     def __getitem__(self, index):
# #         ann = self.annotation[index]
# #         try:
# #             image_bytes = self.camera_df['[CameraImageComponent].image'].iloc[ann.get("camera_index", 0)]
# #             image = Image.open(BytesIO(image_bytes)).convert("RGB")
# #         except Exception as e:
# #             print(f"[CameraImage] Error loading image: {e}")
# #             return None

# #         image = self.vis_processor(image)
# #         caption = self.text_processor(ann.get("caption", ""))

# #         return {
# #             "image": image,
# #             "text_input": caption,
# #             "image_id": ann.get("image_id", str(ann.get("camera_index", index)))
# #         }


# # class WaymoLidarDataset(CaptionDataset):
# #     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
# #         super().__init__(vis_processor, text_processor, vis_root, ann_paths)
# #         parquet_files = sorted(glob.glob(os.path.join(vis_root, "*.parquet")))
# #         if not parquet_files:
# #             raise FileNotFoundError(f"No .parquet files found in {vis_root}")
# #         self.lidar_df = pd.read_parquet(parquet_files[0])

# #     def __getitem__(self, index):
# #         ann = self.annotation[index]
# #         try:
# #             row = self.lidar_df.iloc[ann.get("lidar_index", 0)]
# #             raw_values = row['[LiDARComponent].range_image_return1.values']
# #             raw_shape = row['[LiDARComponent].range_image_return1.shape']

# #             values = np.array(ast.literal_eval(raw_values) if isinstance(raw_values, str) else raw_values)
# #             shape = ast.literal_eval(raw_shape) if isinstance(raw_shape, str) else raw_shape
# #             range_image = values.reshape(shape)

# #             bev_image = Image.fromarray((range_image[:, :, 0] * 255).astype(np.uint8)).convert("RGB")
# #             image = self.vis_processor(bev_image)

# #         except Exception as e:
# #             print(f"[LiDAR] Error loading data: {e}")
# #             return None

# #         caption = self.text_processor(ann.get("caption", ""))

# #         return {
# #             "image": image,
# #             "text_input": caption,
# #             "image_id": ann.get("image_id", str(ann.get("lidar_index", index)))
# #         }


# # # --------------------
# # # Builder Classes (Non-registry)
# # # --------------------

# # class WaymoDatasetBuilder:
# #     def __init__(self, cfg):
# #         self.config = cfg
# #         self.data_type = cfg.data_type if isinstance(cfg.data_type, list) else [cfg.data_type]
# #         self.vis_processors = {"train": None, "eval": None}
# #         self.text_processors = {"train": None, "eval": None}

# #     def _build_proc_from_cfg(self, cfg):
# #         if cfg is None:
# #             return None
# #         if cfg.name == "blip2_image_train":
# #             return Blip2ImageTrainProcessor.from_config(cfg)
# #         elif cfg.name == "blip_caption":
# #             return BlipCaptionProcessor.from_config(cfg)
# #         else:
# #             raise ValueError(f"Unknown processor: {cfg.name}")











# # class WaymoCameraBuilder(BaseDatasetBuilder):
# #     train_dataset_cls = WaymoCameraDataset
# #     eval_dataset_cls = WaymoCameraDataset


# # class WaymoLidarBuilder(BaseDatasetBuilder):
# #     train_dataset_cls = WaymoLidarDataset
# #     eval_dataset_cls = WaymoLidarDataset


# # # --------------------
# # # Testing Main Block
# # # --------------------

# # if __name__ == "__main__":
# #     import json

# #     camera_ann_path = "lavis/my_datasets/waymo/waymo_camera_annotations.json"
# #     camera_data_path = "lavis/my_datasets/waymo/train/camera_image"
# #     lidar_ann_path = "lavis/my_datasets/waymo/waymo_lidar_annotations.json"
# #     lidar_data_path = "lavis/my_datasets/waymo/train/lidar"

# #     with open(camera_ann_path, "r") as f:
# #         camera_annotations = json.load(f)
# #     with open(lidar_ann_path, "r") as f:
# #         lidar_annotations = json.load(f)

# #     camera_processor = Blip2ImageTrainProcessor(image_size=224)
# #     text_processor = BlipCaptionProcessor()

# #     print("\n[Testing Camera Dataset]")
# #     camera_dataset = WaymoCameraDataset(camera_processor, text_processor, camera_data_path, [camera_ann_path])
# #     sample = camera_dataset[0]
# #     print(f"Sample keys: {sample.keys()} | Image ID: {sample['image_id']}")

# #     print("\n[Testing LiDAR Dataset]")
# #     lidar_dataset = WaymoLidarDataset(camera_processor, text_processor, lidar_data_path, [lidar_ann_path])
# #     sample = lidar_dataset[0]
# #     print(f"Sample keys: {sample.keys()} | Image ID: {sample['image_id']}")






# # # import os
# # # import ast
# # # import torch
# # # import pandas as pd
# # # import numpy as np
# # # from PIL import Image
# # # from PIL import ImageFile
# # # from io import BytesIO
# # # import glob

# # # from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionInstructDataset
# # # from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
# # # from lavis.common.registry import registry
# # # from lavis.processors.processor import Blip2ImageTrainProcessor, BlipCaptionProcessor

# # # ImageFile.LOAD_TRUNCATED_IMAGES = True

# # # # --------------------
# # # # Dataset Definitions
# # # # --------------------

# # # class WaymoCameraDataset(CaptionDataset):
# # #     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
# # #         super().__init__(vis_processor, text_processor, vis_root, ann_paths)
# # #         # print("[DEBUG] Loading all Parquet files under:", vis_root)
# # #         parquet_files = sorted(glob.glob(os.path.join(vis_root, "*.parquet")))
# # #         if not parquet_files:
# # #             raise FileNotFoundError(f"No .parquet files found in {vis_root}")

# # #         # print(f"[DEBUG] Found {len(parquet_files)} Parquet files. Loading the first one for test...")
# # #         self.camera_df = pd.read_parquet(parquet_files[0])
# # #         # print("[DEBUG] Successfully loaded test Parquet file!")

# # #     def __getitem__(self, index):
# # #         ann = self.annotation[index]

# # #         try:
# # #             # print(f"Loading image at index {ann['camera_index']}...")
# # #             image_bytes = self.camera_df['[CameraImageComponent].image'].iloc[ann["camera_index"]]
# # #             image = Image.open(BytesIO(image_bytes)).convert("RGB")
# # #             # print("Loaded image successfully!")
# # #         except Exception as e:
# # #             print(f"[CameraImage] Error loading image at index {ann['camera_index']}: {e}")
# # #             return None

# # #         image = self.vis_processor(image)
# # #         caption = self.text_processor(ann["caption"])

# # #         return {
# # #             "image": image,
# # #             "text_input": caption,
# # #             "image_id": ann.get("image_id", str(ann["camera_index"]))
# # #         }


# # # class WaymoLidarDataset(CaptionDataset):
# # #     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
# # #         super().__init__(vis_processor, text_processor, vis_root, ann_paths)
# # #         # print("[DEBUG] Loading all LiDAR Parquet files under:", vis_root)
# # #         parquet_files = sorted(glob.glob(os.path.join(vis_root, "*.parquet")))
# # #         if not parquet_files:
# # #             raise FileNotFoundError(f"No .parquet files found in {vis_root}")

# # #         # print(f"[DEBUG] Found {len(parquet_files)} Parquet files. Loading the first one for test...")
# # #         self.lidar_df = pd.read_parquet(parquet_files[0])
# # #         # print("[DEBUG] Successfully loaded test LiDAR Parquet file!")

# # #     def __getitem__(self, index):
# # #         ann = self.annotation[index]

# # #         try:
# # #             row = self.lidar_df.iloc[ann["lidar_index"]]
# # #             raw_values = row['[LiDARComponent].range_image_return1.values']
# # #             raw_shape = row['[LiDARComponent].range_image_return1.shape']

# # #             if isinstance(raw_values, str):
# # #                 values = np.array(ast.literal_eval(raw_values))
# # #             else:
# # #                 values = np.array(raw_values)

# # #             if isinstance(raw_shape, str):
# # #                 shape = ast.literal_eval(raw_shape)
# # #             else:
# # #                 shape = raw_shape

# # #             range_image = values.reshape(shape)
# # #             # range_image_tensor = torch.tensor(range_image[:, :, 0:1]).float().permute(2, 0, 1)
# # #             # image = self.vis_processor(range_image_tensor)

# # #             # instead of passing a tensor we are passing a PIL image
# # #             # bev_image = Image.fromarray((range_image[:, :, 0] * 255).astype(np.uint8))  # channel 0 to grayscale
# # #             bev_image = Image.fromarray((range_image[:, :, 0] * 255).astype(np.uint8)).convert("RGB") # 3-channel RGB for the processor (just for now)   

# # #             image = self.vis_processor(bev_image)

# # #         except Exception as e:
# # #             print(f"[LiDAR] Error loading data at index {ann['lidar_index']}: {e}")
# # #             return None

# # #         caption = self.text_processor(ann["caption"])

# # #         return {
# # #             "image": image,
# # #             "text_input": caption,
# # #             "image_id": ann.get("image_id", str(ann["lidar_index"]))
# # #         }


# # # # --------------------
# # # # Builder Definitions
# # # # --------------------

# # # @registry.register_builder("waymo_camera")
# # # class WaymoCameraBuilder(BaseDatasetBuilder):
# # #     train_dataset_cls = WaymoCameraDataset
# # #     eval_dataset_cls = WaymoCameraDataset

# # #     DATASET_CONFIG_DICT = {
# # #         "default": "configs/datasets/waymo/defaults.yaml",
# # #     }


# # # @registry.register_builder("waymo_lidar")
# # # class WaymoLidarBuilder(BaseDatasetBuilder):
# # #     train_dataset_cls = WaymoLidarDataset
# # #     eval_dataset_cls = WaymoLidarDataset

# # #     DATASET_CONFIG_DICT = {
# # #         "default": "configs/datasets/waymo/defaults.yaml",
# # #     }


# # # # --------------------
# # # # Testing Main Block
# # # # --------------------

# # # if __name__ == "__main__":
# # #     import json

# # #     # Dummy minimal setup for testing
# # #     camera_ann_path = "lavis/my_datasets/waymo/waymo_camera_annotations.json"
# # #     camera_data_path = "lavis/my_datasets/waymo/train/camera_image"

# # #     lidar_ann_path = "lavis/my_datasets/waymo/waymo_lidar_annotations.json"
# # #     lidar_data_path = "lavis/my_datasets/waymo/train/lidar"

# # #     # Load dummy annotations
# # #     with open(camera_ann_path, "r") as f:
# # #         camera_annotations = json.load(f)
# # #     with open(lidar_ann_path, "r") as f:
# # #         lidar_annotations = json.load(f)

# # #     # Dummy processors
# # #     camera_processor = Blip2ImageTrainProcessor(image_size=224)
# # #     text_processor = BlipCaptionProcessor()

# # #     print("\n[Testing Camera Dataset]")
# # #     camera_dataset = WaymoCameraDataset(camera_processor, text_processor, camera_data_path, [camera_ann_path])
# # #     sample = camera_dataset[0]
# # #     print(f"Sample keys: {sample.keys()} | Image ID: {sample['image_id']}")

# # #     print("\n[Testing LiDAR Dataset]")
# # #     lidar_dataset = WaymoLidarDataset(camera_processor, text_processor, lidar_data_path, [lidar_ann_path])
# # #     sample = lidar_dataset[0]
# # #     print(f"Sample keys: {sample.keys()} | Image ID: {sample['image_id']}")





