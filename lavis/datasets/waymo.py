

import os
import ast
import torch
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFile
from io import BytesIO
import glob

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionInstructDataset
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.common.registry import registry
from lavis.processors.processor import Blip2ImageTrainProcessor, BlipCaptionProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --------------------
# Dataset Definitions
# --------------------

class WaymoCameraDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # print("[DEBUG] Loading all Parquet files under:", vis_root)
        parquet_files = sorted(glob.glob(os.path.join(vis_root, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {vis_root}")

        # print(f"[DEBUG] Found {len(parquet_files)} Parquet files. Loading the first one for test...")
        self.camera_df = pd.read_parquet(parquet_files[0])
        # print("[DEBUG] Successfully loaded test Parquet file!")

    def __getitem__(self, index):
        ann = self.annotation[index]

        try:
            # print(f"Loading image at index {ann['camera_index']}...")
            image_bytes = self.camera_df['[CameraImageComponent].image'].iloc[ann["camera_index"]]
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            # print("Loaded image successfully!")
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
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # print("[DEBUG] Loading all LiDAR Parquet files under:", vis_root)
        parquet_files = sorted(glob.glob(os.path.join(vis_root, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {vis_root}")

        # print(f"[DEBUG] Found {len(parquet_files)} Parquet files. Loading the first one for test...")
        self.lidar_df = pd.read_parquet(parquet_files[0])
        # print("[DEBUG] Successfully loaded test LiDAR Parquet file!")

    def __getitem__(self, index):
        ann = self.annotation[index]

        try:
            row = self.lidar_df.iloc[ann["lidar_index"]]
            raw_values = row['[LiDARComponent].range_image_return1.values']
            raw_shape = row['[LiDARComponent].range_image_return1.shape']

            if isinstance(raw_values, str):
                values = np.array(ast.literal_eval(raw_values))
            else:
                values = np.array(raw_values)

            if isinstance(raw_shape, str):
                shape = ast.literal_eval(raw_shape)
            else:
                shape = raw_shape

            range_image = values.reshape(shape)
            # range_image_tensor = torch.tensor(range_image[:, :, 0:1]).float().permute(2, 0, 1)
            # image = self.vis_processor(range_image_tensor)

            # instead of passing a tensor we are passing a PIL image
            # bev_image = Image.fromarray((range_image[:, :, 0] * 255).astype(np.uint8))  # channel 0 to grayscale
            bev_image = Image.fromarray((range_image[:, :, 0] * 255).astype(np.uint8)).convert("RGB") # 3-channel RGB for the processor (just for now)   

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


# --------------------
# Builder Definitions
# --------------------

@registry.register_builder("waymo_camera")
class WaymoCameraBuilder(BaseDatasetBuilder):
    train_dataset_cls = WaymoCameraDataset
    eval_dataset_cls = WaymoCameraDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/waymo/defaults.yaml",
    }


@registry.register_builder("waymo_lidar")
class WaymoLidarBuilder(BaseDatasetBuilder):
    train_dataset_cls = WaymoLidarDataset
    eval_dataset_cls = WaymoLidarDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/waymo/defaults.yaml",
    }


# --------------------
# Testing Main Block
# --------------------

if __name__ == "__main__":
    import json

    # Dummy minimal setup for testing
    camera_ann_path = "lavis/my_datasets/waymo/waymo_camera_annotations.json"
    camera_data_path = "lavis/my_datasets/waymo/train/camera_image"

    lidar_ann_path = "lavis/my_datasets/waymo/waymo_lidar_annotations.json"
    lidar_data_path = "lavis/my_datasets/waymo/train/lidar"

    # Load dummy annotations
    with open(camera_ann_path, "r") as f:
        camera_annotations = json.load(f)
    with open(lidar_ann_path, "r") as f:
        lidar_annotations = json.load(f)

    # Dummy processors
    camera_processor = Blip2ImageTrainProcessor(image_size=224)
    text_processor = BlipCaptionProcessor()

    print("\n[Testing Camera Dataset]")
    camera_dataset = WaymoCameraDataset(camera_processor, text_processor, camera_data_path, [camera_ann_path])
    sample = camera_dataset[0]
    print(f"Sample keys: {sample.keys()} | Image ID: {sample['image_id']}")

    print("\n[Testing LiDAR Dataset]")
    lidar_dataset = WaymoLidarDataset(camera_processor, text_processor, lidar_data_path, [lidar_ann_path])
    sample = lidar_dataset[0]
    print(f"Sample keys: {sample.keys()} | Image ID: {sample['image_id']}")
