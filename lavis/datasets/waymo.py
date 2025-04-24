import logging
import os
import glob
import pandas as pd
import torch
import numpy as np
import ast
from PIL import Image
from io import BytesIO
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import warnings
import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from lavis.common import utils
from lavis.processors.processor import BaseProcessor, Blip2ImageTrainProcessor
from omegaconf import OmegaConf, ListConfig
import matplotlib.pyplot as plt

class WaymoCameraDataset:
    def __init__(self, vis_processor, camera_root):
        self.vis_processor = vis_processor
        print("[CHECK] Loading camera dataset from:", camera_root)

        self.camera_files = sorted(glob.glob(os.path.join(camera_root, "*.parquet")))
        if not self.camera_files:
            raise FileNotFoundError(f"No parquet files found in {camera_root}")

    def __getitem__(self, idx):
        print(f"[DEBUG] {self.__class__.__name__} __getitem__({idx})")
        parquet_file = self.camera_files[idx]
        try:
            df = pd.read_parquet(parquet_file)
            image_bytes = df['[CameraImageComponent].image'].iloc[0]
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = self.vis_processor(image)
        except Exception as e:
            print(f"[CameraImage] Error loading image from file {parquet_file}: {e}")
            return None

        return {
            "image": image,
            "image_id": os.path.basename(parquet_file)
        }

    def __len__(self):
        return len(self.camera_files)

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}
        batch = {
            k: torch.stack([s[k] for s in samples]) if isinstance(samples[0][k], torch.Tensor) else [s[k] for s in samples]
            for k in samples[0]
        }
        return batch


class WaymoLidarDataset:
    def __init__(self, vis_processor, lidar_root):
        self.vis_processor = vis_processor
        self.lidar_files = sorted(glob.glob(os.path.join(lidar_root, "*.parquet")))
        if not self.lidar_files:
            raise FileNotFoundError(f"No parquet files found in {lidar_root}")

    def __getitem__(self, idx):
        print(f"[DEBUG] {self.__class__.__name__} __getitem__({idx})")
        parquet_file = self.lidar_files[idx]
        try:
            df = pd.read_parquet(parquet_file)
            row = df.iloc[0]
            raw_values = row['[LiDARComponent].range_image_return1.values']
            raw_shape = row['[LiDARComponent].range_image_return1.shape']
            values = np.array(ast.literal_eval(raw_values) if isinstance(raw_values, str) else raw_values)
            shape = ast.literal_eval(raw_shape) if isinstance(raw_shape, str) else raw_shape
            range_image = values.reshape(shape)
            bev_image = Image.fromarray((range_image[:, :, 0] * 255).astype(np.uint8)).convert("RGB")
            image = self.vis_processor(bev_image)
        except Exception as e:
            print(f"[LiDAR] Error loading data from file {parquet_file}: {e}")
            return None

        return {
            "lidar": image,
            "image_id": os.path.basename(parquet_file)
        }

    def __len__(self):
        return len(self.lidar_files)

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}
        batch = {
            k: torch.stack([s[k] for s in samples]) if isinstance(samples[0][k], torch.Tensor) else [s[k] for s in samples]
            for k in samples[0]
        }
        return batch


class WrappedConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        assert len(datasets) == 2, "Expecting camera and lidar datasets"
        super().__init__(datasets)
        self.camera_dataset = datasets[0]
        self.lidar_dataset = datasets[1]

    def __getitem__(self, idx):
        print(f"[DEBUG] WrappedConcatDataset __getitem__({idx})")
        try:
            if idx >= len(self.camera_dataset) or idx >= len(self.lidar_dataset):
                return None
            cam_sample = self.camera_dataset[idx]
            lidar_sample = self.lidar_dataset[idx]
            print(f"[DEBUG] cam_sample keys: {cam_sample.keys() if cam_sample else 'None'}")
            print(f"[DEBUG] lidar_sample keys: {lidar_sample.keys() if lidar_sample else 'None'}")
            if cam_sample is None or lidar_sample is None:
                return None
            return {**cam_sample, **lidar_sample, "label": 1}  # dummy label
        except Exception as e:
            print(f"[ERROR] Dataset error at index {idx}: {e}")
            return None

    def __len__(self):
        return min(len(self.camera_dataset), len(self.lidar_dataset))

    def collater(self, samples):
        print(f"[DEBUG] collater received {len(samples)} samples")
        
        for i, s in enumerate(samples):
            print(f"  sample {i}: keys = {s.keys() if s else 'None'}")
        
        samples = [s for s in samples if s is not None]

        if len(samples) == 0:
            print("[WARNING] All samples were None!")
            return {}
        batch = self.camera_dataset.collater(samples)
        batch.update({"lidar": torch.stack([s["lidar"] for s in samples])})
        batch["label"] = torch.tensor([s["label"] for s in samples])
        print(f"[DEBUG] Final collated batch keys+++++++++++++++++++: {batch.keys()}")
        return batch


class WaymoDatasetBuilder:
    def __init__(self, cfg):
        self.config = cfg
        self.data_type = list(cfg.data_type)

    def build_datasets(self):
        build_info = self.config.build_info
        vis_processor = Blip2ImageTrainProcessor.from_config(self.config.vis_processor.train)

        camera_dataset = WaymoCameraDataset(vis_processor, build_info["camera"].storage)
        lidar_dataset = WaymoLidarDataset(vis_processor, build_info["lidar"].storage)

        # return {
        #     "waymo": {
        #         "train": WrappedConcatDataset([camera_dataset, lidar_dataset])
        #     }
        # }
        return {
            "train": WrappedConcatDataset([camera_dataset, lidar_dataset])
    }


if __name__ == "__main__":

    # Set test paths
    camera_data_path = "lavis/my_datasets/waymo/train/tiny_camera_image"
    lidar_data_path = "lavis/my_datasets/waymo/train/tiny_lidar"

    # Dummy image processor (you can replace this with the actual processor if needed)
    dummy_processor = lambda img: img

    print("\n[Testing WaymoCameraDataset]")
    camera_dataset = WaymoCameraDataset(
        vis_processor=dummy_processor,
        camera_root=camera_data_path
    )

    for i in range(min(5, len(camera_dataset))):
        sample = camera_dataset[i]
        if sample:
            print(f"[Camera] Image ID: {sample['image_id']}")
            plt.imshow(sample['image'])
            plt.title(f"Camera Sample {i}")
            plt.axis('off')
            plt.show()

    print("\n[Testing WaymoLidarDataset]")
    lidar_dataset = WaymoLidarDataset(
        vis_processor=dummy_processor,
        lidar_root=lidar_data_path
    )

    for i in range(min(5, len(lidar_dataset))):
        sample = lidar_dataset[i]
        if sample:
            print(f"[LiDAR] Image ID: {sample['image_id']}")
            plt.imshow(sample['lidar'])
            plt.title(f"LiDAR Sample {i}")
            plt.axis('off')
            plt.show()

    print("\n[Testing WrappedConcatDataset]")
    combined_dataset = WrappedConcatDataset([camera_dataset, lidar_dataset])
    for i in range(min(5, len(combined_dataset))):
        sample = combined_dataset[i]
        if sample:
            print(f"[Combined] Image ID: {sample['image_id']}")
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(sample['image'])
            axs[0].set_title("Camera")
            axs[1].imshow(sample['lidar'])
            axs[1].set_title("LiDAR")
            for ax in axs:
                ax.axis('off')
            plt.show()


























