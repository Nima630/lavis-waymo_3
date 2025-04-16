"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass


    # def build_datasets(self, cfg):
    #     datasets = super().build_datasets(cfg)

    #     # Flatten nested datasets like {"coco_caption": {"train": ..., "val": ...}}
    #     flat_datasets = {}

    #     for key, value in datasets.items():
    #         if isinstance(value, dict):
    #             for split_name, dset in value.items():
    #                 flat_datasets[split_name] = dset
    #         else:
    #             flat_datasets[key] = value

    #     print(">>> [DEBUG] Flattened datasets keys:", flat_datasets.keys())
    #     return flat_datasets



    # def build_datasets(self, cfg):
    #     # this will return something like:
    #     # {
    #     #   "coco_caption": {"train": <Dataset>, "val": ..., "test": ...},
    #     #   "vg_caption": {"train": <Dataset>}
    #     # }

    #     datasets = super().build_datasets(cfg)

    #     # Now reorganize it to:
    #     # {
    #     #   "train": [<coco_train>, <vg_train>],
    #     #   "val": [<coco_val>],
    #     #   ...
    #     # }

    #     merged_datasets = {}

    #     for dataset_name, splits in datasets.items():
    #         for split, dset in splits.items():
    #             if split not in merged_datasets:
    #                 merged_datasets[split] = []
    #             merged_datasets[split].append(dset)

    #     print(">>> [DEBUG] Merged datasets:", {k: len(v) for k, v in merged_datasets.items()})
    #     return merged_datasets


    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # ✅ No flattening or merging — return as-is
        # Let LAVIS reorg it properly based on original builder structure
        # for dataset_name, splits in datasets.items():
        #     for split_name, split_data in splits.items():
        #         print(f">>> [DEBUG] Dataset: {dataset_name} | Split: {split_name} | Type: {type(split_data)} | Length: {len(split_data)}")

        for name, split_dict in datasets.items():
            for split, dset in split_dict.items():
                print(f">>> [DEBUG] Dataset: {name} | Split: {split} | Type: {type(dset)} | Length: {len(dset)}")


        # print(">>> [DEBUG] Keeping datasets structure:", {k: list(v.keys()) for k, v in datasets.items()})
        return datasets
