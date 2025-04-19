import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from lavis.datasets.waymo import WaymoDatasetBuilder

from lavis.models.blip2_models.blip2_qformer_w import Blip2Qformer  # ✅ your custom model
from lavis.tasks.base_task import BaseTask
from lavis.common.config_w import Config  # ✅ your new config_w.py
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# Optional: if you're keeping runners for now
from lavis.runners.runner_base import RunnerBase

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override options in YAML with key=value",
    )
    return parser.parse_args()

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    job_id = now()
    print("[TRACE] Starting train_w.py")
    args = parse_args()
    cfg = Config(args)


    init_distributed_mode(cfg.run_cfg)
    cfg.run_cfg.gpu = int(os.environ.get("LOCAL_RANK", 0))
    setup_seeds(cfg)
    setup_logger()
    print("[TRACE] Config initialized")
    print("[DEBUG] Full config:\n", cfg.pretty_print())
    # Use unified WaymoDatasetBuilder
    task = BaseTask()
    builder = WaymoDatasetBuilder(cfg.datasets_cfg["waymo"])
    # datasets = builder.build_datasets()
    # # train_dataset = datasets.get("train")
    # train_dataset = datasets.get("waymo", {}).get("train")

    # print(">>> [DEBUG] Loaded train dataset with", len(train_dataset), "samples")


    datasets = builder.build_datasets()
    # train_dataset = datasets.get("train")
    train_dataset = datasets.get("waymo", {}).get("train")


    if train_dataset is None:
        raise ValueError("Train dataset not found in builder output.")

    print(">>> [DEBUG] Loaded train dataset with", len(train_dataset), "samples")



    model = Blip2Qformer.from_config(cfg.model_cfg)
    runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty. Check your input paths or data content.")

    runner.train()

# def main():
#     # print("[TRACE] Starting train_w.py")
    
#     job_id = now()

#     cfg = Config(parse_args())  # ✅ uses config_w
#     init_distributed_mode(cfg.run_cfg)
#     setup_seeds(cfg)
#     setup_logger()
#     cfg.pretty_print()
#     print("\n>>> [DEBUG] Full config:\n", cfg.pretty_print())
#     # ✅ create task and dataset
#     task = BaseTask()
#     datasets = task.build_datasets(cfg)

#     # Choose one or both based on your config/data_type
#     builder = WaymoCameraBuilder(cfg.datasets_cfg["waymo"])
#     datasets = builder.build_datasets()
#     print(">>> [DEBUG] Custom Waymo datasets loaded:", datasets.keys())

#     # train_dataset = datasets.get("waymo", {}).get("train")
#     train_dataset = datasets.get("waymo_camera")


#     if train_dataset is None:
#         raise ValueError("Train dataset not found in 'waymo'.")
#     print(">>> [DEBUG] Train dataset length:", len(train_dataset))

#     # ✅ build model directly
#     model = Blip2Qformer.from_config(cfg.model_cfg)

#     # ✅ build runner directly
#     runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
#     runner.train()



if __name__ == "__main__":
    main()





























# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# from lavis.datasets.waymo import WaymoCameraBuilder, WaymoLidarBuilder

# from lavis.models.blip2_models.blip2_qformer_w import Blip2Qformer  # ✅ your custom model
# from lavis.tasks.base_task import BaseTask
# from lavis.common.config_w import Config  # ✅ your new config_w.py
# from lavis.common.dist_utils import get_rank, init_distributed_mode
# from lavis.common.logger import setup_logger
# from lavis.common.optims import (
#     LinearWarmupCosineLRScheduler,
#     LinearWarmupStepLRScheduler,
# )
# from lavis.common.utils import now

# # Optional: if you're keeping runners for now
# from lavis.runners.runner_base import RunnerBase


# def parse_args():
#     parser = argparse.ArgumentParser(description="Training")
#     parser.add_argument("--cfg-path", required=True, help="Path to YAML config.")
#     parser.add_argument(
#         "--options",
#         nargs="+",
#         help="Override options in YAML with key=value",
#     )
#     return parser.parse_args()


# def setup_seeds(config):
#     seed = config.run_cfg.seed + get_rank()
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     cudnn.benchmark = False
#     cudnn.deterministic = True


# def main():
#     print("[TRACE] Starting train_w.py")
#     args = parse_args()
#     cfg = Config(args)

#     print(">>> [DEBUG] Full config:\n", cfg.pretty_print())

#     # Use custom builder directly
#     builder = WaymoCameraBuilder(cfg.datasets_cfg["waymo"])
#     datasets = builder.build_datasets()
#     train_dataset = datasets.get("train")

#     print(">>> [DEBUG] Loaded train dataset with", len(train_dataset), "samples")

# # def main():
# #     # print("[TRACE] Starting train_w.py")
    
# #     job_id = now()

# #     cfg = Config(parse_args())  # ✅ uses config_w
# #     init_distributed_mode(cfg.run_cfg)
# #     setup_seeds(cfg)
# #     setup_logger()
# #     cfg.pretty_print()
# #     print("\n>>> [DEBUG] Full config:\n", cfg.pretty_print())
# #     # ✅ create task and dataset
# #     task = BaseTask()
# #     datasets = task.build_datasets(cfg)

# #     # Choose one or both based on your config/data_type
# #     builder = WaymoCameraBuilder(cfg.datasets_cfg["waymo"])
# #     datasets = builder.build_datasets()
# #     print(">>> [DEBUG] Custom Waymo datasets loaded:", datasets.keys())

# #     # train_dataset = datasets.get("waymo", {}).get("train")
# #     train_dataset = datasets.get("waymo_camera")


# #     if train_dataset is None:
# #         raise ValueError("Train dataset not found in 'waymo'.")
# #     print(">>> [DEBUG] Train dataset length:", len(train_dataset))

# #     # ✅ build model directly
# #     model = Blip2Qformer.from_config(cfg.model_cfg)

# #     # ✅ build runner directly
# #     runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
# #     runner.train()


# if __name__ == "__main__":
#     main()




# import argparse
# import os
# import random

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# from lavis.models.blip2_models.blip2_qformer_w import Blip2Qformer  # import directly
# from lavis.tasks.base_task import BaseTask
# from lavis.common.config import Config
# from lavis.common.dist_utils import get_rank, init_distributed_mode
# from lavis.common.logger import setup_logger
# from lavis.common.optims import (
#     LinearWarmupCosineLRScheduler,
#     LinearWarmupStepLRScheduler,
# )
# from lavis.common.registry import registry
# from lavis.common.utils import now

# from lavis.datasets.builders import *
# from lavis.models import *
# from lavis.processors import *
# from lavis.runners import *

# def parse_args():
#     parser = argparse.ArgumentParser(description="Training")
#     parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
#     parser.add_argument(
#         "--options",
#         nargs="+",
#         help="override settings from config using key=value pairs",
#     )
#     return parser.parse_args()

# def setup_seeds(config):
#     seed = config.run_cfg.seed + get_rank()
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     cudnn.benchmark = False
#     cudnn.deterministic = True

# def get_runner_class(cfg):
#     runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
#     return runner_cls

# def main():
#     print("[TRACE] Starting train_w.py")
#     job_id = now()

#     cfg = Config(parse_args())
#     init_distributed_mode(cfg.run_cfg)
#     setup_seeds(cfg)
#     setup_logger()
#     cfg.pretty_print()

#     task = BaseTask()
#     datasets = task.build_datasets(cfg)
#     train_dataset = datasets.get("waymo", {}).get("train")
#     if train_dataset is None:
#         raise ValueError("Train dataset not found under 'waymo'.")
#     else:
#         print("[DEBUG] Train dataset loaded successfully.")

#     # model = task.build_model(cfg)
#     model_cfg = cfg.model_cfg
#     model = Blip2QformerW.from_config(model_cfg)
#     runner = get_runner_class(cfg)(
#         cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
#     )
#     runner.train()

# if __name__ == "__main__":
#     main()
