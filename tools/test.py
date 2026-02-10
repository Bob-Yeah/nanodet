# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import os
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import onnxruntime as ort

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    load_config,
    mkdir,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="test", help="task to run, test or val"
    )
    parser.add_argument("--config", type=str, help="model config file(.yml) path")
    parser.add_argument("--model", type=str, help="ckeckpoint file(.ckpt) path")
    args = parser.parse_args()
    return args


def main(args):
    load_config(cfg, args.config)
    local_rank = -1
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    cfg.defrost()
    timestr = datetime.datetime.now().__format__("%Y%m%d%H%M%S")
    cfg.save_dir = os.path.join(cfg.save_dir, timestr)
    mkdir(local_rank, cfg.save_dir)
    logger = NanoDetLightningLogger(cfg.save_dir)

    assert args.task in ["val", "test"]
    cfg.update({"test_mode": args.task})

    logger.info("Setting up data...")
    if (args.task == "test"):
        val_dataset = build_dataset(cfg.data.test, args.task)
    else:
        val_dataset = build_dataset(cfg.data.val, args.task)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )
    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    logger.info("Creating model...")
    
    # Check if the model is ONNX format
    if args.model.endswith('.onnx'):
        logger.info("Loading ONNX model using ONNX Runtime...")
        try:
            # Create ONNX Runtime session
            onnx_session = ort.InferenceSession(args.model)
            logger.info("ONNX model loaded successfully")
            
            # Get input and output names
            input_name = onnx_session.get_inputs()[0].name
            output_names = [output.name for output in onnx_session.get_outputs()]
            
            logger.info(f"ONNX model input: {input_name}")
            logger.info(f"ONNX model outputs: {output_names}")
            
            # Create a wrapper class for ONNX model to work with PyTorch Lightning
            class ONNXModelWrapper(pl.LightningModule):
                def __init__(self, session, input_name, output_names):
                    super().__init__()
                    self.session = session
                    self.input_name = input_name
                    self.output_names = output_names
                
                def forward(self, x):
                    # Convert PyTorch tensor to numpy array
                    if isinstance(x, torch.Tensor):
                        x_np = x.detach().cpu().numpy()
                    else:
                        x_np = x
                    
                    # Run inference
                    outputs = self.session.run(self.output_names, {self.input_name: x_np})
                    
                    # Convert outputs to PyTorch tensors
                    return [torch.from_numpy(output) for output in outputs]
                
                def test_step(self, batch, batch_idx):
                    # Preprocess batch input similar to TrainingTask
                    batch_imgs = batch["img"]
                    if isinstance(batch_imgs, list):
                        batch_imgs = [img.to(self.device) for img in batch_imgs]
                        from nanodet.data.batch_process import stack_batch_img
                        batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
                        batch["img"] = batch_img_tensor
                    
                    # Run inference
                    outputs = self(batch["img"])
                    
                    # Return outputs in the same format as TrainingTask
                    return outputs
            
            task = ONNXModelWrapper(onnx_session, input_name, output_names)
            
        except Exception as e:
            logger.info(f"Failed to load ONNX model: {e}")
            raise RuntimeError(f"ONNX model loading failed: {e}")
    else:
        # Load PyTorch checkpoint
        task = TrainingTask(cfg, evaluator)
        ckpt = torch.load(args.model)
        if "pytorch-lightning_version" not in ckpt:
            warnings.warn(
                "Warning! Old .pth checkpoint is deprecated. "
                "Convert the checkpoint with tools/convert_old_checkpoint.py "
            )
            ckpt = convert_old_model(ckpt)
        task.load_state_dict(ckpt["state_dict"])

    if cfg.device.gpu_ids == -1:
        logger.info("Using CPU training")
        accelerator, devices = "cpu", None
    else:
        accelerator, devices = "gpu", cfg.device.gpu_ids

    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        logger=logger,
    )
    logger.info("Starting testing...")
    trainer.test(task, val_dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
