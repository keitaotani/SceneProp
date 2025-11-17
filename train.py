import os
import argparse
import yaml
from pathlib import Path
from collections import OrderedDict
import subprocess
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from sge.model import RELRCNN

from utils import flatten_dict, AverageMeter

from sge.dataset.visual_genome import VGSubgraph, VG_VLMPAG, collate_fn
from sge.dataset.coco_stuff import COCOStuff
from sge.dataset.augmentation import DataAugmentationForSG
from sge.dataset.gqa import GQASceneGraphDataset
from sge.loss.scene_graph_matching_loss import CalculateLosses as SGMCalculateLosses


def to_float(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    else:
        return x


class UnifiedLossCalculator:
    def __init__(self, loss_weights, sg_dataiter, sgm_calculate_losses):
        self.loss_weights = loss_weights
        self.sg_dataiter = sg_dataiter
        self.sgm_calculate_losses = sgm_calculate_losses
    
    def __call__(self):
        b_imgs, b_boxes, b_rels, b_preds, b_scenelabels = next(self.sg_dataiter)
        losses = self.sgm_calculate_losses(b_imgs, b_boxes, b_rels, b_preds, b_scenelabels)

        losses = flatten_dict(losses, joinstr="/")

        for loss in losses.values():
            if isinstance(loss, torch.Tensor):
                device = loss.device
                break

        for key in losses.keys():
            if not isinstance(losses[key], torch.Tensor):
                losses[key] = torch.tensor(losses[key], device=device)

        total_loss = 0.0
        for key, weight in self.loss_weights.items():
            total_loss += losses[key] * weight
        losses["total"] = total_loss

        return losses


def main(config, resume=None):
    gpuid = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(gpuid)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank)

    config["world_size"] = world_size

    ### Load pretrained model
    pretrained_weights = OrderedDict()
    for keyname, weight in torch.load(config["train"]["pretrained_model"], map_location=torch.device(gpuid))["model"].items():
        if "module.backbone" in keyname:
            keyname = keyname.replace("module.backbone.", "backbone.")
            pretrained_weights[keyname] = weight

    ### Load dataset
    if config["dataset"]["using_dataset"] == 'visual_genome':
        sg_dataset = VGSubgraph(
            split='train',
            path_imdb     = config["dataset"]['visual_genome']["path_imdb"],
            path_sgg      = config["dataset"]['visual_genome']["path_sgg"],
            path_category = config["dataset"]['visual_genome']["path_category"],
            transform     = DataAugmentationForSG(config["train"]["data_augmentation"])
            )
        sg_valdataset = VGSubgraph(
            split='test',
            path_imdb     = config["dataset"]['visual_genome']["path_imdb"],
            path_sgg      = config["dataset"]['visual_genome']["path_sgg"],
            path_category = config["dataset"]['visual_genome']["path_category"],
            transform     = DataAugmentationForSG(config["train"]["data_augmentation"], val=True)
            )
    elif config["dataset"]["using_dataset"] == 'VGFO':
        sg_dataset = VG_VLMPAG(
            split='train',
            path_imdb     = config["dataset"]['VGFO']["path_imdb"],
            path_sg       = config["dataset"]['VGFO']["path_train_sg"],
            path_category = config["dataset"]['VGFO']["path_category"],
            transform     = DataAugmentationForSG(config["train"]["data_augmentation"])
            )
        sg_valdataset = VG_VLMPAG(
            split='val',
            path_imdb     = config["dataset"]['VGFO']["path_imdb"],
            path_sg       = config["dataset"]['VGFO']["path_train_sg"],
            path_category = config["dataset"]['VGFO']["path_category"],
            transform     = DataAugmentationForSG(config["train"]["data_augmentation"], val=True)
            )
    elif config["dataset"]["using_dataset"] == 'VGPO':
        sg_dataset = VG_VLMPAG(
            split='train',
            path_imdb     = config["dataset"]['VGPO']["path_imdb"],
            path_sg       = config["dataset"]['VGPO']["path_train_sg"],
            path_category = config["dataset"]['VGPO']["path_category"],
            partial_cat   = True,
            transform     = DataAugmentationForSG(config["train"]["data_augmentation"])
            )
        sg_valdataset = VG_VLMPAG(
            split='val',
            path_imdb     = config["dataset"]['VGPO']["path_imdb"],
            path_sg       = config["dataset"]['VGPO']["path_train_sg"],
            path_category = config["dataset"]['VGPO']["path_category"],
            partial_cat   = True,
            transform     = DataAugmentationForSG(config["train"]["data_augmentation"], val=True)
            )
    elif config["dataset"]["using_dataset"] == 'COCOStuff':
        sg_dataset = COCOStuff(
            image_dir = config["dataset"]['COCOStuff']["image_dir_train"],
            annotation_file = config["dataset"]['COCOStuff']["path_train_sg"],
            category_file = config["dataset"]['COCOStuff']["path_category"],
            transform = DataAugmentationForSG(config["train"]["data_augmentation"])
            )
        sg_valdataset = COCOStuff(
            image_dir = config["dataset"]['COCOStuff']["image_dir_val"],
            annotation_file = config["dataset"]['COCOStuff']["path_val_sg"],
            category_file = config["dataset"]['COCOStuff']["path_category"],
            transform = DataAugmentationForSG(config["train"]["data_augmentation"], val=True)
            )
    elif config["dataset"]["using_dataset"] == 'GQA_SceneGraph':
        sg_dataset = GQASceneGraphDataset(
            image_dir        = config["dataset"]['GQA_SceneGraph']["image_dir"],
            scene_graph_path = config["dataset"]['GQA_SceneGraph']["path_train_sg"],
            transform        = DataAugmentationForSG(config["train"]["data_augmentation"])
            )
        sg_valdataset = GQASceneGraphDataset(
            image_dir        = config["dataset"]['GQA_SceneGraph']["image_dir"],
            scene_graph_path = config["dataset"]['GQA_SceneGraph']["path_val_sg"],
            transform        = DataAugmentationForSG(config["train"]["data_augmentation"], val=True)
            )
    else:
        raise ValueError("Unknown dataset")
    
    sg_sampler = torch.utils.data.distributed.DistributedSampler(
        sg_dataset,
        num_replicas=world_size,
        rank = rank,
        shuffle=True)
    sg_valsampler = torch.utils.data.distributed.DistributedSampler(
        sg_valdataset,
        num_replicas=world_size,
        rank = rank,
        shuffle=False)

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    sg_dataiter = cycle(torch.utils.data.DataLoader(
        sg_dataset,
        batch_size=config["train"]["batch_size"],
        sampler=sg_sampler,
        num_workers=8,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True
    ))
    sg_valdataiter = cycle(torch.utils.data.DataLoader(
        sg_valdataset,
        batch_size=config["train"]["batch_size"],
        sampler=sg_valsampler,
        num_workers=1,
        collate_fn=collate_fn,
        drop_last=True
    ))

    objcat_names, relcat_names, scenecat_names = sg_dataset.get_category_names()
    is_object_cat = torch.tensor([name in sg_dataset.get_object_category_names() for name in objcat_names], dtype=torch.bool, device="cuda")

    model = RELRCNN(
        n_objcat = len(objcat_names),
        n_relcat = len(relcat_names),
        n_scene_cat = len(scenecat_names) if 'scene_classification' in config['train']['loss_weights'] else 0,
        n_proposals = config["max_nmsed_per_img"],
        nms_thresh = config["nms_thresh"],
        num_transformer_layers = config["num_transformer_layers"] if "num_transformer_layers" in config else 4,
        pe_type = config["pe_type"] if "pe_type" in config else "random_fourier"
    ).cuda()
    model.load_state_dict(pretrained_weights, strict=False)
    del pretrained_weights
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpuid], broadcast_buffers=False, find_unused_parameters=True)
    model.train()

    if "optimizer" not in config["train"] or config["train"]["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["train"]["lr"])
    elif config["train"]["optimizer"] == "MomentumSGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["train"]["lr"],
            momentum=0.9,
            weight_decay=1e-4)
    iteration = 1

    sgm_calculate_losses = SGMCalculateLosses(
        model,
        is_object_cat,
        config["train"]["category_drop_rate"],
        interbatch=config["train"]["interbatch"],
        box_classification_loss_func=config["train"]["object_classification_loss"])

    if resume is not None:
        checkpoint = torch.load(resume)
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iteration = checkpoint["iteration"] + 1

    if rank == 0:
        save_path = Path(config["save_path"])
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "config.yaml", "w") as f:
            yaml.dump(config, f)
        with open(save_path / "commit_hash.txt", "w") as f:
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            f.write(commit_hash)
        writer = SummaryWriter(save_path)

    average_meter = AverageMeter()
    average_meter_val = AverageMeter()

    loss_weights = flatten_dict(config["train"]["loss_weights"], joinstr="/")

    calculate_losses = UnifiedLossCalculator(
        loss_weights,
        sg_dataiter,
        sgm_calculate_losses)
    calculate_losses_val = UnifiedLossCalculator(
        loss_weights,
        sg_valdataiter,
        sgm_calculate_losses)

    if rank == 0:
        pbar = tqdm(total=config["train"]["max_iter"], dynamic_ncols=True, desc=str(save_path))
        pbar.update(iteration)

    while iteration <= config["train"]["max_iter"]:

        while average_meter.count() < config["train"]["grad_accumulation_steps"]:
            if rank == 0:
                pbar.set_postfix({"accum": f"{average_meter.count()}/{config['train']['grad_accumulation_steps']}"})

            if average_meter.count() < config["train"]["grad_accumulation_steps"] - 1:
                with model.no_sync():
                    losses = calculate_losses()
                    losses["total"].backward()
            else:
                losses = calculate_losses()
                losses["total"].backward()

            losses = {key: value.detach() for key, value in losses.items()}
            average_meter.update(losses)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()

        average_meter.synchronize()
        if rank == 0:
            tqdm.write(
                f"Iteration: {iteration}\t"
                f"Loss:{average_meter['total']: >8.4f}")
            for key, value in average_meter.items():
                writer.add_scalar(f"train_loss/{key}", value, iteration)
            pbar.update(1)
        iteration += 1
        average_meter.reset()

        # Validation
        if iteration % 10 == 0:
            with torch.inference_mode():
                while average_meter_val.count() < config["train"]["grad_accumulation_steps"]:
                    if rank == 0:
                        pbar.set_postfix({"accum_val": f"{average_meter_val.count()}/{config['train']['grad_accumulation_steps']}"})

                    losses = calculate_losses_val()

                    losses = {key: value.detach() for key, value in losses.items()}
                    average_meter_val.update(losses)

                average_meter_val.synchronize()
                if rank == 0:
                    for key, value in average_meter_val.items():
                        writer.add_scalar(f"val_loss/{key}", value, iteration)
                average_meter_val.reset()

        if rank == 0:
            if iteration % 500 == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration
                }
                torch.save(checkpoint, save_path / "checkpoint.pth")

            if iteration % config["train"]["save_freq_iter"] == 0:
                torch.save(model.module.state_dict(), save_path / f"iter_{iteration:07d}.pth")


if __name__ == "__main__":
    # use torchrun to run this script
    # torchrun --nnodes=1 --nproc_per_node=4 train.py"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config, args.resume)