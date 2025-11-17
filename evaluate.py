import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sge.model import RELRCNN
from sge.graphical_model.estimator import MLEstimator, calculate_rank_of_true
from sge.data_container import ImageList
from sge.dataset.visual_genome import VGSubgraph, VG_VLMPAG
from sge.dataset.augmentation import DataAugmentationForSG
from sge.dataset.gqa import GQASceneGraphDataset
from sge.dataset.coco_stuff import COCOStuff


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--without-rel', action='store_true')
    parser.add_argument('--no-label', action='store_true')
    args = parser.parse_args()

    model_path = Path(args.model)
    if args.config is None:
        config_path = model_path.parent / 'config.yaml'
        config = yaml.safe_load(config_path.open())
    else:
        config = yaml.safe_load(Path(args.config).open())
    if args.output is None:
        outputname = "output"
        if args.without_rel:
            outputname += "_without_rel"
        if args.no_label:
            outputname += "_no_label"
        output_path = model_path.parent / f"{outputname}.csv"
    else:
        assert args.output.endswith('.csv')
        output_path = Path(args.output)

    eval_per_n_edges = False

    if config["dataset"]["using_dataset"] == 'visual_genome':
        dataset = VGSubgraph(
            split='test',
            path_imdb     = config["dataset"]['visual_genome']["path_imdb"],
            path_sgg      = config["dataset"]['visual_genome']["path_sgg"],
            path_category = config["dataset"]['visual_genome']["path_category"],
            transform     = DataAugmentationForSG(config["train"]["data_augmentation"], evaluate=True))
    elif config["dataset"]["using_dataset"] == 'VGFO':
        dataset = VG_VLMPAG(
            split='all',
            path_imdb     = config["dataset"]['VGFO']["path_imdb"],
            path_sg       = config["dataset"]['VGFO']["path_test_sg"],
            path_category = config["dataset"]['VGFO']["path_category"],
            transform     = DataAugmentationForSG(config["train"]["data_augmentation"], evaluate=True))
        eval_per_n_edges = True
    elif config["dataset"]["using_dataset"] == 'VGPO':
        dataset = VG_VLMPAG(
            split='all',
            path_imdb     = config["dataset"]['VGPO']["path_imdb"],
            path_sg       = config["dataset"]['VGPO']["path_test_sg"],
            path_category = config["dataset"]['VGPO']["path_category"],
            transform     = DataAugmentationForSG(config["train"]["data_augmentation"], evaluate=True))
        eval_per_n_edges = True
    elif config["dataset"]["using_dataset"] == 'GQA_SceneGraph':
        dataset = GQASceneGraphDataset(
            image_dir        = config["dataset"]['GQA_SceneGraph']["image_dir"],
            scene_graph_path = config["dataset"]['GQA_SceneGraph']["path_val_sg"],
            transform        = DataAugmentationForSG(config["train"]["data_augmentation"], evaluate=True))
    elif config["dataset"]["using_dataset"] == 'COCOStuff':
        dataset = COCOStuff(
            image_dir = config["dataset"]['COCOStuff']["image_dir_val"],
            annotation_file = config["dataset"]['COCOStuff']["path_val_sg"],
            category_file = config["dataset"]['COCOStuff']["path_category"],
            transform = DataAugmentationForSG(config["train"]["data_augmentation"], val=True)
            )
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']['using_dataset']}")

    objcat_names, relcat_names, scene_categories = dataset.get_category_names()
    if config["dataset"]["using_dataset"] == 'VGPO':
        model_objcat_names = dataset.partial_object_names
        to_model_catid = {}
        for i, objcat_name in enumerate(objcat_names):
            if objcat_name in model_objcat_names:
                to_model_catid[i] = model_objcat_names.index(objcat_name)
            else:
                to_model_catid[i] = len(model_objcat_names)
    else:
        model_objcat_names = objcat_names
        to_model_catid = None

    model = RELRCNN(
        n_objcat = len(model_objcat_names),
        n_relcat = len(relcat_names),
        n_proposals = config["max_nmsed_per_img"],
        nms_thresh = config["nms_thresh"],
        num_transformer_layers = config["num_transformer_layers"] if "num_transformer_layers" in config else 4,
        pe_type = config["pe_type"] if "pe_type" in config else "random_fourier"
    ).to("cuda")
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=lambda x: x[0],
        pin_memory=True
    )

    ml_estimator = MLEstimator(
        bp_iter=10,
        n_candidates=100
    )

    ranks_of_true = []
    n_edges = []
    unseens = []

    if args.no_label:
        label_retrieval_ranks = []

    for img, boxes, rels, preds, scenelabels in tqdm(dataloader, dynamic_ncols=True):
        if len(boxes) == 0:
            continue  # images without objects don't affect the evaluation score

        with torch.inference_mode():
            (
                object_classes,  # For object classification
                relationship_classes, # For relationship classification
                scene_classes,  # For scene classification
                pred_boxes
            ) = model.inference(
                ImageList(img[None], [(img.shape[2], img.shape[1])])
            )

            if to_model_catid is not None:
                object_classes = torch.cat([
                    object_classes[0],
                    object_classes[0].logsumexp(dim=-1, keepdim=True)
                ], dim=-1)
                labels = boxes.get_field('labels').numpy()
                labels = np.array([to_model_catid[l] for l in labels])
                labels = torch.from_numpy(labels).to(torch.int64).cuda()
            else:
                object_classes = object_classes[0]
                labels = boxes.get_field('labels').cuda()
            
            relationship_classes = relationship_classes[0]

            if args.without_rel:
                rels = torch.empty((0, 2), dtype=torch.int32)
                preds = torch.empty((0, ), dtype=torch.int32)

            if args.no_label:
                for i in range(len(labels)):
                    labels_with_nolabel = labels.clone()
                    labels_with_nolabel[i] = -1

                    i_pred_anchor_mle = ml_estimator(
                        object_classes,
                        relationship_classes, 
                        labels_with_nolabel,
                        rels.cuda(),
                        preds.cuda()
                    )
                    pred_boxes_mle = pred_boxes[0].bbox[i_pred_anchor_mle]
                    rank = calculate_rank_of_true(pred_boxes_mle, boxes.bbox.cuda(non_blocking=True))[i]
                    ranks_of_true.append(rank)

                    #labellogit = object_classes[i_pred_anchor_mle[i]].logsumexp(dim=0)
                    labellogit = object_classes[i_pred_anchor_mle[i, 0]]
                    label_candidate = object_classes[i_pred_anchor_mle[i, 0]].argsort(descending=True)
                    rank = label_candidate.tolist().index(labels[i].item())
                    label_retrieval_ranks.append(rank)
            else:
                i_pred_anchor_mle = ml_estimator(
                    object_classes,
                    relationship_classes, 
                    labels,
                    rels.cuda(),
                    preds.cuda()
                )
                pred_boxes_mle = pred_boxes[0].bbox[i_pred_anchor_mle]

                ranks_of_true += calculate_rank_of_true(pred_boxes_mle, boxes.bbox.cuda(non_blocking=True))

            if eval_per_n_edges:
                n_edges += boxes.get_field('n_edges').numpy().tolist()

        unseens += (labels == len(model_objcat_names)).cpu().numpy().tolist()

    rank_of_true = np.array(ranks_of_true)
    n_edges = np.array(n_edges)
    unseens = np.array(unseens)
    seens = ~unseens

    result = {}

    for prefix, seemask in [("", seens), ("Unseen ", unseens)]:
        if np.sum(seemask) == 0:
            continue

        r = rank_of_true[seemask]
        for k in [1, 5, 10, 20, 50, 100]:
            recall = np.mean((r < k) & (r >= 0))
            print(f"{prefix}Object Recall@{k}: {recall}")
            result[f"{prefix}Object Recall@{k}"] = recall

        if eval_per_n_edges:
            for n_edge in [1, 2, 3, 4, 5, 6, 7, 8]:
                mask = (n_edges == n_edge) & seemask
                r = rank_of_true[mask]
                for k in [1, 5]:
                    recall = np.mean((r < k) & (r >= 0))
                    print(f"{prefix}Object Recall@{k} (n_edges={n_edge}): {recall}")
                    result[f"{prefix}Object Recall@{k} (n_edges={n_edge})"] = recall
        
        if args.no_label:
            r = np.array(label_retrieval_ranks)[seemask]
            for k in [1, 5]:
                recall = np.mean((r < k) & (r >= 0))
                print(f"{prefix}Label Retrieval Recall@{k}: {recall}")
                result[f"{prefix}Label Retrieval Recall@{k}"] = recall

    if output_path.exists():
        output = pd.read_csv(output_path)
    else:
        output = pd.DataFrame()

    output = pd.concat([
        output,
        pd.DataFrame([{
            "ModelName": model_path.name,
            **result
        }])
    ], ignore_index=True)

    output = output.sort_values(by="ModelName")
    output.to_csv(output_path, index=False)