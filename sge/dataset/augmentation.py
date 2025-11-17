import numpy as np
import networkx as nx

import torch
from torchvision.transforms import ColorJitter
from sge.data_container import BoxList


class RandomFlip:
    def __init__(self, impose_rate):
        self.impose_rate = impose_rate
    
    def __call__(self, img, boxes, *args):
        if np.random.rand() > self.impose_rate:
            return img, boxes, *args
        
        img = img.flip(-1)
        boxes = boxes.transpose(0)  # FLIP_LEFT_RIGHT = 0
        return img, boxes, *args


class AddGaussianNoise:
    def __init__(self, impose_rate, max_std):
        self.impose_rate = impose_rate
        self.max_std = max_std
    
    def __call__(self, img, boxes, *args):
        if np.random.rand() > self.impose_rate:
            return img, boxes, *args
        
        std = torch.rand(1) * self.max_std
        noise = torch.randn_like(img) * std
        img = torch.clamp(img + noise, 0, 1)
        return img, boxes, *args


class RandomColorJitter:
    def __init__(self, impose_rate, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.impose_rate = impose_rate
        self.jitter = ColorJitter(brightness, contrast, saturation, hue)
    
    def __call__(self, img, boxes, *args):
        if np.random.rand() > self.impose_rate:
            return img, boxes, *args
        
        img = torch.clamp(img, 0., 1.)
        img = self.jitter(img)
        return img, boxes, *args


class RandomResize:
    def __init__(self, impose_rate, min_scale=0.5, max_scale=1.0, aspect_shift=0.1, base_long_side=800):
        assert aspect_shift >= 0
        aspect_shift = min(aspect_shift, max_scale /  min_scale - 1)
        self.impose_rate = impose_rate
        self.min_scale = np.array([min_scale, min_scale])
        self.max_scale = np.array([max_scale, max_scale])
        self.min_asp_scale = np.array([min_scale, min_scale * (1 + aspect_shift)])
        self.max_asp_scale = np.array([max_scale / (1 + aspect_shift), max_scale])
        self.modes = ['nearest-exact', 'bilinear', 'bicubic', 'area']
        self.base_long_side = base_long_side
    
    def __call__(self, img, boxes, *args):
        _, H, W = img.shape
        long_side = max(H, W)
        base_scale = self.base_long_side / long_side

        if np.random.rand() > self.impose_rate:
            resized_H, resized_W = int(H * base_scale), int(W * base_scale)
            img = torch.nn.functional.interpolate(img[None], size=(resized_H, resized_W), mode='bilinear')[0]
            boxes = boxes.resize((resized_W, resized_H))
            return img, boxes, *args
        else:
            r1 = np.random.rand()
            r2 = np.random.rand()
            scale =  r2      * (r1 * self.min_scale     + (1 - r1) * self.max_scale    ) + \
                    (1 - r2) * (r1 * self.min_asp_scale + (1 - r1) * self.max_asp_scale)
            if np.random.rand() > 0.5:
                scale = scale[::-1]
            resized_H, resized_W = int(H * scale[0] * base_scale), int(W * scale[1] * base_scale)

            mode = self.modes[torch.randint(len(self.modes), ())]
            img = torch.nn.functional.interpolate(img[None], size=(resized_H, resized_W), mode=mode)[0]
            boxes = boxes.resize((resized_W, resized_H))

            return img, boxes, *args


def chop_edge_of_graph(graph, max_size):
    connected_components = [list(cc) for cc in nx.connected_components(graph)]
    new_graphs = []
    for cc in connected_components:
        subgraph = nx.Graph(graph.subgraph(cc))
        if len(cc) <= max_size:
            new_graphs.append(subgraph)
            continue
        edges = list(subgraph.edges())
        node_centrality = nx.betweenness_centrality(subgraph)
        edge_centralities = []
        for s, o in edges:
            edge_centralities.append(node_centrality[s] * node_centrality[o] + 1e-5)
        edge_centralities = np.array(edge_centralities)
        i_target_edge = np.random.choice(len(edges), p=edge_centralities / np.sum(edge_centralities))
        target_edge = edges[i_target_edge]
        subgraph.remove_edge(*target_edge)
        new_graphs.extend(chop_edge_of_graph(subgraph, max_size))
    return new_graphs


class RandomDropEdgesToLimitSize:
    """
    Randomly drop edges to limit the number of edges to the specified number.
    Make sure that the graph does not contain loops.
    """
    def __init__(self, max_bboxes=50):
        self.max_bboxes = max_bboxes

    def __call__(self, img, bboxes, relationships, predicates):
        selected_nodes = self._select(len(bboxes), relationships)
        selected_nodes = np.array(list(selected_nodes))
        old_idx_to_new_idx = {old: new for new, old in enumerate(selected_nodes)}

        new_relationships = []
        new_predicates = []
        for (s, o), p in zip(relationships.tolist(), predicates.tolist()):
            if s in selected_nodes and o in selected_nodes:
                new_relationships.append([old_idx_to_new_idx[s], old_idx_to_new_idx[o]])
                new_predicates.append(p)
        new_relationships = torch.tensor(new_relationships, dtype=torch.int64)
        new_predicates = torch.tensor(new_predicates, dtype=torch.int64)

        new_bboxes = BoxList(bboxes.bbox[selected_nodes], bboxes.size, mode='xyxy')
        for fieldname in bboxes.fields():
            value = bboxes.get_field(fieldname)
            if type(value) == list:
                new_bboxes.add_field(fieldname, [value[i] for i in selected_nodes])
            elif hasattr(value, '__getitem__'):
                new_bboxes.add_field(fieldname, value[selected_nodes])
            else:
                new_bboxes.add_field(fieldname, value)

        return img, new_bboxes, new_relationships, new_predicates

    def _select(self, len_bboxes, relationships):
        graph = nx.Graph()
        graph.add_nodes_from(range(len_bboxes))
        graph.add_edges_from(relationships.numpy().tolist())

        chopped_graphs = chop_edge_of_graph(graph, self.max_bboxes)

        n_count = 0
        selected_graphs = []

        while True:
            n_nodes_of_graph = np.array([len(g.nodes) for g in chopped_graphs])
            candidates = np.where(n_nodes_of_graph <= (self.max_bboxes - n_count))[0]
            if len(candidates) == 0:
                break
            i_selected = np.random.choice(candidates)
            selected_graph = chopped_graphs.pop(i_selected)
            selected_graphs.append(selected_graph)
            n_count += len(selected_graph.nodes)
        
        nodes = set()
        for graph in selected_graphs:
            nodes.update(graph.nodes)

        return nodes


def drop_self_loop_edges(img, boxes, rels, preds, *args):
    is_self_loop = rels[:, 0] == rels[:, 1]
    return img, boxes, rels[~is_self_loop], preds[~is_self_loop], *args


def random_drop_duplicate_edges(img, boxes, rels, preds, *args):
    if len(rels) == 0:
        return img, boxes, rels, preds, *args
    
    combination, _ = rels.sort(dim=1)
    _, group = torch.unique(combination, dim=0, return_inverse=True)

    selected_rels = []
    for i in range(group.max() + 1):
        candidate, = torch.where(group == i)
        i_selected = candidate[torch.randint(len(candidate), ())]
        selected_rels.append(i_selected)
    selected_rels = torch.stack(selected_rels)

    return img, boxes, rels[selected_rels], preds[selected_rels], *args


def random_drop_loopy_edges(img, boxes, rels, preds, *args):
    """Randomly drop duplicate edges using Kruskal's algorithm with union-find."""

    priorities = torch.randperm(len(rels))
    rels = rels[priorities]
    preds = preds[priorities]

    directions = {i.item(): i.item() for i in torch.unique(rels)}
    def find_root(i):
        if directions[i] == i:
            return i
        directions[i] = find_root(directions[i])
        return directions[i]

    is_selected = torch.zeros(len(rels), dtype=torch.bool)
    for i in range(len(rels)):
        a = find_root(rels[i, 0].item())
        b = find_root(rels[i, 1].item())
        if a == b:
            continue
        is_selected[i] = True
        directions[a] = min(a, b)
        directions[b] = min(a, b)

    return img, boxes, rels[is_selected], preds[is_selected], *args


class DataAugmentationForSG:
    def __init__(self, config, val=False, evaluate=False):
        val = val or evaluate
        
        self.augmentations = [
            random_drop_duplicate_edges,
            drop_self_loop_edges]

        if not evaluate:
            self.augmentations += [
                random_drop_loopy_edges,
                RandomDropEdgesToLimitSize(
                    config["drop_node"]["max_n"])
            ]
        
        if val:
            self.augmentations += [
                RandomResize(
                    0.0,
                    config["resize"]["min_scale"],
                    config["resize"]["max_scale"],
                    config["resize"]["aspect_shift"],
                    config["resize"]["base_long_side"])
            ]
        else:
            self.augmentations += [
                RandomResize(
                    config["resize"]["prob"],
                    config["resize"]["min_scale"],
                    config["resize"]["max_scale"],
                    config["resize"]["aspect_shift"],
                    config["resize"]["base_long_side"]),
                RandomFlip(
                    config["flip"]["prob"]),
                RandomColorJitter(
                    config["color_jitter"]["prob"],
                    config["color_jitter"]["brightness"],
                    config["color_jitter"]["contrast"],
                    config["color_jitter"]["saturation"],
                    config["color_jitter"]["hue"]),
                AddGaussianNoise(
                    config["gaussian_noise"]["prob"],
                    config["gaussian_noise"]["sigma"])
            ]
    
    def __call__(self, img, boxes, rels, preds, *args):
        for augmentation in self.augmentations:
            img, boxes, rels, preds = augmentation(img, boxes, rels, preds)
        return img, boxes, rels, preds, *args