import json
from PIL import Image
import numpy as np
import torch

from ..data_container import BoxList
from ..data_container.imagelist import to_image_list
from .gqa_categories import (
    object_categories, attribute_categories, scene_categories,
    alias_to_category, plural_to_singular, relationship_categories, query_candidates)

object_categories = sorted(object_categories)
attribute_categories = sorted(attribute_categories)
objattr_categories = sorted(set(object_categories + attribute_categories))


def cleanse_word(word):
    word = word.strip()
    if word in alias_to_category:
        word = alias_to_category[word]
    if word in plural_to_singular:
        word = plural_to_singular[word]
    return word


def transform_plural_to_singular(name):
    if name in plural_to_singular:
        return plural_to_singular[name]
    else:
        return name


class GQASceneGraphDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, scene_graph_path, transform=None):
        self.image_dir = image_dir

        print("Loading scene graph...")
        scene_graph = json.load(open(scene_graph_path))
        self.image_ids = []
        self.scene_graph = []
        for image_id, graph in scene_graph.items():
            if len(graph['objects']) == 0:
                continue
            self.image_ids.append(image_id)
            self.scene_graph.append(graph)
        print("Done.")

        self.transform = transform

    def __len__(self):
        return len(self.scene_graph)
    
    def get_annotation(self, idx):
        bboxes = []
        labels = []
        plurals = []
        attributes = []
        relationships = []
        predicates = []

        scene_graph = self.scene_graph[idx]

        objectids, objects = zip(*scene_graph['objects'].items())
        objectids = list(objectids)
        objects = list(objects)

        for i, obj in enumerate(objects):
            bboxes.append([obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']])
            name = transform_plural_to_singular(cleanse_word(obj['name']))
            labels.append(objattr_categories.index(name))
            plurals.append(name != obj['name'])
            attributes.append([
                objattr_categories.index(transform_plural_to_singular(attr))
                for attr in obj['attributes']
                ])
            for rel in obj['relations']:
                relationships.append([i, objectids.index(rel['object'])])
                predicates.append(relationship_categories.index(cleanse_word(rel['name'])))
        
        bboxes = torch.tensor(bboxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        plurals = torch.tensor(plurals, dtype=torch.bool)
        relationships = torch.tensor(relationships, dtype=torch.long).reshape(-1, 2)
        predicates = torch.tensor(predicates, dtype=torch.long)

        scene_label_names = []
        if 'location' in scene_graph:
            scene_label_names.append(scene_graph['location'])
        if 'weather' in scene_graph:
            scene_label_names.append(scene_graph['weather'])
        scene_label_names = [transform_plural_to_singular(cleanse_word(name)) for name in scene_label_names]
        scene_labels = [scene_categories.index(cleanse_word(name)) for name in scene_label_names]

        bboxes = BoxList(bboxes, (scene_graph['width'], scene_graph['height']), mode='xyxy')
        bboxes.add_field('labels', labels)
        bboxes.add_field('plurals', plurals)
        bboxes.add_field('attributes', attributes)

        return bboxes, relationships, predicates, scene_labels
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        imagepath = f'{self.image_dir}/{image_id}.jpg'
        image = np.array(Image.open(imagepath).convert('RGB')).transpose(2, 0, 1)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        bboxes, relationships, predicates, scene_labels = self.get_annotation(idx)

        ret = (
            image,
            bboxes,
            relationships,
            predicates,
            scene_labels
        )

        if self.transform is not None:
            ret = self.transform(*ret)

        return ret

    def get_category_names(self):
        return objattr_categories, relationship_categories, scene_categories
    
    def get_object_category_names(self):
        return object_categories
    
    def get_attribute_category_names(self):
        return attribute_categories


def collate_fn_for_qa(batch):
    imgs, variables, factors, boxes, questiontexts, answertexts = zip(*batch)
    imgs = to_image_list(imgs)
    return imgs, variables, factors, boxes, questiontexts, answertexts