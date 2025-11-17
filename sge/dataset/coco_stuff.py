import json
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from sge.data_container import BoxList
from sge.data_container.imagelist import to_image_list


class COCOStuff(torch.utils.data.Dataset):
    def __init__(
        self,
        image_dir,
        annotation_file,
        category_file,
        transform=None):

        super().__init__()

        self.image_dir = Path(image_dir)

        with open(annotation_file, 'r') as f:
            self.data = json.load(f)
        
        with open(category_file, 'r') as f:
            category_dict = json.load(f)
        self.obj_categories = category_dict['object']
        self.rel_categories = category_dict['predicate']

        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        image_path = self.image_dir / data['image_name']
        image = np.array(Image.open(image_path).convert('RGB')).transpose(2, 0, 1)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        H, W = image.size(1), image.size(2)

        boxes_xyxy = torch.tensor(data['boxes_xyxy'], dtype=torch.float32)
        labels = [self.obj_categories.index(obj) for obj in data['object_names']]
        labels = torch.tensor(labels, dtype=torch.int64)
        predicates = [self.rel_categories.index(rel) for rel in data['predicate_names']]
        predicates = torch.tensor(predicates, dtype=torch.int64)
        relationships = torch.tensor([data['subject_idx'], data['object_idx']], dtype=torch.int64).T

        bboxes = BoxList(boxes_xyxy, (W, H), mode='xyxy')
        bboxes.add_field('labels', labels)

        ret = (
            image,
            bboxes,
            relationships,
            predicates,
            []  # scene labels
        )

        if self.transform is not None:
            ret = self.transform(*ret)

        return ret

    def get_category_names(self):
        return self.obj_categories.copy(), self.rel_categories.copy(), []
    
    def get_object_category_names(self):
        return self.obj_categories.copy()
    
    def get_attribute_category_names(self):
        return []