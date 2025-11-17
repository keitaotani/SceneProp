import json
import numpy as np
import h5py

import torch
from sge.data_container import BoxList
from sge.data_container.imagelist import to_image_list


class VGSubgraph(torch.utils.data.Dataset):
    def __init__(
        self,
        path_imdb = 'data_tools/imdb_1024.h5',
        path_sgg = 'data_tools/VG-SGG.h5',
        path_category = 'data_tools/VG-SGG-dicts.json',
        split = 'train',
        transform = None):

        super().__init__()

        assert split in ['train', 'test']

        self.fh_imdb = h5py.File(path_imdb, 'r')
        self.fh_sgg = h5py.File(path_sgg, 'r')
        self.transform = transform
        
        splitid = {'train': 0, 'test': 2}
        in_split = self.fh_sgg['split'][:] == splitid[split]
        with_boxes = self.fh_sgg['img_to_last_box'][:] - self.fh_sgg['img_to_first_box'][:] > 0
        with_rels = self.fh_sgg['img_to_last_rel'][:] - self.fh_sgg['img_to_first_rel'][:] > 0
        self.available_idx = np.where(in_split & with_boxes & with_rels)[0]

        category_dict = json.load(open(path_category, 'r'))
        self.obj_categories = [category_dict['idx_to_label'][str(i+1)] for i in range(150)]
        self.rel_categories = [category_dict['idx_to_predicate'][str(i+1)] for i in range(50)]
    
    def __len__(self):
        return len(self.available_idx)
    
    def __getitem__(self, idx):
        idx = self.available_idx[idx]

        # load data ----------------------
        img_w = self.fh_imdb['image_widths'][idx]
        img_h = self.fh_imdb['image_heights'][idx]
        img = self.fh_imdb['images'][idx]
        img = img[::-1, :img_h, :img_w].copy()
        img = torch.from_numpy(img) / 255.

        first_box = self.fh_sgg['img_to_first_box'][idx]
        last_box  = self.fh_sgg['img_to_last_box'][idx]
        first_rel = self.fh_sgg['img_to_first_rel'][idx]
        last_rel  = self.fh_sgg['img_to_last_rel'][idx]

        boxes = self.fh_sgg['boxes_1024'][first_box:last_box+1]
        labels = self.fh_sgg['labels'][first_box:last_box+1][:, 0]
        relationships = self.fh_sgg['relationships'][first_rel:last_rel+1] - first_box
        predicates = self.fh_sgg['predicates'][first_rel:last_rel+1][:, 0]

        # boxlist ----------------------
        cx, cy, w, h = boxes.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = BoxList(torch.from_numpy(np.stack([x1, y1, x2, y2], axis=1)), (img_w, img_h), mode='xyxy')
        boxes.add_field('labels', torch.from_numpy(labels) - 1)

        ret = (
            img,
            boxes,
            torch.from_numpy(relationships).to(torch.int64),
            torch.from_numpy(predicates) - 1,
            []  # scene labels
        )

        if self.transform is not None:
            ret = self.transform(*ret)

        return ret
    
    def get_category_names(self):
        # The last element represents scene categories.
        return self.obj_categories, self.rel_categories, []
    
    def get_object_category_names(self):
        return sorted(self.obj_categories)
    
    def get_attribute_category_names(self):
        return []


class VG_VLMPAG(torch.utils.data.Dataset):
    def __init__(
        self,
        path_imdb = 'data_tools/imdb_1024.h5',
        path_sg = 'data_tools/localization_VGFO_vg150vr40_train.json',
        path_category = 'data_tools/category_names.json',
        split = 'train',
        partial_cat = False,
        only_max_size = False,
        transform = None):

        super().__init__()

        assert split in ['train', 'val', 'all']  # when testing, use 'all' and set path_sg to test data

        self.fh_imdb = h5py.File(path_imdb, 'r')
        sg_data = json.load(open(path_sg, 'r'))
        category_names = json.load(open(path_category, 'r'))

        self.full_object_names = category_names['object']
        self.partial_object_names = category_names['object_in_vgpo_train']
        if partial_cat:  # for VG-PO
            self.object_names = self.partial_object_names
        else:  # for VG-FO
            self.object_names = self.full_object_names
        self.predicate_names = category_names['predicate']
        
        if split == 'all':
            self.image_data = sg_data['image_data']
        else:
            splitid = {'train': 0, 'val': 2}
            self.image_data = [d for d in sg_data['image_data'] if d['split'] == splitid[split]]
        self.objects = sg_data['objects']

        self.only_max_size = only_max_size
        self.transform = transform
    
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image_data = self.image_data[idx]
        imdb_id = image_data['imdb_id']

        # load data ----------------------
        img_w = self.fh_imdb['image_widths'][imdb_id]
        img_h = self.fh_imdb['image_heights'][imdb_id]
        img = self.fh_imdb['images'][imdb_id]
        img = img[::-1, :img_h, :img_w].copy()
        img = torch.from_numpy(img) / 255.

        global_objids = []
        objcatids = []
        relationships = []
        predicates = []
        n_edges = []

        connected_scene_graphs = image_data['connected_scene_graphs']
        if self.only_max_size:
            connected_scene_graphs = [max(connected_scene_graphs, key=lambda x: len(x))]

        for connected_graph in connected_scene_graphs:
            for relationship in connected_graph:
                s_catid = self.object_names.index(relationship['subject_name'])
                o_catid = self.object_names.index(relationship['object_name'])
                v_catid = self.predicate_names.index(relationship['predicate'])
                for objpair in relationship['objects']:
                    s_objid = objpair['subject_id']
                    o_objid = objpair['object_id']
                    if s_objid in global_objids:
                        s_lobjid = global_objids.index(s_objid)
                    else:
                        s_lobjid = len(global_objids)
                        global_objids.append(s_objid)
                        objcatids.append(s_catid)
                        n_edges.append(len(connected_graph))
                    if o_objid in global_objids:
                        o_lobjid = global_objids.index(o_objid)
                    else:
                        o_lobjid = len(global_objids)
                        global_objids.append(o_objid)
                        objcatids.append(o_catid)
                        n_edges.append(len(connected_graph))
                    relationships.append((s_lobjid, o_lobjid))
                    predicates.append(v_catid)

        # boxlist ----------------------
        boxes_xyxy = []
        for objid in global_objids:
            box = self.objects[str(objid)]['bbox_1024']
            x1, y1, x2, y2 = box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h']
            boxes_xyxy.append([x1, y1, x2, y2])
        boxes_xyxy = np.array(boxes_xyxy).reshape(-1, 4)
        boxes = BoxList(torch.from_numpy(boxes_xyxy), (img_w, img_h), mode='xyxy')
        boxes.add_field('labels', torch.from_numpy(np.array(objcatids)).to(torch.int64))
        boxes.add_field('n_edges', torch.from_numpy(np.array(n_edges)).to(torch.int64))

        ret = (
            img,
            boxes,
            torch.tensor(relationships).to(torch.int64).reshape(-1, 2),
            torch.tensor(predicates).to(torch.int64),
            []  # scene labels
        )

        if self.transform is not None:
            ret = self.transform(*ret)

        return ret
    
    def get_category_names(self):
        # The last element represents scene categories.
        return self.object_names.copy(), self.predicate_names.copy(), []
    
    def get_object_category_names(self):
        return self.object_names.copy()
    
    def get_attribute_category_names(self):
        return []


def collate_fn(batch):
    imgs, boxes, relationships, predicates, scene_labels = zip(*batch)
    imgs = to_image_list(imgs)
    return imgs, boxes, relationships, predicates, scene_labels