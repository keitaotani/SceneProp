from typing import Tuple, List, Optional
import numpy as np
import torch
from ..data_container import BoxList, ImageList


class AnchorGenerator:
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        sizes=(64, 128, 256, 512, 1024),
        aspect_ratios=(1.0,),
        anchor_strides=(8, 16, 32, 64, 128)
    ):
        if len(anchor_strides) != len(sizes):
            raise RuntimeError("FPN should have #anchor_strides == #sizes")
        self.strides = anchor_strides

        self.cell_anchors = []
        for size, anchor_stride in zip(sizes, anchor_strides):
            if not isinstance(size, (list, tuple)):
                size = (size,)
            self.cell_anchors.append(
                torch.as_tensor(
                    generate_anchors(
                        size, aspect_ratios, anchor_stride
                    )
                )
            )

    def grid_anchors(self, grid_sizes, device="cpu"):
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.cell_anchors
        ):
            base_anchors = base_anchors.to(device)

            grid_height, grid_width = size
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def add_visibility_to(self, boxlist, straddle_thresh = 0.):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        inds_inside = (
            (anchors[..., 0] >= -straddle_thresh)
            & (anchors[..., 1] >= -straddle_thresh)
            & (anchors[..., 2] < image_width + straddle_thresh)
            & (anchors[..., 3] < image_height + straddle_thresh)
        )
        boxlist.add_field("visibility", inds_inside)

    def __call__(self, image_list: ImageList, feature_maps: Tuple[torch.Tensor]) -> List[List[BoxList]]:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, device=feature_maps[0].device)
        anchors = []
        for image_height, image_width in image_list.image_sizes:
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(
                    anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors


def generate_anchors(
    sizes: Tuple[float, ...] = (32., 64., 128., 256., 512.),
    aspect_ratios: Tuple[float, ...] = (0.5, 1., 2.),
    stride: Optional[int] = None
):
    """ Generate anchors like the original Faster R-CNN implementation.

    Parameters
    ----------
    sizes : Tuple[float, ...]
        Anchor sizes.
    aspect_ratios : Tuple[float, ...]
        Anchor aspect ratios.
    stride : Optional[int]
        Shift the centers of anchors by half of the stride.
        It is useful when the anchors are used for convolutional feature maps.
        If None, the anchors are not shifted.

    Returns
    -------
    anchors : np.ndarray
        Tensor of shape (N, 4) representing N anchors.
        N = len(aspect_ratios) * len(sizes)
    
    Examples
    --------
    >>> generate_anchors(sizes=(32, 64), aspect_ratios=(0.25, 1., 4.))
    [[-31.    -7.75  31.     7.75]
    [-63.   -15.75  63.    15.75]
    [-15.5  -15.5   15.5   15.5 ]
    [-31.5  -31.5   31.5   31.5 ]
    [ -7.75 -31.     7.75  31.  ]
    [-15.75 -63.    15.75  63.  ]]
    >>> generate_anchors(sizes=(32, 64), aspect_ratios=(0.25, 1., 4.), stride=16)
    [[-23.5   -0.25  38.5   15.25]
    [-55.5   -8.25  70.5   23.25]
    [ -8.    -8.    23.    23.  ]
    [-24.   -24.    39.    39.  ]
    [ -0.25 -23.5   15.25  38.5 ]
    [ -8.25 -55.5   23.25  70.5 ]]
    """
    sizes = np.array(sizes, dtype=np.float32)
    aspect_ratios = np.array(aspect_ratios, np.float32)

    base_h = np.sqrt(aspect_ratios)
    base_w = np.sqrt(1.0 / aspect_ratios)
    base_anchors = np.stack([-base_w, -base_h, base_w, base_h], axis=1) / 2.0
    anchors = sizes[None, :, None] * base_anchors[:, None, :]
    anchors = anchors.reshape(-1, 4)

    # The anchors are shrinked by 0.5 as same as the original Faster R-CNN implementation.
    anchors[:, :2] += 0.5
    anchors[:, 2:] -= 0.5

    if stride is not None:
        xy_shift = (stride - 1) / 2.0
        anchors += xy_shift

    return anchors