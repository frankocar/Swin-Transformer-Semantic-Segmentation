from .builder import DATASETS
from .custom import CustomDataset

import os
import numpy as np
from sklearn.preprocessing import minmax_scale


@DATASETS.register_module()
class LandslidesDataset(CustomDataset):
    """Landslides dataset.

    In segmentation map annotation for Landslides, 0 stands for background, which
    is not included in the landslide category. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.npz' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'landslide')

    PALETTE = [[255, 255, 255]]

    def __init__(self, **kwargs):
        super(LandslidesDataset, self).__init__(
            img_suffix='.npz',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)

    def load_image(self, path):
        # image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        # path = os.path.join(self.root_dir, self.img_dir, self.set_name, image_info['file_name'])

        with np.load(path) as data:
            if len(data) != 1 or "arr_0" not in data:
                raise Exception("More than 1 array in the npz or name invalid")
            arr = data['arr_0'].astype(np.float32)

        if arr.shape[0] != arr.shape[1]:
            print("ERROR", path)
               
        # data = minmax_scale(np.clip(arr, 0, 99999), feature_range=(0, 1))
        data = np.clip(arr, 0, 999999)
        # return np.repeat(data[:, :, np.newaxis], 3, axis=2)
        return data