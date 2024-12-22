import numpy as np

from .builder import DATASETS
from .custom import CustomDataset
from .api_wrappers import COCO, COCOeval


@DATASETS.register_module()
class MYDataset(CustomDataset):
    CLASSES = ('bowl', 'mouse', 'apple','up','down','right','left','downright','dowmleft')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        data_infos = []
        self.img_ids = self.coco.get_img_ids()


        for i in self.img_ids:
            ann_info = self.coco.load_anns(self.coco.get_ann_ids(img_ids=[i]))
            data_infos.append(
                dict(
                    filename=self.coco.load_imgs([i])[0]['file_name'],
                    width=self.coco.load_imgs([i])[0]['width'],
                    height=self.coco.load_imgs([i])[0]['height'],
                    ann = dict(
                        bboxes=np.array(ann_info[0]['bbox']).astype(np.float32),
                        labels=np.array(ann_info[0]['category_id']).astype(np.int64)
                    ),

                ))
        return data_infos
    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']