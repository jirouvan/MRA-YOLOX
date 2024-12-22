
import asyncio
import os
import numpy as np
import os
from tqdm import tqdm
from argparse import ArgumentParser
import json
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            # 判断是否传入catId，如果传入就计算指定类别的指标
            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            # 判断是否传入catId，如果传入就计算指定类别的指标
            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info

def show_result(model,image_path,result_cls,result_state):
    thr = 0.1
    show_result_pyplot(
        'cls',
        model,
        image_path,
        result_cls,
        palette='coco',
        score_thr=thr)
    image_path = './resultout1.jpg'
    show_result_pyplot(
        "state",
        model,
        image_path,
        result_state,
        palette='coco',
        score_thr=thr)
    cls_num = 0
    for i in range(len(result_cls)):
        if (result_cls[i] != 0).any():
            if result_cls[i][0][4]>thr:
                cls_num+=1
    print("类别个数")
    print(cls_num)
def iou(box1, box2):
    '''
    两个框（二维）的 iou 计算

    注意：边框以左上为原点

    box:[top, left, bottom, right]
    '''
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return iou

def predict_json_cls(image_root,val_json,model,thr=0.3):
    images_root = image_root
    predicts_cls=[]
    val_json = json.load(open(val_json,'r'))
    for image in tqdm(val_json['images']):
        image_path = image['file_name']
        result_cls, result_state,result_merge = inference_detector(model, os.path.join(images_root,image_path))

        for pre_cls,pre_result in enumerate(result_cls):
            for content in pre_result:
                if content[-1] > thr:
                    bbox = list(content[:-1].astype(np.float64))
                    predict = {
                        'image_id': image['id'],
                        'category_id': pre_cls,
                        'bbox': [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]],
                        'score': np.float64(content[-1])
                    }
                    predicts_cls.append(predict)
        json_str_cls = json.dumps(predicts_cls,indent=4)

        with open('./predict_result_cls.json','w') as json_file:
            json_file.write(json_str_cls)

def predict_json_state(image_root,val_json,model,thr=0.3):
    images_root = image_root
    predicts_state = []
    val_json = json.load(open(val_json,'r'))
    for image in tqdm(val_json['images']):
        image_path = image['file_name']
        result_cls, result_state,result_merge = inference_detector(model, os.path.join(images_root,image_path))

        for pre_cls,pre_result in enumerate(result_state):
            for content in pre_result:
                if content[-1] > thr:
                    bbox = list(content[:-1].astype(np.float64))
                    predict = {
                        'image_id': image['id'],
                        'category_id': pre_cls,
                        'bbox': [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]],
                        'score': np.float64(content[-1])
                    }
                    predicts_state.append(predict)

        json_str_state = json.dumps(predicts_state, indent=4)

        with open('./predict_result_state.json','w') as json_file:
            json_file.write(json_str_state)

def predict_json_merge(image_root,val_json,model,thr=0.3):
    images_root = image_root
    predicts_merge = []
    val_json = json.load(open(val_json,'r'))
    for image in tqdm(val_json['images']):
        image_path = image['file_name']
        result_cls, result_state,result_merge = inference_detector(model, os.path.join(images_root,image_path))

        for pre_cls,pre_result in enumerate(result_merge):
            for content in pre_result:
                if content[-1] > thr:
                    bbox = list(content[:-1].astype(np.float64))
                    predict = {
                        'image_id': image['id'],
                        'category_id': pre_cls,
                        'bbox': [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]],
                        'score': np.float64(content[-1])
                    }
                    predicts_merge.append(predict)

        json_str_merge = json.dumps(predicts_merge,indent=4)

        with open('./predict_result_merge.json','w') as json_file:
            json_file.write(json_str_merge)

def coco_map(val_json,pre_json):
    coco_true = COCO(annotation_file=val_json)
    coco_pre = coco_true.loadRes(pre_json)
    coco_evaluator = COCOeval(cocoGt=coco_true,cocoDt=coco_pre,iouType='bbox')
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

def coco_ap_per_class(val_json,pre_json):
    dts = json.load(open(pre_json,'r'))
    imgIds = [imid['image_id'] for imid in dts]

    imgIds = sorted(list(set(imgIds)))
    del dts
    coco_true = COCO(annotation_file=val_json)
    coco_pre = coco_true.loadRes(pre_json)
    coco_evaluator = COCOeval(cocoGt=coco_true,cocoDt=coco_pre,iouType='bbox')
    coco_evaluator.params.imgIds = imgIds
    # coco_evaluator.params.catIds = [9]
    coco_evaluator.params.catIds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

def coco_ap_for_classes(val_json,pre_json):
    coco_true = COCO(annotation_file=val_json)
    coco_pre = coco_true.loadRes(pre_json)
    coco_evaluator = COCOeval(cocoGt=coco_true,cocoDt=coco_pre,iouType='bbox')
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()

    voc_map_info_list = []
    class_dict = json.load(open(val_json,'r'))['categories']
    category_index = {cls_dict['id']:cls_dict['name']for cls_dict in class_dict}
    for i in range(len(category_index)):
        stats, _ = summarize(coco_evaluator, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index[i], stats[0]))

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)

mode = 'show_result'# predict_json coco_map show_result
# num_state = 10
thr = 0.3

cfg = './jiyan/my_yolox_s.py'
checkpoint = './jiyan/epoch_80.pth'
# cfg = './checkpoints/baseline+head+eca+sa+mae+gan/my_yolox_s.py'
# checkpoint = './checkpoints/baseline+head+eca+sa+mae+gan/latest.pth'
image_root = './data/coco/images'
# image_path = './data/coco/images/1779cu.jpg'
image_path = 'F:\\ACCV\\demo\\1.png'
model = init_detector(cfg, checkpoint, device='cuda:0')

cal_sub = False
if cal_sub:
    categories= ['bowl', 'apple', 'mouse', 'keyboard', 'banana', 'carrot', 'cup', 'orange', 'chair','book']
else:
    categories = ['total']

for categorie in categories :
    print(f'-------------------{categorie}-------------------------')
    if mode in ['predict_json','coco_map']:
        # mode = 'predict_json'# predict_json coco_map show_result
        if cal_sub:
            val_cls = f"./data/annotation/cls/{categorie}.json"
            val_state = f"./data/annotation/state/{categorie}.json"
            val_merge = f"./data/annotation/merge/{categorie}.json"
            predict_json_cls(image_root, val_cls, model, thr)
            predict_json_state(image_root, val_state, model, thr)
            predict_json_merge(image_root, val_merge, model, thr)
            print(f'-------------------{categorie} ap per for classes------------------------------')
            coco_ap_per_class(val_cls, './predict_result_cls.json')
            coco_ap_per_class(val_state, './predict_result_state.json')
            coco_ap_per_class(val_merge, './predict_result_merge.json')
        else:
            # val_cls = "./data/annotation/class1_val.json"  # 整个测试集关于物体种类的json文件
            val_cls = './data/coco/annotations/val_cls.json'

            # val_state = './data/annotation/class2_val.json'  # 整个测试集关于遮挡方向的json文件
            val_state = './data/coco/annotations/val_state.json'

            val_merge = './data/coco/annotations/instances_val2017.json'# 整个测试集关于小类（100个）的json文件


if mode == 'predict_json':
    print('-------------------prepare cls json------------------------')
    predict_json_cls(image_root,val_cls,model, thr)
    print('-------------------prepare state json------------------------')
    predict_json_state(image_root, val_state, model, thr)
    print('-------------------prepare merge json------------------------')
    predict_json_merge(image_root, val_merge, model, thr)
    print('-------------------end------------------------')


elif mode == 'coco_map':
#mAP
     # print('-----------------------cls----------------------------------')
     # coco_map(val_cls,'./predict_result_cls.json')
     #
     # print('-----------------------state----------------------------------')
     # coco_map(val_state,'./predict_result_state.json')
     #
     # print('-----------------------merge----------------------------------')
     # coco_map(val_merge,'./predict_result_merge.json')
#各类的AP
     # if not cal_sub:
     #     print('-------------------ap for classes------------------------------')
     #     coco_ap_for_classes(val_cls, './predict_result_cls.json')
     #     coco_ap_for_classes(val_state, './predict_result_state.json')
     #     coco_ap_for_classes(val_merge, './predict_result_merge.json')
#新的AP
     print('-------------------ap per for classes------------------------------')
     coco_ap_per_class(val_cls, './predict_result_state.json')
     coco_ap_per_class(val_state, './predict_result_cls.json')
     coco_ap_per_class(val_merge, './predict_result_merge.json')


elif mode == 'show_result':
    result_cls,result_state,result_merge = inference_detector(model,image_path)
    show_result(model,image_path,result_cls,result_state)

