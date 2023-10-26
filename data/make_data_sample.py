# coding=utf8
# @Time    : 2023/10/26 0:07
# @Author  : tk
# @FileName: make_data
import json
import os.path
from shutil import copyfile
import datasets
from torchvision.datasets.coco import CocoDetection
from pathlib import Path

class MyCocoDetection(CocoDetection):
    def _load_image(self, id: int) :
        path = self.coco.loadImgs(id)[0]["file_name"]
        return os.path.join(self.root, path)

    def __getitem__(self, idx):
        img, target = super(MyCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        return img, target


def build(coco_path,image_set,outfile,limit_n=-1):
    root = coco_path
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = MyCocoDetection(img_folder, ann_file)
    if limit_n < 0:
        limit_n = len(dataset)
    with open(outfile,mode='w',encoding='utf-8') as f:
        for i in range(min(len(dataset),limit_n)):
            path,labels = dataset[i]
            f.write(json.dumps({
                "path": path,
                "labels": labels
            },ensure_ascii=False) + '\n')





data_dir = Path("/data/cv/data/coco")

build(data_dir,"train","train.json",100)

build(data_dir,"val","dev.json",20)
