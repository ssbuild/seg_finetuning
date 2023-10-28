# -*- coding: utf-8 -*-
# @Time:  21:53
# @Author: tk
# @File：debug_data
import io

from PIL import Image
from fastdatasets.parquet.dataset import load_dataset

ds = load_dataset.RandomDataset("d:\\tmp\\train-00000-of-00001.parquet")

for i in range(len(ds)):
    d = ds[i]
    print(d)
    print(d.keys())


    pixel_values = d[ "pixel_values.bytes" ]



    print(Image.open(io.BytesIO(pixel_values)))
    #dict_keys(['pixel_values.bytes', 'pixel_values.path', 'label.bytes', 'label.path'])
    break