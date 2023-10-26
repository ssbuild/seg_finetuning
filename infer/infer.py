# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import os
import sys



sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
from PIL import Image
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config
from aigc_zoo.model_zoo.object_detection.llm_model import MyTransformer


deep_config = get_deepspeed_config()


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments,))
    (model_args,)  = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config()
    processor = dataHelper.processor


    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=config.torch_dtype)
    model = pl_model.get_llm_model()
    model = model.eval()
    model.half().cuda()


    image = Image.open("../assets/000000039769.jpg")
    inputs = processor(images=image, return_tensors="pt")

    inputs = inputs.to(device=model.device,dtype=torch.half)

    outputs = model(**inputs,return_dict=True)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

    # Detected remote with confidence 0.998 at location [40.12, 70.75, 175.62, 118.0]
    # Detected remote with confidence 0.996 at location [333.0, 72.62, 368.0, 187.75]
    # Detected couch with confidence 0.996 at location [0.0, 1.17, 639.5, 473.75]
    # Detected cat with confidence 0.999 at location [13.2, 52.16, 314.0, 471.0]
    # Detected cat with confidence 0.999 at location [345.25, 23.91, 640.0, 368.75]

