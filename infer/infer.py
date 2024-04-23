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
from data_utils import config_args, NN_DataHelper, get_deepspeed_config
from deep_training.zoo.model_zoo.semantic_segmentation.llm_model import MyTransformer


deep_config = get_deepspeed_config()


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments,))
    (model_args,)  = parser.parse_dict(config_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config()
    processor = dataHelper.processor


    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=config.torch_dtype)
    model = pl_model.get_llm_model()
    model = model.eval()
    model.half().cuda()


    image = Image.open("../assets/000000039769.jpg")


    inputs = dataHelper.feature_extractor(images=image, return_tensors="pt")
    inputs = inputs.to(device=model.device, dtype=torch.half)
    outputs = model(**inputs,return_dict=True)
    logits = outputs.logits.detach()

    print(logits.size())
    # shape (batch_size, num_labels, height/4, width/4)
