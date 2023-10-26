# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import os
import sys



sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
from PIL import Image
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser, AutoConfig
from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config
from aigc_zoo.model_zoo.object_detection.llm_model import MyTransformer


deep_config = get_deepspeed_config()


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _,_= dataHelper.load_tokenizer_and_config()
    config = AutoConfig.from_pretrained('./best_ckpt')

    processor = dataHelper.processor
    config.forced_decoder_ids = None


    
    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=config.torch_dtype,)

    # deepspeed 权重使用转换脚本命令
    # 一般根据时间排序选最新的权重文件夹
    # cd best_ckpt/last
    # python zero_to_fp32.py . ../last.ckpt

    train_weight = './best_ckpt/last.ckpt'
    pl_model.load_sft_weight(train_weight,strict=True)

    # 保存hf权重
    # config.save_pretrained('convert/')

    # 保存sft p-tuning-v2 权重
    #  pl_model.save_sft_weight('convert/pytorch_model_sft_ptv2.bin')

    # 保存sft权重
    # pl_model.save_sft_weight('convert/pytorch_model_sft.bin')

    model = pl_model.get_llm_model()

    model.eval().half().cuda()

    image = Image.open("../assets/000000039769.jpg")
    inputs = processor(images=image, return_tensors="pt")

    inputs = inputs.to(device=model.device, dtype=torch.half)

    outputs = model(**inputs, return_dict=True)

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