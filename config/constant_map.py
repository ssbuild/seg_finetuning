# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps
from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "MODELS_MAP"
]


# https://huggingface.co/nvidia/segformer-b3-finetuned-ade-512-512
# https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024

# See all SegFormer models at https://huggingface.co/models?filter=segformer

MODELS_MAP = {
    'mit-b0':{
        'model_type': 'segformer',
        'model_name_or_path': '/data/nlp/pre_models/torch/segformer/mit-b0',
        'config_name': '/data/nlp/pre_models/torch/segformer/mit-b0/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/segformer/mit-b0',
    },
    'segformer-b0-finetuned-ade-512-512': {
        'model_type': 'segformer',
        'model_name_or_path': '/data/nlp/pre_models/torch/segformer/segformer-b0-finetuned-ade-512-512',
        'config_name': '/data/nlp/pre_models/torch/segformer/segformer-b0-finetuned-ade-512-512/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/segformer/segformer-b0-finetuned-ade-512-512',
    },

    'segformer-b5-finetuned-cityscapes-1024-1024': {
        'model_type': 'segformer',
        'model_name_or_path': '/data/nlp/pre_models/torch/segformer/segformer-b5-finetuned-cityscapes-1024-1024',
        'config_name': '/data/nlp/pre_models/torch/segformer/segformer-b5-finetuned-cityscapes-1024-1024/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/segformer/segformer-b5-finetuned-cityscapes-1024-1024',
    },





}


# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING




