# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import pickle
import warnings
import copy
import json
import random
import torch
from PIL import Image
import numpy as np
from numpy_io.pytorch_loader.tokenizer_config_helper import load_configure
from deep_training.utils.hf import BatchFeatureDetr
from typing import Union, Optional, List, Any
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, TrainingArgumentsHF, \
    TrainingArgumentsCL, DataArguments, TrainingArgumentsAC
from aigc_zoo.model_zoo.object_detection.llm_model import PetlArguments, LoraConfig, PromptArguments
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from transformers import PreTrainedTokenizer, HfArgumentParser, PretrainedConfig, Wav2Vec2Processor, requires_backends
from config import *
from module_setup import module_setup

module_setup()


def preprocess(text):
    return text


def postprocess(text):
    return text




class NN_DataHelper(DataHelper):
    index = 1
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __init__(self, *args, **kwargs):
        super(NN_DataHelper, self).__init__(*args, **kwargs)

    def on_get_labels(self, files: List[str]):
        label2id = {
            "N/A": 0,
            "airplane": 5,
            "apple": 53,
            "backpack": 27,
            "banana": 52,
            "baseball bat": 39,
            "baseball glove": 40,
            "bear": 23,
            "bed": 65,
            "bench": 15,
            "bicycle": 2,
            "bird": 16,
            "blender": 83,
            "boat": 9,
            "book": 84,
            "bottle": 44,
            "bowl": 51,
            "broccoli": 56,
            "bus": 6,
            "cake": 61,
            "car": 3,
            "carrot": 57,
            "cat": 17,
            "cell phone": 77,
            "chair": 62,
            "clock": 85,
            "couch": 63,
            "cow": 21,
            "cup": 47,
            "desk": 69,
            "dining table": 67,
            "dog": 18,
            "donut": 60,
            "door": 71,
            "elephant": 22,
            "eye glasses": 30,
            "fire hydrant": 11,
            "fork": 48,
            "frisbee": 34,
            "giraffe": 25,
            "hair drier": 89,
            "handbag": 31,
            "hat": 26,
            "horse": 19,
            "hot dog": 58,
            "keyboard": 76,
            "kite": 38,
            "knife": 49,
            "laptop": 73,
            "microwave": 78,
            "mirror": 66,
            "motorcycle": 4,
            "mouse": 74,
            "orange": 55,
            "oven": 79,
            "parking meter": 14,
            "person": 1,
            "pizza": 59,
            "plate": 45,
            "potted plant": 64,
            "refrigerator": 82,
            "remote": 75,
            "sandwich": 54,
            "scissors": 87,
            "sheep": 20,
            "shoe": 29,
            "sink": 81,
            "skateboard": 41,
            "skis": 35,
            "snowboard": 36,
            "spoon": 50,
            "sports ball": 37,
            "stop sign": 13,
            "street sign": 12,
            "suitcase": 33,
            "surfboard": 42,
            "teddy bear": 88,
            "tennis racket": 43,
            "tie": 32,
            "toaster": 80,
            "toilet": 70,
            "toothbrush": 90,
            "traffic light": 10,
            "train": 7,
            "truck": 8,
            "tv": 72,
            "umbrella": 28,
            "vase": 86,
            "window": 68,
            "wine glass": 46,
            "zebra": 24
        }
        id2label = {i: label for label,i in label2id.items()}
        return label2id, id2label

    def load_config(self,
                    config_name=None,
                    config_class_name=None,
                    model_name_or_path=None,
                    task_specific_params=None,
                    with_labels=True,
                    with_task_params=True,
                    return_dict=False,
                    with_print_labels=True,
                    config_kwargs=None):

        tokenizer = None
        if config_kwargs is None:
            config_kwargs = {}

        model_args: ModelArguments = self.model_args
        training_args = self.training_args
        data_args: DataArguments = self.data_args

        if data_args is not None:
            self.max_seq_length_dict['train'] = data_args.train_max_seq_length
            self.max_seq_length_dict['eval'] = data_args.eval_max_seq_length
            self.max_seq_length_dict['val'] = data_args.eval_max_seq_length
            self.max_seq_length_dict['test'] = data_args.test_max_seq_length
            self.max_seq_length_dict['predict'] = data_args.test_max_seq_length

        if with_task_params:
            task_specific_params = task_specific_params or {}
            task_params = self.on_task_specific_params()
            if task_params is not None:
                task_specific_params.update(task_params)

            if training_args is not None:
                task_specific_params['learning_rate'] = training_args.learning_rate
                task_specific_params[
                    'learning_rate_for_task'] = training_args.learning_rate_for_task or training_args.learning_rate

        kwargs_args = {
            "return_dict": return_dict,
            "task_specific_params": task_specific_params,
        }
        kwargs_args.update(config_kwargs)

        config = load_configure(config_name=config_name or model_args.config_name,
                                class_name=config_class_name,
                                model_name_or_path=model_name_or_path or model_args.model_name_or_path,
                                cache_dir=model_args.cache_dir,
                                model_revision=model_args.model_revision,
                                use_auth_token=model_args.use_auth_token,
                                **kwargs_args
                                )
        self.config = config
        if with_labels and self.label2id is not None and hasattr(config, 'num_labels'):
            if with_print_labels:
                print('==' * 30, 'num_labels = ', config.num_labels)
                print(self.label2id)
                print(self.id2label)

        if with_labels:
            return tokenizer, config, self.label2id, self.id2label
        return tokenizer, config

    def load_tokenizer_and_config(self, *args, **kwargs):
        ret = self.load_config(*args, **kwargs)
        self._preprocess_tokenizer_config()
        self.load_feature_extractor()
        self.load_processer()
        return ret

    def _preprocess_tokenizer_config(self):
        pass

    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: Any, mode: str):
        self.index += 1
        # config = self.config
        # max_seq_length = self.max_seq_length_dict[mode]
        # data_args = self.data_args
        path, labels = data

        d = {
            "path": np.asarray(bytes(path, encoding="utf-8")),
            "labels": np.asarray(bytes(json.dumps(labels,ensure_ascii=False), encoding="utf-8"))
        }

        if not d:
            return None

        if self.index < 3:
            print(d)
        return d

    def _get_paragraph(self, lines):
        D = []
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            D.append((jd["path"], jd["labels"]))
        return D

    # 读取文件
    def on_get_corpus(self, files: List, mode: str):
        D = []
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            D.extend(self._get_paragraph(lines))
        return D

    def collate_fn(self, batch):
        batch = copy.copy(batch)
        images,annotations = [],[]
        for feature in batch:
            path = str(feature["path"][0], encoding="utf-8") if isinstance(feature["path"][0],bytes) else feature["path"][0]
            annotation = str(feature["labels"][0], encoding="utf-8") if isinstance(feature["labels"][0], bytes) else feature["labels"][0]
            images.append(Image.open(path).convert("RGB"))
            annotations.append(json.loads(annotation))
        inputs = self.processor(images=images, annotations=annotations, return_tensors="pt")

        return BatchFeatureDetr(inputs.data)

    def make_dataset_all(self):
        data_args = self.data_args
        # schema for arrow parquet
        schema = {
            "path": "binary_list",
            "labels": "binary_list",
        }

        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train',
                                        schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval', schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test', schema=schema)


if __name__ == '__main__':

    if global_args["trainer_backend"] == "hf":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsHF, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )
    elif global_args["trainer_backend"] == "pl":
        parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments, PromptArguments))
        model_args, training_args, data_args, _, _ = parser.parse_dict(train_info_args)
    elif global_args["trainer_backend"] == "cl":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsCL, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )
    else:
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsAC, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16})

    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    dataHelper.make_dataset_all()
