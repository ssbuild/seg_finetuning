# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
import glob
import io
import sys
import os
from functools import cache

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
from aigc_zoo.model_zoo.semantic_segmentation.llm_model import PetlArguments, LoraConfig, PromptArguments
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
        with open(files[0],mode='r',encoding='utf-8') as f:
            id2label = json.loads(f.read())
        label2id= {label: i for i,label in id2label.items()}
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

        if with_labels and self.label2id is not None:
            kwargs_args['label2id'] = self.label2id
            kwargs_args['id2label'] = self.id2label
            kwargs_args['num_labels'] = len(self.label2id) if self.label2id is not None else None
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
        files = sum([glob.glob(file) for file in files], [])
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            D.extend(self._get_paragraph(lines))
        return D

    # def collate_fn(self, batch):
    #     batch = copy.copy(batch)
    #     images,annotations = [],[]
    #     for feature in batch:
    #         path = str(feature["path"], encoding="utf-8") if isinstance(feature["path"],bytes) else feature["path"]
    #         annotation = str(feature["labels"], encoding="utf-8") if isinstance(feature["labels"], bytes) else feature["labels"]
    #         images.append(Image.open(path).convert("RGB"))
    #         annotations.append(json.loads(annotation))
    #     inputs = self.processor(images=images, annotations=annotations, return_tensors="pt")
    #     return BatchFeatureDetr(inputs.data)


    def collate_fn(self, batch):
        batch = copy.copy(batch)
        images,annotations = [],[]
        for feature in batch:
            pixel_values = feature[ "pixel_values.bytes" ]
            label = feature[ "label.bytes" ]
            images.append(Image.open(io.BytesIO(pixel_values)))
            annotations.append(Image.open(io.BytesIO(label)))
        inputs = self.processor(images=images, segmentation_maps=annotations, return_tensors="pt")
        return inputs

    def make_dataset_all(self):
        data_args = self.data_args
        # schema for arrow parquet
        schema = {
            "path": "bytes",
            "labels": "bytes",
        }

        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train',
                                        schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval', schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test', schema=schema)

        # 记录缓存文件
        with open(os.path.join(data_args.output_dir, 'intermediate_file_index.json'), mode='w',
                  encoding='utf-8') as f:
            f.write(json.dumps({
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "test_files": self.test_files,
            }, ensure_ascii=False))

    @cache
    def load_dataset_files(self):
        data_args = self.data_args
        if not data_args.convert_file:
            return {
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "test_files": self.test_files,
            }

        filename = os.path.join(data_args.output_dir, 'intermediate_file_index.json')
        assert os.path.exists(filename), 'make you dataset firstly'
        with open(filename, mode='r', encoding='utf-8') as f:
            return json.loads(f.read())


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
    print(f'to make dataset is overwrite_cache {data_args.overwrite_cache}')
    dataHelper.make_dataset_all()

    print('make dataset complete!')
    print('check data !')
    dataset = dataHelper.load_sequential_sampler(dataHelper.load_dataset_files()["train_files"],
                                                 with_load_memory=data_args.data_backend == 'record',
                                                 batch_size=1,
                                                 collate_fn=dataHelper.collate_fn)

    print('total', len(dataset))
    for i, d in enumerate(dataset):
        print(d)
        if i > 3:
            break
