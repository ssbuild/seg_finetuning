```text
    10-24 initial semantic segmentation
```

## update information
   - [deep_training](https://github.com/ssbuild/deep_training)

## install
  - pip install -U -r requirements.txt
  - 如果无法安装， 可以切换官方源 pip install -i https://pypi.org/simple -U -r requirements.txt



## weigtht select one is suitable for you
支持且不限于以下权重 , See all SegFormer models at https://huggingface.co/models?filter=segformer
- [mit-b0](https://huggingface.co/nvidia/mit-b0)
- [segformer-b3-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b3-finetuned-ade-512-512)
- [segformer-b5-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024)




## data sample
- open_data https://github.com/ssbuild/open_data
- sidewalk-semantic https://huggingface.co/datasets/segments/sidewalk-semantic
   



## infer
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理微调模型
     python infer_finetuning.py



## training
```text
    # 制作数据
    cd scripts
    bash train_full.sh -m dataset 

    
    注: num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)
    
    # 全参数训练 
        bash train_full.sh -m train

```
   

## 训练参数
[训练参数](args.MD)

## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [tf-task-example](https://github.com/ssbuild/tf-task-example)
- [chatmoss_finetuning](https://github.com/ssbuild/chatmoss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [chatglm2_finetuning](https://github.com/ssbuild/chatglm2_finetuning)
- [chatglm3_finetuning](https://github.com/ssbuild/chatglm3_finetuning)
- [t5_finetuning](https://github.com/ssbuild/t5_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf](https://github.com/ssbuild/llm_rlhf)
- [chatglm_rlhf](https://github.com/ssbuild/chatglm_rlhf)
- [t5_rlhf](https://github.com/ssbuild/t5_rlhf)
- [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)
- [baichuan_finetuning](https://github.com/ssbuild/baichuan_finetuning)

## 
    纯粹而干净的代码


## 参考



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ssbuild/semantic_segmentation&type=Date)](https://star-history.com/#ssbuild/semantic_segmentation&Date)