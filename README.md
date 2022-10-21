# LNGTransformer
针对基于Self-Attention的传统图像分类网络无法兼顾全局信息和计算复杂度的问题，提出一种高效的、可扩展的注意力模块LNG-SA，该模块在任意时期都能进行局部信息，邻居信息和全局信息的交互。通过简单的重复堆叠LNG-SA模块，设计了一个全新的网络，称为LNG-Transformer。




# 
```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --model LNG_T --batch-size 256 --data-path ../imagenet-100
```

