# LNGTransformer
针对基于Self-Attention的传统图像分类网络无法兼顾全局信息和计算复杂度的问题，提出一种高效的、可扩展的注意力模块LNG-SA，该模块在任意时期都能进行局部信息，邻居信息和全局信息的交互。通过简单的重复堆叠LNG-SA模块，设计了一个全新的网络，称为LNG-Transformer。



## 1、数据集路径关系：
```
--imagenet-100
  ｜--train
      ｜--n0144764
      ｜--n01443537
      ｜--.....
  ｜val
      ｜--n0144764
      ｜--n01443537
      ｜--.....
```

## 2、训练
### 单卡运行

```bash
python mian.py --model LNG_T --batch-size 256 --epochs 300 --data-path ../imagenet-100
```

### 多卡并行运行

```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env mian.py --model LNG_T --epochs 300 --batch-size 256 --data-path ../imagenet-100
```


```
-nproc_per_node：多卡运行时，显卡数量。
-model：'LNG_T', 'LNG_S', 'LNG_B','Swin_T', 'Swin_S', 'Swin_B','ViT_B', 'ViT_L','resnet_50', 'resnet_101'。
-batch-size：每一批数量。
-data-path：数据集路径。
```

## 3、测试
```bash
python  mian.py --model LNG_T --eval --batch-size 256 --data-path ../imagenet-100
```

