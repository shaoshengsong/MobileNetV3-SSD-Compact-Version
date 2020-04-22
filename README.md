# MobileNetV3-SSD-Compact-Version
MobileNetV3 SSD的简洁版本

环境
Ubuntu18.04 
版本 PyTorch 1.4

如果您想从头开始需要使用
mbv3_large.old.pth.tar
是backbone用来参数初始化的模型
有可能会历经坎坷


简便方式就是使用预训练模型
checkpoint_ssd300_500epoch.pth.tar
只执行了500次epoch生成的模型,可以用来做预训练模型
如果您要直接测试模型,可以把checkpoint_ssd300_500epoch.pth.tar
该名为checkpoint_ssd300.pth.tar
模型下载地址
链接：https://pan.baidu.com/s/1BZz9X7w3xaopf1dKLkyMcA 
提取码：gwwv


500次epoch模型测试结果
```
1 Mean Average Precision (mAP): 0.619  
max_overlap=0.45


2 Mean Average Precision (mAP): 0.616
max_overlap=0.5

Mean Average Precision (mAP): 0.619的详细结果
{'aeroplane': 0.6828708052635193,
 'bicycle': 0.7328364849090576,
 'bird': 0.5238939523696899,
 'boat': 0.4811790883541107,
 'bottle': 0.19689977169036865,
 'bus': 0.7199779748916626,
 'car': 0.7362964749336243,
 'cat': 0.7842939496040344,
 'chair': 0.3999747931957245,
 'cow': 0.5907276272773743,
 'diningtable': 0.639726459980011,
 'dog': 0.7244610786437988,
 'horse': 0.7985562682151794,
 'motorbike': 0.7596909999847412,
 'person': 0.6491056084632874,
 'pottedplant': 0.32285812497138977,
 'sheep': 0.5846080183982849,
 'sofa': 0.6960644721984863,
 'train': 0.7622987627983093,
 'tvmonitor': 0.5860077142715454}
 ```
 
 200次迭代输出一次,平均损失在2.7左右,向2突破中
```
Epoch: [499][0/1035][5.611666969162847e-05]	Batch Time 1.888 (1.888)	Data Time 1.671 (1.671)	Loss 2.7969 (2.7969)	
Epoch: [499][200/1035][5.611666969162847e-05]	Batch Time 0.104 (0.160)	Data Time 0.000 (0.009)	Loss 2.2807 (2.7760)	
Epoch: [499][400/1035][5.611666969162847e-05]	Batch Time 0.116 (0.155)	Data Time 0.000 (0.005)	Loss 2.4343 (2.7802)	
Epoch: [499][600/1035][5.611666969162847e-05]	Batch Time 0.163 (0.153)	Data Time 0.000 (0.003)	Loss 2.3915 (2.7740)	
Epoch: [499][800/1035][5.611666969162847e-05]	Batch Time 0.084 (0.150)	Data Time 0.001 (0.003)	Loss 2.0844 (2.7727)	
Epoch: [499][1000/1035][5.611666969162847e-05]	Batch Time 0.159 (0.150)	Data Time 0.000 (0.002)	Loss 2.7193 (2.7807)	
```
 使用步骤
一 下载VOC数据集之后,将VOCtrainval_06-Nov-2007和VOCtest_06-Nov-2007合并在一起
数据集下载 可以看这里
https://blog.csdn.net/flyfish1986/article/details/95367745

二 先打开create_data_lists.py文件
改成自己数据集的路径

三 如果想使用mobilenetv3的预训练模型,打开mode.py
找到    def init_weights(self, pretrained=None):#"./mbv3_large.old.pth.tar"
替换成 def init_weights(self, pretrained="./mbv3_large.old.pth.tar"):
您可以固定backbone层只训练最后几层或者全部训练,代码已写好,只要换成False或True

四 运行训练命令python train.py

五 测试mAP命令 python eval.py



以下是站在巨人肩膀上的那个人的肩膀上
hard negative mining和L2Norm没有封装,在代码里直白的编写


[mobilenetv3的预训练模型从这里下载](https://github.com/xiaolai-sqlai/mobilenetv3)

[mmdetection版本](https://github.com/ujsyehao/mobilenetv3-ssd)测试结果没有达到官方所说的水平,而是mAP没有到0.5

[增强函数改编自](https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py)

 [计算mAP的博客](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)

 [坐标变换部分来自](https://github.com/weiliu89/caffe/issues/155)
 
 关于提高mAP的做法

如果还想提高mAP怎么办

本repo没有在COCO数据集下训练,只训练了VOC0712数据集
如果想得到更高的mAP,可以尝试下先在COCO数据集下训练,这时候得到的模型作为预训练模型,然后在VOC0712数据集中微调,最后再在VOC2007下测试,看看是不是会得到大于0.619mAP的数

我使用第二版的部分源码做的PyTorch版的VGG-SSD,测试不同的学习率
如果按照常规做法,在训练达到某个epoch 原学习率* 0.1 , 在训练达到某个epoch 原学习率*0.01这样最终结果会得到比原论文还优秀的0.77x mAP,x代表一个很大的个位数.
如果单独采用余弦退火,mAP可以到0.771mAP,低了0.00y.
所以想增加mAP可以尝试下常规做法的学习率. 
