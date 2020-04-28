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
如果您要直接测试，模型改名为checkpoint_ssd300.pth.tar
模型下载地址
链接：https://pan.baidu.com/s/1BZz9X7w3xaopf1dKLkyMcA 
提取码：gwwv


模型测试结果 mAP 0.679 (未在COCO数据集做预训练版本)

 使用步骤
一 下载VOC数据集之后,将VOCtrainval_06-Nov-2007和VOCtest_06-Nov-2007合并在一起
数据集下载 可以看这里
https://blog.csdn.net/flyfish1986/article/details/95367745

二 先打开create_data_lists.py文件
改成自己数据集的路径

三 如果想使用mobilenetv3的预训练模型,打开mode.py
找到    def init_weights(self, pretrained=None):#"./mbv3_large.old.pth.tar"
替换成 def init_weights(self, pretrained="./mbv3_large.old.pth.tar"):


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
