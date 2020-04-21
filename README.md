# MobileNetV3-SSD-Compact-Version
MobileNetV3 SSD的简洁版本

环境

版本 PyTorch 1.4

一 下载VOC数据集之后,将VOCtrainval_06-Nov-2007和VOCtest_06-Nov-2007合并在一起


二 先打开create_data_lists.py文件
改成自己数据集的路径

三 如果想使用mobilenetv3的预训练模型,打开mode.py
找到    def init_weights(self, pretrained=None):#"./mbv3_large.old.pth.tar"
替换成 def init_weights(self, pretrained="./mbv3_large.old.pth.tar"):
您可以固定backbone层只训练最后几层或者全部训练,代码已写好,只要换成False或True

四 运行训练命令python train.py

五 测试mAP命令 python eval.py



mobilenetv3的预训练模型从这里下载
https://github.com/xiaolai-sqlai/mobilenetv3

mmdetection版本
https://github.com/ujsyehao/mobilenetv3-ssd

增强函数改编自
 https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

 计算mAP的博客
 https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

 坐标变换部分来自
 https://github.com/weiliu89/caffe/issues/155
