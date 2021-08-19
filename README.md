# while True: 卷
# &emsp; Cream of the Crop
<br><br><br>



## 项目介绍：
&emsp; 本仓库为“while True: 卷”队完成飞桨论文复现挑战赛（第四期）中论文#39所使用的仓库

### 论文标题：
#39 Cream of the Crop: Distilling Prioritized Paths For One-Shot Neural Architecture Search :star::star:

### 论文摘要：
One-shot weight sharing methods have recently drawn great attention in neural architecture search due to high efficiency and competitive performance. However, weight sharing across models has an inherent deficiency, i.e., insufficient training of subnetworks in hypernetworks. To alleviate this problem, we present a simple yet effective architecture distillation method. The central idea is that subnetworks can learn collaboratively and teach each other throughout the training process, aiming to boost the convergence of individual models. We introduce the concept of prioritized path, which refers to the architecture candidates exhibiting superior performance during training. Distilling knowledge from the prioritized paths is able to boost the training of subnetworks. Since the prioritized paths are changed on the fly depending on their performance and complexity, the final obtained paths are the cream of the crop. We directly select the most promising one from the prioritized paths as the final architecture, without using other complex search methods, such as reinforcement learning or evolution algorithms. The experiments on ImageNet verify such path distillation method can improve the convergence ratio and performance of the hypernetwork, as well as boosting the training of subnetworks. The discovered architectures achieve superior performance compared to the recent MobileNetV3 and EfficientNet families under aligned settings. Moreover, the experiments on object detection and more challenging search space show the generality and robustness of the proposed method. Code and models are available at https://github.com/microsoft/cream.git2 .

### 论文内容概述：

神经结构搜索优化 Distilling Prioritized Paths

----

## 关于源码：
<br>
我们已经完成了源论文验证集的运行，以下简要介绍运行情况：
（更新于21/8/19）

<br>
<br>

### 关于论文源码的修改:
<br>
在论文作者发布的源码中，我们发现了一些瑕疵。为了使代码能够在无cuda的cpu环境下更好地运行，我们对源代码进行了如下更改：

- 根据源码README指引修改了[Cream/configs/test/test.yaml](Cream/configs/test/test.yaml)，使得程序将根据Cream-14模型（权重已载入，详见**权重与数据**）进行预测。
- 为了方便进行cuda与cpu相关的修改，将[Cream/tools/test.py](Cream/tools/test.py)中`import`的`timm.data.loader`文件迁出至[Cream/tools/](Cream/tools/)文件夹下，并进行了一系列改动
- 由于torch版本差异，对于[Cream/lib/core/test.py](Cream/lib/core/test.py)文件中现**第55行**的误差计算函数`accuracy()`，我们不在采用timm库中的版本，而是将库中的函数粘贴到此文件中并将`return`行的`torch.tensor.view()`函数用`torch.tensor.reshape()`替代，以规避用前者将二维张量转换为一维时的报错
- 注释了[Cream/lib/core/test.py](Cream/lib/core/test.py)、[Cream/tools/loader.py](Cream/tools/loader.py)、[Cream/tools/test.py](Cream/tools/test.py)中涉及cuda的部分， 以避免cuda缺失造成的报错
- 在[Cream/tools/test.py](Cream/tools/test.py)中引入了`torch.device("cpu")`，并注释了GPU相关的部分，以适应CPU运行环境
- 为了方便观察运行进度，修改了[Cream/lib/core/test.py](Cream/lib/core/test.py)文件中现**第73行**的输出log的条件，使得每一个batch完成后均会在命令行窗口输出日志

### 运行环境
<br>

由于原文作者所指定的部分版本的依赖已经无法获得，我们更新了一套新的依赖，详见[Cream\requirements_new](Cream\requirements_new)

### 权重与数据
<br>

- 我们已经将Cream-14模型的权重下载至[Cream\experiments\workspace\ckps\14.pth.tar](Cream\experiments\workspace\ckps\14.pth.tar)。原文作者还在[百度网盘](https://pan.baidu.com/s/1TqQNm2s14oEdyNPimw3T9g)（提取码：wqw6）保存了其他可用权重。虽然这些权重同样可以使修改后的源码正常运行，但我们并不是非常建议您使用这些模型权重，尤其是较大的模型权重（如最大的Cream-604）处理检验集，因为如此规模的网络并不适合在CPU环境下运行（我们曾尝试运行Cream-604网络。虽然它在检验集的评估下有明显更优的表现，但其运行时间是Cream-14网络的近20倍）。
- 原文使用了[ILSVRC-2012](https://image-net.org/challenges/LSVRC/2012/index.php)-Task_1（即原README所谓的ImageNet-2012）的[检验集](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)进行模型的评估。因为该数据集过大，您需要自行下载并将其中的所有图片移动至[Cream\data\imagenet\val](Cream\data\imagenet\val)
- 原文作者提供的[标记文件](Cream\data\imagenet\valprep.sh)已随此仓库同步至Github。下载检验集后，请将其一并移动到[Cream\data\imagenet\val](Cream\data\imagenet\val)，并在该文件夹下启动GitBash运行如下命令以运行该文件：
```
sh valprep.sh
```
- 该文件会自动将检验集图片按标签分类为文件夹。此过程可能长达数小时，请耐心等待。