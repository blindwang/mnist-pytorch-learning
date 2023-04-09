# 手写体数字识别 （Pytorch完成）

## 具体要求

- 网络结构：不同激活函数，不同的层数，每一层不同的神经元
- 优化方法：不同优化方法
- 如果GPU允许，改变不同batch_size

调整上述不同的结构，对ACC的影响，以及损失函数的
变化的曲线图和自己的分析。

## TL;DR



## 代码说明

数据保存在`/data`中，运行过程保存在`/logs`中。可以在`model.py`增加模型，在`run.py`中修改超参。

### 快速开始

```shell
python run.py
```

### 可视化
训练过程中运行下面的代码可以实现可视化

```shell
tensorboard --logdir ./logs
```

要想利用tensorboard保存的数据绘制loss和acc曲线，首先要将log文件夹上传到 TensorBoard.dev。
```shell
tensorboard dev upload --logdir ./logs
```
再运行`utils.py`中的`plot_loss_acc`函数即可。需要传入实验的编号（如`k2BUjrGBRWm8KkuayjqYXQ`）

运行`plot.py`亦可

这是我上传的实验链接~

`compare_base`: [base_link](https://tensorboard.dev/experiment/k2BUjrGBRWm8KkuayjqYXQ/)
`compare_optim`: [optim_link](https://tensorboard.dev/experiment/23g5u8JpQGGZNxGoZmECGg)


### ipynb版代码

`mnist_exp.ipynb`文件是在colab中可以成功运行，本地环境配置可能会有些麻烦（如tensorboard等）


## 调试心得

1. 在修改网络结构时，更换激活函数和增加网络深度都导致了模型loss下降减缓，需要增加epoch和增大学习率来缓解。

猜想原因：
- Sigmoid适合二分类而非多分类
- 在网络深度增加时，产生了梯度消失等问题

2. tensorboard每次运行新的进程要删掉之前的，否则会导致曲线扭曲（因为混杂了之前的进程）
3. cpu中num_worker为0！否则报错：
```text
RuntimeError: DataLoader worker (pid(s) 23200) exited unexpectedly
```
接下来是一些代码讲解（大多数内容在ipynb文件中写得更清楚）

## 1. 准备数据集

首先，我们需要准备手写数字数据集，使用MNIST数据集，Pytorch已经提供了该数据集的下载和加载。
![png](mnist_exp_files/mnist_exp_4_2.png)

dataloader中images的shape为(64, 1, 28, 28)，表示有64张大小为28x28的灰度图像，labels的shape为(64,)，表示有64个标签。

这个过程由`dataloader.py`中的`build_dataloader`函数完成。

## 2. 定义卷积神经网络

接下来，定义一个卷积神经网络模型，包含两个卷积层、两个池化层和两个全连接层。其中，激活函数采用ReLU，优化方法采用SGD。

- nn.Conv2d：卷积层，输入为1个通道，输出为32个通道，卷积核大小为3x3。（nn.Conv2d(1, 32, 3)）
- nn.MaxPool2d：池化层，输入为2x2的窗口大小，步长为2，对输入进行最大池化操作。（nn.MaxPool2d(2, 2)）
- nn.Linear：全连接层，输入大小为64x5x5，输出大小为128。（nn.Linear(64 * 5 * 5, 128)）

维度解释：

假设输入图片大小为28x28，经过第一次卷积层后，输出大小为26x26x32（32是卷积核数量）。然后经过第一次池化层，输出大小为13x13x32。接着经过第二次卷积层，输出大小为11x11x64。再经过第二次池化层，输出大小为5x5x64。因此，经过这些层后，特征图的大小为5x5x64，即64个5x5的特征图。所以，全连接层的输入大小即为64x5x5，输出大小为128。

```
Net(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=1600, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
  (relu): ReLU()
)
```

模型定义在`model.py`中，分别定义了四个网络，用于比较不同网络架构对模型训练的影响

- 简单的卷积神经网络模型（Net）
- LeNet模型（LeNet）
- 使用Sigmoid激活函数的卷积神经网络模型（NetSigmoid）
- 深层卷积神经网络模型（NetDeep）

## 模型训练和评估

这个过程由`train.py`/`evaluate.py`中的`train`/`evaluate`函数完成。

在训练过程中可以实时观察loss变化趋势，便于及时调整参数，修改模型架构。

![img.png](mnist_exp_files/img.png)

![img.png](mnist_exp_files/img-loss.png)

## 修改超参数、网络结构、优化方法，观察模型效果

我们可以尝试修改超参数，比如改变激活函数、改变层数、改变神经元个数、改变优化方法、改变batch_size等，然后观察模型在测试集上的准确率和损失函数的变化。

也即用到在model中定义的四个网络，观察模型效果的变化。