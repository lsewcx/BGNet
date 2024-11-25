from net.bgnet import Net
from torchinfo import summary
import torch

# 初始化模型
model = Net()

# 打印模型参数量
summary(model, input_size=(1, 3, 416, 416))