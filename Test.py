import torch
import torch.nn as nn
import torch.optim as optim

# 简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 检查是否可以使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 创建一个模型实例并移动到设备上
model = SimpleNet().to(device)

# 创建优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建一个随机输入和输出数据
input = torch.randn(1, 10).to(device)
target = torch.randn(1, 1).to(device)

# 前向传播
output = model(input)

# 计算损失
criterion = nn.MSELoss()
loss = criterion(output, target)

# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 打印损失值
print("Loss:", loss.item())


## 8. Results
