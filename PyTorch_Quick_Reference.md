
# PyTorch 快速食用手册

## 1. 构建数据集
使用 `torch.utils.data.Dataset` 和 `torch.utils.data.DataLoader` 进行数据加载。

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建数据集和加载器
dataset = MyDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 2. 构建神经网络

### 使用 `nn.Sequential`
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(28*28, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)
```

### 自定义模型
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
```

---

## 3. 损失函数和优化器

### 内置损失函数
- **线性回归**：`nn.MSELoss()`
- **二分类**：`nn.BCEWithLogitsLoss()`
- **多分类**：`nn.CrossEntropyLoss()`

```python
criterion = nn.CrossEntropyLoss()
```

### 优化器
```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01)
```

---

## 4. 训练和测试

### 训练模型
```python
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### 测试模型
```python
model.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        # 评估性能
```

---

## 5. 部署模型（ONNX）

### 导出为 ONNX
```python
dummy_input = torch.randn(1, 28*28)
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)
```

### 使用 ONNX Runtime 加载
```python
import onnxruntime as ort

ort_session = ort.InferenceSession("model.onnx")
outputs = ort_session.run(None, {"input": dummy_input.numpy()})
```

---

## 6. 激活函数用法

| **激活函数**  | **典型位置**          | **任务类型**           | **输出范围**       |
|--------------|----------------------|-----------------------|--------------------|
| ReLU         | 隐藏层               | 通用                  | `[0, ∞)`          |
| Sigmoid      | 输出层（二分类）       | 二分类任务             | `[0, 1]`          |
| Softmax      | 输出层（多分类）       | 多分类任务             | `[0, 1]` (概率)   |
| 无激活函数    | 输出层（线性回归）     | 连续值回归任务         | 实数范围            |

---

## 7. 常见问题
- **线性回归与分类的区别**：
  - 线性回归：输出连续值，使用 `MSELoss`。
  - 分类：输出概率分布，使用 `BCEWithLogitsLoss` 或 `CrossEntropyLoss`。

- **激活函数位置**：
  - 隐藏层：通常用 `ReLU`。
  - 输出层：根据任务选择 `Sigmoid` 或 `Softmax`，线性回归无需激活函数。

---

**祝您使用愉快！**
