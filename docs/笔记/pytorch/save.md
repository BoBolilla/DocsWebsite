# 保存模型

## 保存和加载整个模型

### 保存模型

```python
torch.save(model, 'model.pth')
```

### 加载模型

```python
model = torch.load('model.pth')
```

## 只保存模型的状态字典（state_dict）

## 保存模型状态字典

```python
torch.save(model.state_dict(), 'model_state_dict.pth')
```

## 加载模型状态字典

加载state_dict时需要手动重新实例化模型。

```python
model = Net()  # 你需要先定义好模型架构
model.load_state_dict(torch.load('model_state_dict.pth'))
```

与保存整个模型相比，保存 state_dict 更加灵活，它只包含模型的参数，而不依赖于完整的模型定义，这意味着你可以在不同的项目中加载模型参数，甚至只加载部分模型的权重。举个例子，对于分类模型，即便你保存的是完整的网络参数，也可以仅导入特征提取层部分，当然，直接导入完整模型再拆分实际上是一样的。对于不完全匹配的模型，加载时可以通过设置 strict=False 来忽略某些不匹配的键。

## 参考

[PyTorch 模型保存与加载的三种常用方式](https://blog.csdn.net/weixin_42426841/article/details/142624088)
