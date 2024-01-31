# 多模态融合模型

## 准备

此实现基于 Python3. 若要运行代码，需要以下依赖项：

- matplotlib==3.8.0

- numpy==1.24.4

- pandas==1.5.1

- Pillow==10.2.0

- scikit_learn==1.1.3

可以直接运行以下命令安装：

```python
pip install -r requirements.txt
```

## 仓库结构

```json
|-- multi.py                      多模态融合模型
|-- adjust_lr.py                  学习率调参
|-- functions.py                  模型及其他函数实现，包括数据预处理部分等
|-- image_only.py                 消融实验——只输入图像
|-- text_only.py                  消融实验——只输入文本
|-- 10215501450_刘钊瑄_实验五.pdf   报告
|-- requirements.txt              运行所需依赖项
|-- Readme.md                     仓库及运行说明
```

## 运行方式

运行多模态融合模型

```python
python main.py --model multi
```

运行 image_only

```python
python main.py --model image_only
```

运行 text_only

```python
python main.py --model text_only
```

## 参考仓库

- https://github.com/RecklessRonan/GloGNN/blob/master
- https://github.com/JinZT-Git/HS-Codes/tree/main/EfficientPose-master
