# HybridDiffusionDet

一个结合扩散模型与传统检测模块的混合目标检测架构。该项目将YOLOv8作为第一阶段快速生成候选框，然后使用扩散模型作为第二阶段细化检测结果，特别是在遮挡和小目标场景中提高检测精度。

## 核心创新点

- **混合架构设计**：将传统检测器（YOLOv8）作为第一阶段生成粗粒度候选框，扩散模型作为第二阶段细化候选框的定位和分类。
- **动态条件扩散**：将传统检测器的输出（框坐标、类别概率）作为扩散模型的条件输入，指导扩散过程生成更精确的检测结果。
- **轻量化优化**：采用潜在扩散模型（LDM）在低维潜在空间操作，减少计算开销。

## 架构概览

![架构图](assets/architecture.png)

HybridDiff-Detector 架构（两阶段混合模型）：

1. **阶段1（传统检测器）**：YOLOv8 快速生成初步检测结果（高召回率但可能存在定位噪声）。
2. **阶段2（扩散细化）**：以YOLO的检测框为条件，在潜在空间中对框坐标和类别进行迭代去噪。

## 安装

```bash
# 克隆仓库
git clone https://github.com/username/HybridDiffusionDet.git
cd HybridDiffusionDet

# 安装依赖
pip install -r requirements.txt

# 安装detectron2（如果pip安装失败）
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## 数据准备

### 自动下载COCO数据集

```bash
# 使用脚本下载COCO数据集
bash scripts/download_coco.sh

# 或者使用Python脚本准备数据集
python tools/prepare_dataset.py --dataset coco --data-path data
```

### 手动下载COCO数据集

如果自动下载脚本不工作，您可以手动下载COCO 2017数据集：

1. 创建数据目录：`mkdir -p data/coco`
2. 下载以下文件：
   - 训练集图像：[train2017.zip](http://images.cocodataset.org/zips/train2017.zip)
   - 验证集图像：[val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
   - 标注文件：[annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
3. 解压文件到`data/coco`目录：
   ```bash
   unzip train2017.zip -d data/coco/
   unzip val2017.zip -d data/coco/
   unzip annotations_trainval2017.zip -d data/coco/
   ```

最终的目录结构应该如下：
```
data/
└── coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   └── ...
    ├── train2017/
    │   ├── 000000000009.jpg
    │   └── ...
    └── val2017/
        ├── 000000000139.jpg
        └── ...
```

### 下载YOLOv8预训练模型

```bash
# 使用脚本下载YOLOv8预训练权重
bash scripts/download_yolo_weights.sh
```

您也可以从[Ultralytics GitHub仓库](https://github.com/ultralytics/assets/releases/tag/v0.0.0)手动下载YOLOv8预训练模型，并将它们放在`weights`目录中：

```
weights/
├── yolov8n.pt  # YOLOv8 nano
├── yolov8s.pt  # YOLOv8 small
├── yolov8m.pt  # YOLOv8 medium
├── yolov8l.pt  # YOLOv8 large
└── yolov8x.pt  # YOLOv8 xlarge
```

## 训练

```bash
# 训练YOLOv8基础检测器（如果没有预训练模型）
python tools/train_yolo.py --config configs/yolov8_base.yaml

# 训练混合扩散检测器
python tools/train_hybrid.py --config configs/hybrid_diffusion_det.yaml
```

## 评估

```bash
# 评估模型性能
python tools/eval.py --config configs/hybrid_diffusion_det.yaml --checkpoint path/to/checkpoint.pth
```

## 推理

```bash
# 单张图像推理
python tools/inference.py --config configs/hybrid_diffusion_det.yaml --checkpoint path/to/checkpoint.pth --image path/to/image.jpg

# 视频推理
python tools/inference.py --config configs/hybrid_diffusion_det.yaml --checkpoint path/to/checkpoint.pth --video path/to/video.mp4
```

## 引用

如果您在研究中使用了本项目，请引用我们的论文：

```
@article{author2023hybriddiffusiondet,
  title={HybridDiffusionDet: A Hybrid Architecture Combining Diffusion Models with Traditional Detectors for Object Detection},
  author={Author, A. and Author, B.},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2023}
}
```

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

# 数据集处理工具集

这个工具集提供了一系列用于处理目标检测数据集的实用工具，包括数据集准备、格式转换、可视化、分析和分割等功能。

## 工具列表

### 1. 数据集准备工具 (prepare_dataset.py)

用于下载和准备常用的目标检测数据集，如COCO、CrowdHuman和VisDrone。

**用法:**
```bash
python tools/prepare_dataset.py --dataset coco --data-path ./data
```

**参数:**
- `--dataset`: 要准备的数据集名称，可选值: coco, crowdhuman, visdrone, all
- `--data-path`: 数据集保存路径

### 2. 数据集格式转换工具 (convert_dataset.py)

将不同格式的数据集（如VOC、YOLO、VisDrone、CrowdHuman）转换为COCO格式。

**用法:**
```bash
python tools/convert_dataset.py --source-format voc --source-dir ./data/VOCdevkit/VOC2012 --output-dir ./data/coco_converted
```

**参数:**
- `--source-format`: 源数据集格式，可选值: voc, yolo, visdrone, crowdhuman
- `--source-dir`: 源数据集目录
- `--output-dir`: 输出目录
- `--class-names-file`: 类别名称文件（仅YOLO格式需要）

### 3. 数据集可视化工具 (visualize_dataset.py)

可视化COCO格式数据集的标注。

**用法:**
```bash
python tools/visualize_dataset.py --coco-json ./data/coco/annotations/instances_train2017.json --images-dir ./data/coco/train2017 --output-dir ./visualization
```

**参数:**
- `--coco-json`: COCO格式标注文件路径
- `--images-dir`: 图像目录路径
- `--output-dir`: 可视化结果输出目录
- `--num-samples`: 要可视化的样本数量，默认为全部
- `--no-labels`: 不显示类别标签

### 4. 数据集分析工具 (analyze_dataset.py)

分析COCO格式数据集的统计信息，包括类别分布、边界框大小分布等。

**用法:**
```bash
python tools/analyze_dataset.py --coco-json ./data/coco/annotations/instances_train2017.json --output-dir ./analysis
```

**参数:**
- `--coco-json`: COCO格式标注文件路径
- `--output-dir`: 分析结果输出目录

### 5. 数据集分割工具 (split_dataset.py)

将COCO格式数据集分割为训练集和验证集。

**用法:**
```bash
python tools/split_dataset.py --coco-json ./data/coco/annotations/instances_train2017.json --images-dir ./data/coco/train2017 --output-dir ./data/coco_split
```

**参数:**
- `--coco-json`: COCO格式标注文件路径
- `--images-dir`: 图像目录路径
- `--output-dir`: 输出目录
- `--train-ratio`: 训练集比例，默认为0.8
- `--seed`: 随机种子，默认为42
- `--category`: 按类别分割数据集，指定类别名称

## 安装依赖

```bash
pip install -r requirements.txt
```

## 示例

### 准备COCO数据集

```bash
python tools/prepare_dataset.py --dataset coco --data-path ./data
```

### 将VOC格式转换为COCO格式

```bash
python tools/convert_dataset.py --source-format voc --source-dir ./data/VOCdevkit/VOC2012 --output-dir ./data/coco_converted
```

### 可视化COCO数据集

```bash
python tools/visualize_dataset.py --coco-json ./data/coco/annotations/instances_train2017.json --images-dir ./data/coco/train2017 --output-dir ./visualization --num-samples 100
```

### 分析COCO数据集

```bash
python tools/analyze_dataset.py --coco-json ./data/coco/annotations/instances_train2017.json --output-dir ./analysis
```

### 分割COCO数据集

```bash
python tools/split_dataset.py --coco-json ./data/coco/annotations/instances_train2017.json --images-dir ./data/coco/train2017 --output-dir ./data/coco_split --train-ratio 0.8
```

### 按类别分割COCO数据集

```bash
python tools/split_dataset.py --coco-json ./data/coco/annotations/instances_train2017.json --images-dir ./data/coco/train2017 --output-dir ./data/coco_split_person --category person
``` 
