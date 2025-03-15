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

```bash
# 下载COCO数据集
bash scripts/download_coco.sh

# 准备数据集
python tools/prepare_dataset.py --dataset coco --data-path data/coco
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

本项目采用 [MIT 许可证](LICENSE)。 # HybridDiffusionDet
