# ImageNet 权重QAT — 验证三值权重是否泛化到 8-768

## 前置
1. 权重 `checkpoint-284.pth.tar`(266MB, Spikingformer-8-768)：从 Google Drive 下
   `https://drive.google.com/drive/folders/1DE4qm-9vKNwDfJAdYYrDdPFc7_cGo2nK`
   (GitHub 100MB 限制放不下, 单独下)
2. ImageNet 数据：**优先用 GPU 服务器已有的 ImageNet**(标准 ImageFolder: 含 train/ 和 val/)。
   若服务器没有：需自备 ImageNet-1k(train~140GB); 或用 HF 流式(见末尾)。
3. 依赖同 CIFAR: torch torchvision spikingjelly timm cupy numpy

## 跑(关键问题=三值泛化, 先只跑三值+4bit)
```bash
CK=./checkpoint-284.pth.tar; IMN=/path/to/imagenet   # 含 train/ val/
python qat_finetune_imagenet.py --bits 32  --epochs 0 --ckpt $CK --data $IMN            # FP基线应≈75.8%
python qat_finetune_imagenet.py --bits tern --epochs 2 --ckpt $CK --data $IMN --subset 0.1 > log_in_tern.txt 2>&1
python qat_finetune_imagenet.py --bits 4    --epochs 2 --ckpt $CK --data $IMN --subset 0.1 > log_in_w4.txt   2>&1
```
- `--subset 0.1`：只用 10% 训练集微调(加速出信号); 想更准去掉或调大, 但更慢。
- 8-768 在 224×224 上比 CIFAR 重很多, batch/epoch 按显存调。

## 看什么
log 末尾 `=== ImageNet X: FP=.. PTQ=.. QAT-best=.. ===`。
判据: **三值 QAT 掉 <2-3% → 三值权重泛化成立**(CIFAR 已验只掉0.5%, 就看大模型稳不稳)。

## 没有本地 ImageNet 时的 HF 流式(备选)
装 `datasets`，用 `load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)` 流式取子集微调
(需 HF token + 已接受 gated 协议)。要走这条我再给你改数据加载部分。
