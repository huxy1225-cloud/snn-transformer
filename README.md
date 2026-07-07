# snn-transformer — 权重QAT微调运行包

Spikingformer-4-384 (CIFAR-10) 的权重量化感知训练(QAT)，扫权重位宽(8/4/2/三值)出"位宽-精度"曲线。**只量权重**(激活本就是1-bit脉冲)。

## 快速开始 (GPU 服务器)
```bash
pip install -r requirements.txt          # 注意 cupy 版本按你的 CUDA 选
CK=./checkpoint-405.pth.tar
python qat_finetune.py --bits 32 --epochs 0 --ckpt $CK           # 先验FP基线≈95.8%
python qat_finetune.py --bits 8   --epochs 20 --ckpt $CK > log_w8.txt   2>&1
python qat_finetune.py --bits 4   --epochs 20 --ckpt $CK > log_w4.txt   2>&1
python qat_finetune.py --bits 2   --epochs 30 --ckpt $CK > log_w2.txt   2>&1
python qat_finetune.py --bits tern --epochs 30 --ckpt $CK > log_tern.txt 2>&1
```
CIFAR-10 首次自动下载(torchvision)。每个 log 末尾打印 `=== 结果 X-bit: FP=.. PTQ=.. QAT-best=.. ===`。

详见 [README_qat.md](README_qat.md)。
