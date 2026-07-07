# 权重 QAT 微调 — GPU 服务器运行说明

目的：从 Spikingformer-4-384 的 ckpt-405 出发，扫权重位宽(8/4/2/三值)，得"位宽 vs 精度"曲线，判断 CD1(权重量化)能压到几 bit 不掉精度。**只量权重(激活本就是1-bit脉冲)。**

## 1. 要传到 GPU 服务器的东西
- 本目录两个文件：`model.py`、`qat_finetune.py`（放同一目录）
- 权重：`ckpt_cifar/checkpoint-405.pth.tar`（37MB）
- CIFAR-10 **不用传**（脚本用 torchvision 首次自动下载 ~170MB）

传输示例（在本机跑）：
```bash
scp model.py qat_finetune.py  user@GPU服务器:/path/qat/
scp /home/research/huxinyue/research/ckpt_cifar/checkpoint-405.pth.tar  user@GPU服务器:/path/qat/
```

## 2. GPU 服务器的环境
需要：`torch(带cuda) torchvision spikingjelly timm cupy numpy`
```bash
pip install torch torchvision spikingjelly timm cupy-cuda12x numpy   # cupy版本按CUDA选
```
- **注意**：model.py 里 LIF 用 `backend='cupy'`。GPU 服务器装了 cupy 就直接跑（快）。
- 若不想装 cupy：把 model.py 里所有 `backend='cupy'` 改成 `backend='torch'`（慢一点但能跑）。

## 3. 运行（按顺序）
```bash
cd /path/qat/
CK=./checkpoint-405.pth.tar

# ① 先验 FP 基线(应≈95.8%, 确认模型/权重/数据都对)
python qat_finetune.py --bits 32 --epochs 0 --ckpt $CK

# ② 扫位宽(每个跑一次, 各~20 epoch, 单卡 CIFAR 几小时内)
python qat_finetune.py --bits 8   --epochs 20 --ckpt $CK   > log_w8.txt   2>&1
python qat_finetune.py --bits 4   --epochs 20 --ckpt $CK   > log_w4.txt   2>&1
python qat_finetune.py --bits 2   --epochs 30 --ckpt $CK   > log_w2.txt   2>&1
python qat_finetune.py --bits tern --epochs 30 --ckpt $CK  > log_tern.txt 2>&1
```
可选：`--device cuda:0` 指定卡；`--no-skip-first` 连第一层卷积也量化(默认第一层保FP)。

## 4. 看什么结果
每个 log 末尾有一行：
```
=== 结果 X-bit: FP=95.xx%  PTQ=??.??%  QAT-best=??.??% ===
```
- **FP**：全精度基线；**PTQ**：直接量化不微调(通常低bit掉很多)；**QAT-best**：微调后最好精度。
- 汇总成"位宽→QAT精度"表：这就是 CD1 的位宽-精度曲线。
  - 判据：若 4-bit QAT 掉 <1%、2-bit/三值掉 <2-3% → CD1 成立，权重能大幅压位宽。
  - 若低 bit 掉太多 → CD1 收益打折，停在能接受的位宽。

## 5. 跑完把 log 传回来
把 4 个 log 传回本机(或贴给我)，我据此画位宽-精度曲线 + 折算 CIM 权重面积/能耗省多少，判断 CD1 值不值得进设计。

## 已验证/未验证
- 假量化+STE 数学已在 CPU 验证正确(误差随位宽增大、STE梯度直通)。
- **训练流程未在GPU实测**(本机无GPU)。首次跑若报错(多半是环境/backend/路径)，把报错贴给我，我改。
