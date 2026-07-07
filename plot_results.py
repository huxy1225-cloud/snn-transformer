# -*- coding: utf-8 -*-
"""画位宽-精度曲线 + 折算 CIM 权重面积/能耗。结果存 PNG + 打印表格。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['axes.unicode_minus'] = False  # 标签全英文, 用默认 DejaVu

# ---- QAT 实测结果 ----
FP = 95.59
# name, effective_bits(用于面积/能耗折算), x轴位置, PTQ, QAT
rows = [
    ("8-bit",  8.0,   8.0,  95.76, 95.15),
    ("4-bit",  4.0,   4.0,  95.42, 95.28),
    ("三值",   1.58,  1.58, 51.40, 95.09),  # log2(3)=1.58 信息位; 数字实现常用2物理bit
    ("2-bit",  2.0,   2.0,  13.45, 93.36),
]

names = ['ternary' if r[0]=='三值' else r[0] for r in rows]
bits  = np.array([r[1] for r in rows])
xpos  = np.array([r[2] for r in rows])
ptq   = np.array([r[3] for r in rows])
qat   = np.array([r[4] for r in rows])

# 按 x 排序画线
order = np.argsort(xpos)

fig, ax = plt.subplots(figsize=(8,5), dpi=140)
ax.axhline(FP, color='#888', ls='--', lw=1.4, label=f'FP32 baseline {FP:.2f}%')
ax.plot(xpos[order], qat[order], 'o-', color='#2563eb', lw=2.2, ms=9, label='QAT-best (finetuned)')
ax.plot(xpos[order], ptq[order], 's--', color='#f59e0b', lw=1.6, ms=7, alpha=0.85, label='PTQ (no finetune)')

for x,y,n in zip(xpos, qat, names):
    ax.annotate(f'{y:.2f}%', (x,y), textcoords='offset points', xytext=(0,10),
                ha='center', fontsize=9, color='#1e3a8a', fontweight='bold')
for x,y in zip(xpos, ptq):
    ax.annotate(f'{y:.1f}', (x,y), textcoords='offset points', xytext=(0,-16),
                ha='center', fontsize=8, color='#b45309')

ax.set_xlabel('Weight bit-width (bit, ternary≈1.58)', fontsize=11)
ax.set_ylabel('CIFAR-10 Top-1 accuracy (%)', fontsize=11)
ax.set_title('Spikingformer-4-384 Weight Quantization: Bit-width vs Accuracy (weight-only)', fontsize=11.5, fontweight='bold')
ax.set_xticks([1.58,2,4,8]); ax.set_xticklabels(['tern\n1.58','2','4','8'])
ax.set_ylim(0, 100); ax.grid(alpha=0.25); ax.legend(loc='center right', fontsize=9)

fig.tight_layout()
fig.savefig('bitwidth_accuracy.png', dpi=140, bbox_inches='tight')
print("saved bitwidth_accuracy.png")

# ---- CIM 权重面积/能耗折算 ----
# 假设: 激活为1-bit脉冲, MAC退化为"有脉冲则累加权重"。
#   权重存储面积 ∝ 权重位宽 (SRAM cell 数 / 器件数)
#   累加器/MAC 能耗 ∝ 权重位宽 (加法器位宽)
# 相对 FP32(32b) 和相对 实用INT8 两个基准各折算一次。
print("\n=== CIM 权重面积/能耗折算 (∝ 权重位宽) ===")
print(f"{'方案':<8}{'有效bit':>8}{'QAT精度':>9}{'掉点':>8}{'vs FP32省':>11}{'vs INT8省':>11}")
for n,b,q in zip(names, bits, qat):
    save_fp = (1 - b/32.0)*100
    save_i8 = (1 - b/8.0)*100
    drop = q - FP
    print(f"{n:<8}{b:>8.2f}{q:>8.2f}%{drop:>7.2f}%{save_fp:>10.1f}%{save_i8:>10.1f}%")
print("FP32     32.00  95.59%   0.00%       0.0%       —")
print("\n注: 三值信息位=1.58; 若数字实现按2物理bit存, 面积/能耗折算同2-bit(vs FP32省93.8%, vs INT8省75%)。")
