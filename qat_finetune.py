# -*- coding: utf-8 -*-
"""
权重量化 QAT 微调 — 从 ckpt-405 起，扫权重位宽(8/4/2/三值)，出"位宽 vs 精度"曲线。
自包含：模型用本目录 model.py 的 vit_snn；CIFAR-10 用 torchvision 自动下载。
在**有 GPU 的服务器**上跑。用法见文件末 / README_qat.md。

  python qat_finetune.py --bits 4  --epochs 20 --ckpt /path/checkpoint-405.pth.tar
  python qat_finetune.py --bits 2  ...
  python qat_finetune.py --bits tern ...
  python qat_finetune.py --bits 32 --epochs 0   # 32=不量化，仅验证FP基线精度(应~95.8%)
"""
import argparse, os, torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.utils.parametrize as P
from spikingjelly.clock_driven import functional
from model import vit_snn   # 同目录 model.py

# ---------------- 假量化 + STE ----------------
def ste(q, w):                       # 前向用q, 反向梯度直通到w
    return w + (q - w).detach()

class FakeQuant(nn.Module):
    """对权重做每输出通道对称量化(INT-b)或三值(tern), 带STE。挂到 conv.weight 上。"""
    def __init__(self, bits):
        super().__init__()
        self.bits = bits            # int 或 'tern'
    def forward(self, w):           # w: [out, in, ...]
        oc = w.shape[0]
        wf = w.reshape(oc, -1)
        if self.bits == 'tern':     # 三值 TWN: Δ=0.7*mean|w|, α=Δ以上|w|均值
            delta = 0.7 * wf.abs().mean(1, keepdim=True)
            mask = (wf.abs() > delta).float()
            alpha = (wf.abs() * mask).sum(1, keepdim=True) / (mask.sum(1, keepdim=True) + 1e-8)
            q = torch.sign(wf) * mask * alpha
        else:                       # INT-b 对称: s=max|w|/(2^(b-1)-1)
            b = int(self.bits); qmax = 2 ** (b - 1) - 1
            s = wf.abs().max(1, keepdim=True).values / qmax + 1e-8
            q = torch.clamp(torch.round(wf / s), -qmax - 1, qmax) * s
        return ste(q.reshape_as(w), w)

def apply_qat(model, bits, skip_first=True):
    """给所有 Conv1d/Conv2d 权重挂假量化; 默认跳过第一层卷积(block0)保FP。"""
    n = 0; first = True
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            if skip_first and first:
                first = False; continue    # 第一层(编码卷积)保FP,常规做法
            first = False
            P.register_parametrization(m, "weight", FakeQuant(bits)); n += 1
    print(f"[qat] 已对 {n} 个卷积层挂 {bits}-bit 假量化 (skip_first={skip_first})")

# ---------------- 数据 (torchvision, 自动下载) ----------------
def loaders(data_root, bs=64, workers=4):
    import torchvision as tv, torchvision.transforms as T
    mean, std = [0.4914,0.4822,0.4465], [0.247,0.2435,0.2616]
    tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean,std)])
    te = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
    d_tr = tv.datasets.CIFAR10(data_root, train=True,  download=True, transform=tr)
    d_te = tv.datasets.CIFAR10(data_root, train=False, download=True, transform=te)
    return (torch.utils.data.DataLoader(d_tr, bs, shuffle=True,  num_workers=workers, pin_memory=True),
            torch.utils.data.DataLoader(d_te, bs, shuffle=False, num_workers=workers, pin_memory=True))

@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); correct = total = 0
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        functional.reset_net(model)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item(); total += y.numel()
    return 100.0 * correct / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bits', default='4')                # 8/4/2/tern/32(FP)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='./cifar_data_tv')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--no-skip-first', action='store_true')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'
    print("[env] device =", dev, "| torch cuda =", torch.cuda.is_available())

    # 模型 (CIFAR: Spikingformer-4-384)
    model = vit_snn(img_size_h=32, img_size_w=32, patch_size=4, in_channels=3, num_classes=10,
                    embed_dims=384, num_heads=12, mlp_ratios=4, depths=4, sr_ratios=1, T=4).to(dev)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck.get('model', ck)); sd = {(k[7:] if k.startswith('module.') else k):v for k,v in sd.items()}
    print("[load]", model.load_state_dict(sd, strict=False))

    tr, te = loaders(args.data, args.bs)

    if args.bits != '32':
        acc_fp = evaluate(model, te, dev); print(f"[FP基线] acc = {acc_fp:.2f}%")
        apply_qat(model, args.bits, skip_first=not args.no_skip_first)
        acc_ptq = evaluate(model, te, dev); print(f"[PTQ, {args.bits}-bit, 未微调] acc = {acc_ptq:.2f}%")
    else:
        print("[FP] 仅验证:", f"{evaluate(model, te, dev):.2f}%"); return

    # QAT 微调
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs) if args.epochs>0 else None
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for i,(x,y) in enumerate(tr):
            x,y = x.to(dev), y.to(dev)
            functional.reset_net(model)
            out = model(x); loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            if i % 200 == 0: print(f"  ep{ep} it{i} loss {loss.item():.3f}", flush=True)
        if sched: sched.step()
        acc = evaluate(model, te, dev); best = max(best, acc)
        print(f"[QAT {args.bits}-bit] epoch {ep}  acc = {acc:.2f}%  (best {best:.2f}%)", flush=True)
    print(f"\n=== 结果 {args.bits}-bit: FP={acc_fp:.2f}%  PTQ={acc_ptq:.2f}%  QAT-best={best:.2f}% ===")

if __name__ == '__main__':
    main()
