# -*- coding: utf-8 -*-
"""
权重量化 QAT — ImageNet 版 (Spikingformer-8-768)。验证"三值权重能否泛化到大模型"。
自包含：模型用同目录 model.py 的 vit_snn；ImageNet 用 torchvision ImageFolder。
在**有 GPU + 有 ImageNet 的服务器**上跑。

数据(二选一)：
  --data /path/to/imagenet   # 标准 ImageFolder, 下含 train/ 和 val/  (多数GPU服务器已有)
  (若无本地ImageNet, 见 README_imagenet.md 的 HF 流式方案)

用法(关键问题=三值泛化, 先只跑三值)：
  python qat_finetune_imagenet.py --bits 32 --epochs 0 --ckpt ckpt-284.pth.tar --data /imagenet   # FP基线(应≈75.8%)
  python qat_finetune_imagenet.py --bits tern --epochs 2 --ckpt ckpt-284.pth.tar --data /imagenet --subset 0.1
  python qat_finetune_imagenet.py --bits 4   --epochs 2 --ckpt ckpt-284.pth.tar --data /imagenet --subset 0.1
"""
import argparse, os, torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.utils.parametrize as P
from spikingjelly.clock_driven import functional
from model import vit_snn

def ste(q, w): return w + (q - w).detach()
class FakeQuant(nn.Module):
    def __init__(self, bits): super().__init__(); self.bits = bits
    def forward(self, w):
        oc = w.shape[0]; wf = w.reshape(oc, -1)
        if self.bits == 'tern':
            delta = 0.7 * wf.abs().mean(1, keepdim=True); mask = (wf.abs() > delta).float()
            alpha = (wf.abs() * mask).sum(1, keepdim=True) / (mask.sum(1, keepdim=True) + 1e-8)
            q = torch.sign(wf) * mask * alpha
        else:
            b = int(self.bits); qmax = 2 ** (b - 1) - 1
            s = wf.abs().max(1, keepdim=True).values / qmax + 1e-8
            q = torch.clamp(torch.round(wf / s), -qmax - 1, qmax) * s
        return ste(q.reshape_as(w), w)

def apply_qat(model, bits, skip_first=True):
    n = 0; first = True
    for _, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            if skip_first and first: first = False; continue
            first = False
            P.register_parametrization(m, "weight", FakeQuant(bits)); n += 1
    print(f"[qat] {n} 个卷积挂 {bits} 假量化 (skip_first={skip_first})")

def build_loaders(root, bs, workers):
    import torchvision as tv, torchvision.transforms as T
    m, s = [0.485,0.456,0.406], [0.229,0.224,0.225]
    tr_t = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(m,s)])
    te_t = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(m,s)])
    tr = tv.datasets.ImageFolder(os.path.join(root,'train'), tr_t)
    te = tv.datasets.ImageFolder(os.path.join(root,'val'),   te_t)
    return (torch.utils.data.DataLoader(tr, bs, shuffle=True,  num_workers=workers, pin_memory=True),
            torch.utils.data.DataLoader(te, bs, shuffle=False, num_workers=workers, pin_memory=True))

@torch.no_grad()
def evaluate(model, dl, dev, max_batches=None):
    model.eval(); c = t = 0
    for i,(x,y) in enumerate(dl):
        x,y = x.to(dev), y.to(dev); functional.reset_net(model)
        c += (model(x).argmax(1)==y).sum().item(); t += y.numel()
        if max_batches and i+1>=max_batches: break
    return 100.0*c/t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bits', default='tern'); ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--ckpt', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--subset', type=float, default=1.0, help='只用训练集的这个比例微调(0~1), 加速')
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--no-skip-first', action='store_true')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'
    print("[env] device", dev, "cuda", torch.cuda.is_available())

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck.get('model', ck)); sd = {(k[7:] if k.startswith('module.') else k):v for k,v in sd.items()}
    emb = sd['head.weight'].shape[1]; depth = 1+max(int(k.split('.')[1]) for k in sd if k.startswith('block.'))
    print(f"[cfg] Spikingformer-{depth}-{emb}")
    model = vit_snn(img_size_h=224, img_size_w=224, patch_size=16, in_channels=3, num_classes=1000,
                    embed_dims=emb, num_heads=emb//64, mlp_ratios=4, depths=depth, sr_ratios=1, T=4).to(dev)
    print("[load]", model.load_state_dict(sd, strict=False))
    tr, te = build_loaders(args.data, args.bs, args.workers)

    if args.bits == '32':
        print(f"[FP] val acc = {evaluate(model, te, dev):.2f}%"); return
    acc_fp = evaluate(model, te, dev); print(f"[FP基线] {acc_fp:.2f}%")
    apply_qat(model, args.bits, skip_first=not args.no_skip_first)
    acc_ptq = evaluate(model, te, dev); print(f"[PTQ {args.bits}] {acc_ptq:.2f}%")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_tr = len(tr); stop = int(n_tr*args.subset) if args.subset<1.0 else n_tr
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for i,(x,y) in enumerate(tr):
            if i>=stop: break
            x,y = x.to(dev), y.to(dev); functional.reset_net(model)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            if i%100==0: print(f"  ep{ep} it{i}/{stop} loss {loss.item():.3f}", flush=True)
        acc = evaluate(model, te, dev); best=max(best,acc)
        print(f"[QAT {args.bits}] ep{ep} val {acc:.2f}% (best {best:.2f}%)", flush=True)
    print(f"\n=== ImageNet {args.bits}: FP={acc_fp:.2f}% PTQ={acc_ptq:.2f}% QAT-best={best:.2f}% ===")

if __name__ == '__main__':
    main()
