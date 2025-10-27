# model/train.py
import os, time, torch, numpy as np, argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from .config import LR, WEIGHT_DECAY, EPOCHS, PATIENCE, TARGET_SR, BATCH_SIZE, NUM_WORKERS
from .dataset_drone import load_split, make_loader
from .model_drone import DroneClassifier

def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def evaluate(model, loader, device, amp=False):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        cm = torch.cuda.amp.autocast if (amp and device.type=="cuda") else torch.cpu.amp.autocast
        with cm():
            for wav, y in loader:
                wav, y = wav.to(device), y.to(device)
                prob = torch.sigmoid(model(wav))
                ys.append(y.cpu()); ps.append(prob.cpu())
    y = torch.cat(ys).numpy(); p = torch.cat(ps).numpy()
    return dict(auc=roc_auc_score(y,p), ap=average_precision_score(y,p),
                f1=f1_score(y,(p>=0.5).astype(int)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--amp", action="store_true", help="mixed precision")
    parser.add_argument("--limit-steps", type=int, default=0, help="for quick dry-run")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    tr, va, te = load_split()
    tr_loader = make_loader(tr, train=True, batch_size=args.batch_size, num_workers=args.num_workers)
    va_loader = make_loader(va, train=False, batch_size=args.batch_size, num_workers=args.num_workers)

    y_tr = np.array([int(x["label"]) for x in tr])
    n0, n1 = (y_tr==0).sum(), (y_tr==1).sum()
    pos_weight = torch.tensor([n0/max(1,n1)], dtype=torch.float32).to(device)

    model = DroneClassifier().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(optim, T_max=max(10, args.epochs//2))

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type=="cuda")

    best_auc, wait = 0.0, 0
    total_steps = len(tr_loader) if args.limit_steps == 0 else min(args.limit_steps, len(tr_loader))

    for ep in range(args.epochs):
        model.train(); loss_sum = 0.0
        t0 = time.perf_counter()
        for step, (wav, y) in enumerate(tr_loader, start=1):
            if args.limit_steps and step > args.limit_steps: break
            wav, y = wav.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)

            if args.amp and device.type=="cuda":
                with torch.cuda.amp.autocast():
                    logits = model(wav); loss = criterion(logits, y)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optim); scaler.update()
            else:
                logits = model(wav); loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()

            loss_sum += loss.item() * len(y)

            # 진행률/ETA 출력
            if step % 50 == 0 or step == total_steps:
                elapsed = time.perf_counter() - t0
                sps = step / max(elapsed, 1e-6)
                eta = (total_steps - step) / max(sps, 1e-6)
                print(f"Ep {ep+1} [{step}/{total_steps}] loss {loss.item():.4f} | {sps:.2f} steps/s | ETA {eta/60:.1f}m")

        sched.step()
        tr_loss = loss_sum / (total_steps * args.batch_size)

        m = evaluate(model, va_loader, device, amp=args.amp)
        print(f"Epoch {ep+1}/{args.epochs} done. train_loss {tr_loss:.4f} | AUC {m['auc']:.4f} AP {m['ap']:.4f} F1 {m['f1']:.4f}")

        if m["auc"] > best_auc:
            best_auc, wait = m["auc"], 0
            os.makedirs("chk", exist_ok=True)
            torch.save({"model": model.state_dict()}, "chk/best.pt")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stop"); break

    # 1-에폭 점검용: 테스트는 옵션
    print("Done. Best AUC:", best_auc)

if __name__ == "__main__":
    main()
