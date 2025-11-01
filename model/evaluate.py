import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from model.dataset_drone import load_split
from model.model_drone import DroneClassifier
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--val", action="store_true", help="Use validation set instead of test set")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
train_ds, val_ds, test_ds = load_split()
dataset = val_ds if args.val else test_ds
loader = DataLoader(dataset, batch_size=32)

model = DroneClassifier().to(device)
model.load_state_dict(torch.load("chk/drone_model.pt"))
model.eval()

y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logit = model(x)
        prob = torch.sigmoid(logit).cpu().numpy().flatten()
        pred = (prob > 0.3).astype(int)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred)
        y_prob.extend(prob)

acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
auc = roc_auc_score(y_true, y_prob)
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
