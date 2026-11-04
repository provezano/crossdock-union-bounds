"""MLP parameter predictor (paper Sect. 4.7; architecture/optimiser from the public ML.py, training loop = SPEC D8).
Inputs: the 7 paper features; outputs: [NGEN, CXPB, mu, lambda, ts]; MinMax scaling of inputs and outputs (ML.py);
net = Linear(7,1000)-Dropout(.5)-ReLU-Linear(1000,1000)-Dropout(.2)-ReLU-Linear(1000,5) (ML.py); Adam lr 1e-4;
MSE loss; full batch; EPOCHS epochs; torch seed 0; 1 thread.
usage: mlp.py train TUNEDIR MODEL.pt [EPOCHS]   |   (import) Predictor(MODEL.pt)"""
import sys, os, json, glob, time, hashlib
import numpy as np
import torch, torch.nn as nn
torch.set_num_threads(1)
OUT = ["NGEN", "CXPB", "mu", "lam", "ts"]
EPOCHS = 2000


def net(d_in, d_out, H=1000):
    return nn.Sequential(nn.Linear(d_in, H), nn.Dropout(0.5), nn.ReLU(), nn.Linear(H, H), nn.Dropout(0.2), nn.ReLU(),
                         nn.Linear(H, d_out))


class MinMax:
    def __init__(self, lo, hi): self.lo = np.asarray(lo, float); self.hi = np.asarray(hi, float)
    @classmethod
    def fit(cls, X): return cls(X.min(0), X.max(0))
    def rng(self): r = self.hi - self.lo; return np.where(r > 0, r, 1.0)
    def tf(self, X): return (np.asarray(X, float) - self.lo) / self.rng()
    def inv(self, Y): return np.asarray(Y, float) * self.rng() + self.lo


def load_tuning(tdir):
    X, Y, keys = [], [], []
    for f in sorted(glob.glob(f"{tdir}/*.json")):
        r = json.load(open(f)); X.append(r["features"]); Y.append([r["best_params"][k] for k in OUT]); keys.append(r["key"])
    return np.array(X, float), np.array(Y, float), keys


def train(tdir, path, epochs=EPOCHS, seed=0):
    X, Y, keys = load_tuning(tdir); sx, sy = MinMax.fit(X), MinMax.fit(Y)
    torch.manual_seed(seed)
    m = net(X.shape[1], Y.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=1e-4); lossf = nn.MSELoss()
    x = torch.tensor(sx.tf(X), dtype=torch.float32); y = torch.tensor(sy.tf(Y), dtype=torch.float32)
    t0 = time.process_time(); hist = []
    for ep in range(epochs):
        m.train(); opt.zero_grad(); loss = lossf(m(x), y); loss.backward(); opt.step()
        if ep % 100 == 0 or ep == epochs - 1: hist.append((ep, float(loss)))
    m.eval()
    with torch.no_grad(): fit_mse = float(lossf(m(x), y))
    torch.save(dict(state=m.state_dict(), sx=[sx.lo.tolist(), sx.hi.tolist()], sy=[sy.lo.tolist(), sy.hi.tolist()],
                    d_in=X.shape[1], d_out=Y.shape[1], keys=keys, epochs=epochs, seed=seed), path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    info = dict(n=len(keys), epochs=epochs, cpu=time.process_time() - t0, train_mse_scaled_eval=fit_mse,
                loss_hist=hist, sha256=h, target_mean=Y.mean(0).tolist(), target_std=Y.std(0).tolist())
    json.dump(info, open(path + ".info.json", "w"), indent=1)
    return info


class Predictor:
    def __init__(self, path):
        ck = torch.load(path, weights_only=False)
        self.m = net(ck["d_in"], ck["d_out"]); self.m.load_state_dict(ck["state"]); self.m.eval()
        self.sx = MinMax(*ck["sx"]); self.sy = MinMax(*ck["sy"])
    def raw(self, feats):
        with torch.no_grad():
            z = self.m(torch.tensor(self.sx.tf(np.array([feats])), dtype=torch.float32)).numpy()
        return self.sy.inv(z)[0].tolist()


if __name__ == "__main__":
    if sys.argv[1] == "train":
        ep = int(sys.argv[4]) if len(sys.argv) > 4 else EPOCHS
        print(json.dumps(train(sys.argv[2], sys.argv[3], ep), indent=1)[:2000])
