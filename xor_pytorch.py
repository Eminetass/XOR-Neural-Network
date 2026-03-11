"""
XOR Probleminin Yapay Sinir Ağları ile Çözümü — PyTorch Uygulaması
===================================================================
Bu dosya XOR problemini PyTorch framework'ü ile çözer.
NumPy uygulamasının sonuçlarıyla karşılaştırma ve PyTorch'un
otomatik türev (autograd) mekanizmasının gösterimi amaçlanmıştır.

Farklı mimariler ve optimizerlar karşılaştırılmaktadır:
  • SGD vs Adam
  • Derin (3 katmanlı) vs Sığ (2 katmanlı) ağ
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# ─── Veri ───────────────────────────────────────────────────────────────────
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])

Y = torch.tensor([[0.], [1.], [1.], [0.]])

# ─── Model Tanımları ─────────────────────────────────────────────────────────

class ShallowXOR(nn.Module):
    """Klasik 2 katmanlı MLP: 2→4→1"""
    def __init__(self, hidden=4, activation=nn.Sigmoid()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            activation,
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    def get_hidden(self, x):
        return self.net[1](self.net[0](x))  # Gizli katman aktivasyonu


class DeepXOR(nn.Module):
    """Derin 3 katmanlı MLP: 2→8→4→1"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class ResidualXOR(nn.Module):
    """
    Yenilikçi yaklaşım: Residual bağlantılı XOR çözücü
    Skip connection ile gradyan akışı iyileştirilmiştir.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)
        self.act = nn.Tanh()
        self.proj = nn.Linear(2, 8)  # skip connection projeksiyon

    def forward(self, x):
        identity = self.proj(x)          # Boyut eşitleme
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out) + identity)   # Residual toplama
        return torch.sigmoid(self.fc3(out))


# ─── Eğitim Fonksiyonu ───────────────────────────────────────────────────────
def train_model(model, optimizer, epochs=5000, verbose=True):
    criterion = nn.BCELoss()
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and epoch % 1000 == 0:
            preds = (output > 0.5).float()
            acc = (preds == Y).float().mean().item() * 100
            print(f"  Epoch {epoch:5d} | Loss: {loss.item():.6f} | Acc: {acc:.0f}%")

    return losses


def evaluate_model(model, name):
    model.eval()
    with torch.no_grad():
        output = model(X)
        preds  = (output > 0.5).float()
        acc    = (preds == Y).float().mean().item() * 100
        loss   = nn.BCELoss()(output, Y).item()

    print(f"\n  [{name}] Sonuçlar:")
    print(f"  {'x1':>3} | {'x2':>3} | {'Beklenen':>8} | {'Ham Çıktı':>10} | {'Tahmin':>6}")
    print(f"  ----|-----|---------|-----------|-------")
    for xi, yi, oi, pi in zip(X, Y, output, preds):
        print(f"   {int(xi[0])} |   {int(xi[1])} |       {int(yi.item())}   |"
              f"   {oi.item():.4f}  |    {int(pi.item())}")
    print(f"  Doğruluk: {acc:.0f}% | Son Loss: {loss:.6f}")
    return acc, loss


# ─── Görselleştirme ─────────────────────────────────────────────────────────
def plot_decision_boundary_torch(model, title, ax):
    h = 0.01
    xx, yy = np.meshgrid(np.arange(-0.3, 1.31, h),
                         np.arange(-0.3, 1.31, h))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        Z = model(grid).numpy().reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=50, cmap="RdYlGn", alpha=0.85)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

    X_np = X.numpy()
    Y_np = Y.numpy().flatten()
    colors = ["#e74c3c" if y == 0 else "#2ecc71" for y in Y_np]
    labels = ["(0,0)=0", "(0,1)=1", "(1,0)=1", "(1,1)=0"]
    for xi, c, lbl in zip(X_np, colors, labels):
        ax.scatter(*xi, c=c, s=200, zorder=5, edgecolors="black", linewidths=1.5)
        ax.annotate(lbl, xi, textcoords="offset points",
                    xytext=(8, 4), fontsize=9, fontweight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.grid(True, alpha=0.3)


def plot_training_curves(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    fig.suptitle("PyTorch — Eğitim Kayıp Eğrileri", fontsize=14, fontweight="bold")

    if len(results) == 1:
        axes = [axes]

    colors = ["#3498db", "#e67e22", "#9b59b6", "#2ecc71"]
    for ax, (name, losses, _), color in zip(axes, results, colors):
        ax.plot(losses, color=color, linewidth=1.5, alpha=0.9)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/pytorch_training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_all_decisions(results_models):
    n = len(results_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    fig.suptitle("PyTorch — Farklı Mimariler Karar Sınırları", fontsize=14, fontweight="bold")

    if n == 1:
        axes = [axes]

    for ax, (name, model, acc, final_loss) in zip(axes, results_models):
        plot_decision_boundary_torch(
            model,
            f"{name}\nAcc={acc:.0f}% | Loss={final_loss:.4f}",
            ax
        )

    plt.tight_layout()
    plt.savefig("figures/pytorch_decision_boundaries.png", dpi=150, bbox_inches="tight")
    plt.show()


def visualize_hidden_activation(model, ax):
    """Gizli katman aktivasyonlarını görselleştirir"""
    model.eval()
    with torch.no_grad():
        h = model.get_hidden(X).numpy()

    X_np = X.numpy()
    Y_np = Y.numpy().flatten()
    colors = ["#e74c3c" if y == 0 else "#2ecc71" for y in Y_np]
    labels = ["(0,0)=0", "(0,1)=1", "(1,0)=1", "(1,1)=0"]

    for i, (color, lbl) in enumerate(zip(colors, labels)):
        ax.scatter(h[i, 0], h[i, 1], c=color, s=300,
                   edgecolors="black", linewidths=2, zorder=5)
        ax.annotate(lbl, (h[i, 0], h[i, 1]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=9, fontweight="bold")

    ax.set_title("Gizli Katman Uzayı\n(Doğrusal Ayrılabilirlik)", fontsize=11)
    ax.set_xlabel("Nöron 1")
    ax.set_ylabel("Nöron 2")
    ax.grid(True, alpha=0.3)


# ─── Ana Akış ────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(42)
    os.makedirs("figures", exist_ok=True)

    print("=" * 65)
    print("XOR PROBLEMİ — PyTorch UYGULAMASI")
    print("=" * 65)

    # ── 1. Sığ ağ: SGD vs Adam karşılaştırması ──
    print("\n[1] SGD vs Adam Optimizer Karşılaştırması (ShallowXOR)")
    results_loss  = []
    results_model = []

    configs = [
        ("SGD (lr=0.5)",   ShallowXOR(hidden=4, activation=nn.Sigmoid()),
         lambda m: optim.SGD(m.parameters(), lr=0.5)),
        ("Adam (lr=0.01)", ShallowXOR(hidden=4, activation=nn.Sigmoid()),
         lambda m: optim.Adam(m.parameters(), lr=0.01)),
    ]

    for name, model, opt_fn in configs:
        print(f"\n  Eğitiliyor: {name}")
        opt    = opt_fn(model)
        losses = train_model(model, opt, epochs=5000)
        acc, fl = evaluate_model(model, name)
        results_loss.append((name, losses, fl))
        results_model.append((name, model, acc, fl))

    plot_training_curves(results_loss)

    # ── 2. Derin ağ ──
    print("\n[2] Derin Ağ (2→8→4→1, Tanh)")
    deep_model = DeepXOR()
    deep_opt   = optim.Adam(deep_model.parameters(), lr=0.01)
    deep_losses = train_model(deep_model, deep_opt, epochs=3000)
    acc_d, fl_d = evaluate_model(deep_model, "DeepXOR")
    results_model.append(("DeepXOR\n(2→8→4→1)", deep_model, acc_d, fl_d))

    # ── 3. Residual ağ ──
    print("\n[3] Residual (Skip Connection) Ağı")
    res_model  = ResidualXOR()
    res_opt    = optim.Adam(res_model.parameters(), lr=0.01)
    res_losses = train_model(res_model, res_opt, epochs=3000)
    acc_r, fl_r = evaluate_model(res_model, "ResidualXOR")
    results_model.append(("ResidualXOR\n(Skip Conn.)", res_model, acc_r, fl_r))

    # ── 4. Tüm karar sınırları ──
    plot_all_decisions(results_model)

    # ── 5. Gizli katman görselleştirme ──
    print("\n[4] Gizli Katman Uzayı Analizi")
    best_model = ShallowXOR(hidden=4, activation=nn.Sigmoid())
    best_opt   = optim.Adam(best_model.parameters(), lr=0.01)
    train_model(best_model, best_opt, epochs=5000, verbose=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    visualize_hidden_activation(best_model, ax)
    plt.tight_layout()
    plt.savefig("figures/pytorch_hidden_space.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n✓ Tüm görseller 'figures/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
