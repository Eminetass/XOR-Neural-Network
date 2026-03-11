"""
XOR Probleminin Yapay Sinir Ağları ile Çözümü - NumPy (Sıfırdan Uygulama)
==========================================================================
Bu dosya XOR problemini hiçbir derin öğrenme kütüphanesi kullanmadan,
yalnızca NumPy ile sıfırdan uygular. Geri yayılım (backpropagation)
algoritması elle implemente edilmiştir.

Mimari: 2 → 4 → 1  (Giriş → Gizli Katman → Çıkış)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─── Veri ───────────────────────────────────────────────────────────────────
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

Y = np.array([[0],
              [1],
              [1],
              [0]], dtype=float)


# ─── Aktivasyon Fonksiyonları ────────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_act(z):
    return np.tanh(z)

def tanh_deriv(z):
    return 1 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)


# ─── MLP Sınıfı ─────────────────────────────────────────────────────────────
class MLP:
    """
    Çok Katmanlı Perceptron (Multi-Layer Perceptron)
    Geri yayılım algoritması ile eğitilir.
    """

    def __init__(self, hidden_size=4, activation="sigmoid", lr=0.1, seed=42):
        np.random.seed(seed)
        self.lr = lr
        self.hidden_size = hidden_size

        # Aktivasyon seçimi
        acts = {
            "sigmoid": (sigmoid, sigmoid_deriv),
            "tanh":    (tanh_act, tanh_deriv),
            "relu":    (relu, relu_deriv),
        }
        self.act, self.act_deriv = acts[activation]
        self.activation_name = activation

        # Ağırlıklar (Xavier başlatma)
        self.W1 = np.random.randn(2, hidden_size) * np.sqrt(2 / 2)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, 1))

        self.losses = []

    def forward(self, X):
        """İleri yayılım"""
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.act(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)          # Çıkış her zaman sigmoid
        return self.A2

    def loss(self, y_pred, y_true):
        """Binary Cross-Entropy kayıp fonksiyonu"""
        eps = 1e-8
        return -np.mean(y_true * np.log(y_pred + eps) +
                        (1 - y_true) * np.log(1 - y_pred + eps))

    def backward(self, X, y_true):
        """Geri yayılım — zincirleme türev"""
        m = X.shape[0]

        # Çıkış katmanı gradyanları
        dA2 = self.A2 - y_true                          # dL/dA2
        dW2 = self.A1.T @ dA2 / m
        db2 = np.sum(dA2, axis=0, keepdims=True) / m

        # Gizli katman gradyanları
        dA1 = dA2 @ self.W2.T
        dZ1 = dA1 * self.act_deriv(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Ağırlık güncellemesi (Gradient Descent)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, Y, epochs=10000):
        """Eğitim döngüsü"""
        for epoch in range(epochs):
            y_pred = self.forward(X)
            l = self.loss(y_pred, Y)
            self.losses.append(l)
            self.backward(X, Y)

            if epoch % 2000 == 0:
                preds = (y_pred > 0.5).astype(int)
                acc = np.mean(preds == Y) * 100
                print(f"  Epoch {epoch:5d} | Loss: {l:.6f} | Acc: {acc:.0f}%")

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


# ─── Görselleştirme ─────────────────────────────────────────────────────────
def plot_decision_boundary(model, title, ax):
    """Karar sınırını 2D grid üzerinde çizer"""
    h = 0.01
    x_min, x_max = -0.3, 1.3
    y_min, y_max = -0.3, 1.3

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=50, cmap="RdYlGn", alpha=0.8)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

    colors = ["#e74c3c" if y == 0 else "#2ecc71" for y in Y.flatten()]
    labels_txt = ["(0,0)→0", "(0,1)→1", "(1,0)→1", "(1,1)→0"]
    for i, (xi, color, lbl) in enumerate(zip(X, colors, labels_txt)):
        ax.scatter(*xi, c=color, s=200, zorder=5, edgecolors="black", linewidths=1.5)
        ax.annotate(lbl, xi, textcoords="offset points",
                    xytext=(8, 5), fontsize=9, fontweight="bold")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Giriş x₁")
    ax.set_ylabel("Giriş x₂")
    ax.grid(True, alpha=0.3)


def plot_hidden_space(model, ax):
    """Gizli katman uzayını görselleştirir (XOR'un doğrusal ayrılabilir hale gelişi)"""
    hidden = model.act(X @ model.W1 + model.b1)  # (4, hidden_size)

    colors = ["#e74c3c", "#2ecc71", "#2ecc71", "#e74c3c"]
    labels = ["(0,0)=0", "(0,1)=1", "(1,0)=1", "(1,1)=0"]
    for i in range(4):
        ax.scatter(hidden[i, 0], hidden[i, 1],
                   c=colors[i], s=300, zorder=5,
                   edgecolors="black", linewidths=2, label=labels[i])
        ax.annotate(labels[i], (hidden[i, 0], hidden[i, 1]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)

    ax.set_title("Gizli Katman Uzayı\n(XOR doğrusal ayrılabilir hale geldi!)", fontsize=11)
    ax.set_xlabel("Nöron 1 aktivasyonu")
    ax.set_ylabel("Nöron 2 aktivasyonu")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def show_single_perceptron_failure():
    """Tek katmanlı perceptronun XOR'u neden çözemediğini gösterir"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Neden Tek Katmanlı Perceptron XOR'u Çözemez?",
                 fontsize=14, fontweight="bold")

    problems = [("AND", np.array([0, 0, 0, 1])),
                ("OR",  np.array([0, 1, 1, 1]))]

    for ax, (name, labels) in zip(axes, problems):
        colors = ["#e74c3c" if l == 0 else "#2ecc71" for l in labels]
        for xi, c in zip(X, colors):
            ax.scatter(*xi, c=c, s=250, edgecolors="black", linewidths=1.5, zorder=5)

        # Doğrusal karar sınırı çizilebilir
        x_line = np.linspace(-0.2, 1.2, 100)
        if name == "AND":
            ax.plot(x_line, 1.5 - x_line, "b--", linewidth=2, label="Karar sınırı")
        else:
            ax.plot(x_line, 0.5 - x_line, "b--", linewidth=2, label="Karar sınırı")

        ax.set_title(f"{name} → Doğrusal ayrılabilir ✓", fontsize=12)
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(-0.4, 1.4)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/single_perceptron_failure.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("→ XOR için böyle bir düz çizgi YOKTUR (doğrusal ayrılamaz!)\n")


def compare_activations():
    """Farklı aktivasyon fonksiyonlarını karşılaştırır"""
    print("\n" + "="*60)
    print("AKTİVASYON FONKSİYONU KARŞILAŞTIRMASI")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Farklı Aktivasyon Fonksiyonları ile XOR Çözümü",
                 fontsize=14, fontweight="bold")

    for ax, act_name in zip(axes, ["sigmoid", "tanh", "relu"]):
        model = MLP(hidden_size=4, activation=act_name, lr=0.5, seed=42)
        print(f"\n[{act_name.upper()}]")
        model.train(X, Y, epochs=10000)

        preds = model.predict(X)
        acc = np.mean(preds == Y) * 100
        final_loss = model.losses[-1]
        print(f"  Son Loss: {final_loss:.6f} | Doğruluk: {acc:.0f}%")
        print(f"  Tahminler: {preds.flatten()} | Beklenen: {Y.flatten().astype(int)}")

        plot_decision_boundary(model,
                               f"{act_name.capitalize()}\nLoss={final_loss:.4f} | Acc={acc:.0f}%",
                               ax)

    plt.tight_layout()
    plt.savefig("figures/activation_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    os.makedirs("figures", exist_ok=True)
    print("=" * 60)
    print("XOR PROBLEMİ — YAPAY SİNİR AĞLARI İLE ÇÖZÜM")
    print("Sıfırdan NumPy Uygulaması")
    print("=" * 60)

    print("\n[1] XOR Gerçek Çıktı Tablosu:")
    print("  x1 | x2 | XOR")
    print("  ---|----|----|")
    for xi, yi in zip(X, Y):
        print(f"   {int(xi[0])} |  {int(xi[1])} |  {int(yi[0])}")

    print("\n[2] Tek katmanlı perceptron neden başarısız?")
    show_single_perceptron_failure()

    print("\n[3] Ana Model Eğitimi (Sigmoid, 4 gizli nöron):")
    model = MLP(hidden_size=4, activation="sigmoid", lr=0.5, seed=0)
    model.train(X, Y, epochs=10000)

    preds = model.predict(X)
    raw = model.forward(X)
    print("\n  Sonuç Tablosu:")
    print("  x1 | x2 | Beklenen | Ham Çıktı | Tahmin")
    print("  ---|----|---------|-----------|----|")
    for xi, yi, ri, pi in zip(X, Y.flatten(), raw.flatten(), preds.flatten()):
        print(f"   {int(xi[0])} |  {int(xi[1])} |     {int(yi)}    |  {ri:.4f}   |   {pi}")

    acc = np.mean(preds == Y) * 100
    print(f"\n  Toplam Doğruluk: {acc:.0f}%")

    # Kayıp eğrisi
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("XOR — NumPy MLP Analizi", fontsize=14, fontweight="bold")

    axes[0].plot(model.losses, color="#3498db", linewidth=1.5)
    axes[0].set_title("Eğitim Kayıp Eğrisi")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.4)

    plot_decision_boundary(model, "Karar Sınırı\n(XOR Çözümü)", axes[1])
    plot_hidden_space(model, axes[2])

    plt.tight_layout()
    plt.savefig("figures/main_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n[4] Aktivasyon karşılaştırması başlatılıyor...")
    compare_activations()

    print("\n✓ Tüm görseller 'figures/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
