# XOR Probleminin Yapay Sinir Ağlarıyla Çözümü

> 2025-2026 Derin Öğrenme Dersi — Ödev 2
> XOR problemini NumPy (sıfırdan) ve PyTorch ile çözen, görsel analize sahip proje

---

## Projeye Genel Bakış

XOR problemi, yapay sinir ağlarının doğrusal olmayan problemleri neden çözebileceğini gösteren en klasik örnektir.
Bu proje:
- XOR'un neden tek katmanlı perceptron ile çözülemediğini **kanıtlar**
- Geri yayılım algoritmasını **sıfırdan NumPy** ile implement eder
- **PyTorch** ile farklı mimarileri karşılaştırır (SGD vs Adam, Residual bağlantılar)
- Gizli katmanın veriye ne yaptığını **görsel olarak** gösterir

---

## Dosya Yapısı

```
XOR-Neural-Network/
├── XOR_Problem_Analysis.ipynb  ← Ana Jupyter Notebook (başlangıç noktası)
├── xor_scratch.py              ← NumPy ile sıfırdan MLP + backpropagation
├── xor_pytorch.py              ← PyTorch: SGD vs Adam, Derin ağ, Residual ağ
├── RAPOR.md                    ← Türkçe teknik rapor
├── requirements.txt            ← Bağımlılıklar
└── figures/                    ← Otomatik oluşturulan görseller
```

---

## Kurulum ve Çalıştırma

```bash
# 1. Repoyu klonla
git clone https://github.com/KULLANICI_ADIN/XOR-Neural-Network.git
cd XOR-Neural-Network

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3a. Jupyter Notebook (önerilen)
jupyter notebook XOR_Problem_Analysis.ipynb

# 3b. NumPy uygulaması
python xor_scratch.py

# 3c. PyTorch uygulaması
python xor_pytorch.py
```

---

## Sonuçlar

| Model | Mimari | Optimizer | Doğruluk |
|-------|--------|-----------|----------|
| NumPy MLP (Sigmoid) | 2→4→1 | SGD | **%100** |
| NumPy MLP (Tanh) | 2→4→1 | SGD | **%100** |
| NumPy MLP (ReLU) | 2→4→1 | SGD | **%100** |
| PyTorch Shallow | 2→4→1 | Adam | **%100** |
| PyTorch Deep | 2→8→4→1 | Adam | **%100** |
| **PyTorch Residual** | **2→8(skip)→1** | **Adam** | **%100** |

---

## Yenilikçi Katkı: Residual XOR Ağı

Modern mimari (ResNet, BERT) prensiplerini XOR problemine uygulayan **skip connection** yaklaşımı:

```python
class ResidualXOR(nn.Module):
    def forward(self, x):
        identity = self.proj(x)         # Boyut eşitleme
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out) + identity)  # Skip bağlantı
        return torch.sigmoid(self.fc3(out))
```

---

## Temel Bulgular

1. **Tek katmanlı perceptron XOR'u asla çözemez** (doğrusal ayrılamazlık)
2. **Sadece 4 gizli nöron yeterli** — minimum parametreyle %100 doğruluk
3. **Gizli katman, veriyi doğrusal ayrılabilir uzaya dönüştürür**
4. **Adam optimizeri SGD'den ~6× daha hızlı yakınsar**

---

## Rapor

Projenin tam teknik açıklaması için [RAPOR.md](RAPOR.md) dosyasına bakınız.

---

## Kaynaklar

- Minsky & Papert (1969). *Perceptrons*
- Rumelhart, Hinton & Williams (1986). Backpropagation — *Nature*
- He et al. (2016). Deep Residual Learning — *CVPR*
- Goodfellow et al. (2016). *Deep Learning* — MIT Press
