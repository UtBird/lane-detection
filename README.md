# 🚗 Lane Detection with OpenCV

Bu proje, Python ve OpenCV kullanarak video görüntülerinde şerit tespiti yapmayı amaçlamaktadır. Kamera kalibrasyonu, perspektif dönüşümü ve görüntü işleme teknikleriyle, araç kamerasından alınan görüntülerde şerit çizgileri tespit edilerek sürüş güvenliği artırılabilir.

## 📁 Proje Yapısı

```
lane-detection/
├── camera_cal/             # Kamera kalibrasyon görüntüleri
├── yol15.mp4               # Örnek sürüş videosu
├── camera_calibration.py   # Kamera kalibrasyon kodu
├── deneme.py               # Test amaçlı kodlar
├── forward_display.py      # Görselleştirme arayüzü
├── line_function.py        # Şerit çizgisi işlemleri
├── main.py                 # Ana uygulama dosyası
├── process.py              # Görüntü işleme pipeline'ı
├── Screenshot_10.png       # Örnek çıktı görseli
├── images.jpg              # Örnek giriş görüntüsü
```

## ⚙️ Özellikler

* 📷 **Kamera Kalibrasyonu**: `camera_calibration.py` dosyası ile kamera distorsiyonu giderilir.
* 🛣️ **Perspektif Dönüşümü**: Kuş bakışı görünüm elde edilerek şerit tespiti kolaylaştırılır.
* 🎯 **Görüntü İşleme**: Renk eşikleri ve kenar tespiti ile şerit çizgileri belirlenir.
* 📈 **Polinom Uydurma**: Tespit edilen şerit çizgilerine ikinci dereceden polinom uydurulur.
* 🖼️ **Görselleştirme**: `forward_display.py` ile işlenmiş görüntüler üzerinde şeritler gösterilir.

## 🧩 Kurulum

1. **Depoyu Klonlayın**

   ```bash
   git clone https://github.com/UtBird/lane-detection.git
   cd lane-detection
   ```

2. **Gerekli Kütüphaneleri Yükleyin**

   Python 3.6+ sürümü ile aşağıdaki kütüphaneleri yükleyin:

   ```bash
   pip install numpy opencv-python
   ```

## 🚀 Kullanım

1. **Kamera Kalibrasyonu**

   ```bash
   python camera_calibration.py
   ```

   Bu adım, `camera_cal/` klasöründeki kalibrasyon görüntülerini kullanarak kamera matrisini ve distorsiyon katsayılarını hesaplar.

2. **Şerit Tespiti**

   ```bash
   python main.py
   ```

   Bu komut, `yol15.mp4` videosunu işleyerek şerit tespiti yapar ve sonuçları görselleştirir.
