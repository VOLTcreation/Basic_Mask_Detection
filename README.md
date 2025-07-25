# 😷 Real-Time Face Mask Detection

This is a Python application that uses OpenCV, TensorFlow/Keras, and cvlib to perform **real-time face mask detection** using your webcam.

It detects faces and classifies them as:
- ✅ `mask on`
- ❌ `mask off`

---

## ⚠️ Disclaimer

> **The `.h5` model (`mask_data.h5`) used here is not trained on a large or diverse dataset.**
>
> As a result, predictions may not be very accurate and should **not** be used for any critical applications. This is a **demo only**.

---

## 📦 Requirements

Install dependencies with:

```bash
pip install opencv-python tensorflow cvlib numpy
