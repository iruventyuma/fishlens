# 🐟 FishLens — Fish Species Identifier

A deep learning web application that identifies **9 fish species** from images using **ResNet50 Transfer Learning**, with nutrition info, cooking tips, and freshness guidance.

---

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 97.58% |
| Validation Accuracy | 98.3% |
| Architecture | ResNet50 (Transfer Learning) |
| Dataset | Large Scale Fish Dataset (Kaggle) |

---

## 🐠 Supported Species

| Species | Edible |
|---------|--------|
| Black Sea Sprat | ✅ |
| Gilt-Head Bream | ✅ |
| Hourse Mackerel | ✅ |
| Red Mullet | ✅ |
| Red Sea Bream | ✅ |
| Sea Bass | ✅ |
| Shrimp | ✅ |
| Striped Red Mullet | ✅ |
| Trout | ✅ |

---

## 🧠 Model Architecture

```
Input Image (224x224)
        ↓
ResNet50 (Frozen base layers)
        ↓
Linear(2048 → 512) + ReLU + Dropout(0.4)
        ↓
Linear(512 → 256) + ReLU + Dropout(0.3)
        ↓
Linear(256 → 9 classes)
        ↓
Softmax → Predicted Species
```

---

## 🖥️ App Features

- Upload any fish image and get instant species prediction
- Confidence score with visual probability chart
- Nutrition info per 100g (protein, fat, omega-3, calories)
- Best cooking methods for each species
- Freshness & buying tips
- Safe to eat indicator

---





