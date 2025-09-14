# 😀 Facial Expression Recognition using CNN

This project performs facial emotion recognition using a Convolutional Neural Network (CNN) trained on the **Facial Expressions Dataset**.

---

## 📁 Dataset
- Dataset cloned from:  
  `https://github.com/muxspace/facial_expressions.git`
  
- Structure:
  - `data/legend.csv`: Contains the mapping between image file names and corresponding emotion labels.
  - `images/`: Folder containing all emotion images.
  
- The dataset is organized into:
  - `master_data/training/<emotion>/`  
  - `master_data/testing/<emotion>/`

---

## ⚡ Data Preparation
- Parsed `legend.csv` to create a mapping of emotions to image files.
- Created training/testing folders with an 80%-20% split.
- Copied images into respective directories.

---

## 🧱 Model Architecture
A simple CNN with:
1. Three convolutional layers (Conv2D + MaxPooling2D).
2. Flatten layer.
3. Dense layer with 1024 neurons (ReLU).
4. Output layer with 8 neurons (softmax) corresponding to 8 emotions.

---

## ⚙️ Model Compilation
- Optimizer: Adam (`learning rate=0.01`)
- Loss: `categorical_crossentropy`
- Metrics: Accuracy

---

## 🏃 Training Setup
- Image size: 100x100
- Batch size: 128
- Data normalization (rescaling by 1/255)
- Early stopping based on validation accuracy (`patience=2`, `min_delta=0.01`)
- Training for up to 10 epochs

---

## 🚀 Example Usage

```python
model.fit_generator(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=[EarlyStopping(monitor='val_acc', patience=2, min_delta=0.01)]
)
```

🎯 Key Insights

Simple CNN works well for emotion classification.

Early stopping prevents overfitting.

Organized dataset into train/test splits for reproducible results.
