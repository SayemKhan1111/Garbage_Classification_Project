# Garbage Classification Project – Week 2

## 🎯 Objective
Classify garbage images into 6 categories:
- cardboard
- glass
- metal
- paper
- plastic
- trash

---

## 📂 Dataset
- Source: [Kaggle - Trash Type Image Dataset](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)
- Total Images: ~2400
- Structure: Folder-wise (one folder per class)

---

## 🧰 Tools & Libraries Used
- Python
- TensorFlow / Keras
- EfficientNetV2B2 (Transfer Learning)
- Scikit-learn (for evaluation)
- Gradio (for UI)
- Hugging Face Spaces (for deployment)

---

## ✅ Week 1 Summary (30%)
- TensorFlow environment setup
- Dataset downloaded and structured
- Model built using MobileNetV2
- Initial training completed
- Model saved as `garbage_classifier.h5`

---

## ✅ Week 2 Progress (60%)
- Switched to EfficientNetV2B2 (more accurate than MobileNetV2)
- Applied data augmentation (flip, zoom, contrast, etc.)
- Used class weights to handle class imbalance
- Trained model with callbacks:
  - EarlyStopping
  - ReduceLROnPlateau
- Evaluation done:
  - Confusion matrix (Seaborn)
  - Classification report (Sklearn)
- Achieved:
  - Training Accuracy ~97%
  - Validation Accuracy ~88%
- Model saved as `efficientnetv2b2_model.keras`
- Built live prediction app using Gradio
- Deployed on Hugging Face Spaces ✅

🔗 **Live App:**  
[Click to Open Gradio App](https://huggingface.co/spaces/Sayemkhan1111/sayem-garbage-classifier)

---

## 📌 Next Steps (Week 3 Plan)
- Fine-tune deeper layers of EfficientNet
- Improve accuracy for minority class (e.g., trash)
- Add webcam input to Gradio
- Try TFLite export for mobile deployment
- UI enhancements and visual feedback in Gradio

