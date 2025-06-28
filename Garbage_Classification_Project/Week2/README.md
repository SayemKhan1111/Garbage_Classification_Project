# Garbage Classification Project

## Objective
Classify garbage images into 6 categories: cardboard, glass, metal, paper, plastic, trash using deep learning.

## Dataset
- Source: [Kaggle - Trash Type Image Dataset](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)
- Total Images: ~2400
- Format: Folder-wise (one per class)

## Tools Used
- Python
- TensorFlow / Keras
- EfficientNetV2B2 (Transfer Learning)
- Scikit-learn
- Gradio
- Hugging Face Spaces

---

## ✅ Week 1 Progress (30%)
- TensorFlow environment setup
- Dataset downloaded and structured
- Test script created using MobileNetV2
- Model trained and saved as `garbage_classifier.h5`

---

## ✅ Week 2 Progress (60%)
- Switched to EfficientNetV2B2 for improved accuracy
- Used data augmentation & class weights to handle imbalance
- Trained model with early stopping and learning rate scheduler
- Evaluated using confusion matrix and classification report
- Model saved as `efficientnetv2b2_model.keras`
- Built Gradio interface for image upload & prediction
- Deployed app on Hugging Face:  
  👉 [Live Demo](https://huggingface.co/spaces/Sayemkhan1111/sayem-garbage-classifier)

---

## Next Steps (Week 3 Ideas)
- Fine-tune deeper layers of EfficientNetV2B2
- Add real-time webcam classifier in Gradio
- Improve UI and styling
- Track metrics using TensorBoard
- Convert to TFLite / deploy to mobile (optional)
