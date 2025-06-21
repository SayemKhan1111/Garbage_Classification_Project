# Garbage Classification Project

## Objective
Classify garbage images into 6 categories: cardboard, glass, metal, paper, plastic, trash.

## Dataset
- Source: [Kaggle - Trash Type Image Dataset](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)
- Total Images: ~2400
- Format: Folder-wise (one per class)

## Tools Used
- Python
- TensorFlow / Keras
- MobileNetV2 (Pretrained)

## Week 1 Progress (30%)
-TensorFlow environment setup
-Dataset downloaded and structured
-Test script confirmed working
-Model trained with MobileNetV2
-Model saved as `garbage_classifier.h5`

## Next Steps
- Improve accuracy (more epochs, augmentations)
- Try ResNet / EfficientNet
- Build simple web app for predictions
