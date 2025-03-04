# **Healthy vs. Unhealthy Food Classification**  
A Machine Learning Pipeline for Image-Based Food Classification  

## **Project Overview**  
This project explores whether my food choices are healthy or unhealthy using machine learning techniques. I built an image classification pipeline, applying multiple models from logistic regression to convolutional neural networks (CNNs) and transfer learning.  

## **Dataset**  
- **Initial Dataset**: 60 manually labeled images (30 healthy, 30 unhealthy)  
- **Expanded Dataset**: Augmented dataset including Food-11 images (1,244 per class)  
- **Preprocessing**: Image resizing (160×160), normalization, and data augmentation (rotation, flipping, cropping, color adjustments)  

## **Models & Methodologies**  
### **Traditional Models**  
- **Logistic Regression**: Baseline model, achieved **56% accuracy**  
- **Support Vector Machines (SVM)**: Experimented with different kernels (linear, RBF, polynomial), best accuracy **61%**  

### **Deep Learning Models**  
- **CNN**: Designed a simple convolutional neural network, achieved **71% accuracy**  
- **Transfer Learning (MobileNetV2)**:  
  - Feature extraction: **83% accuracy**  
  - Fine-tuned model: **88% accuracy**  

## **Key Findings**  
- Traditional models struggled with image classification due to lack of spatial feature extraction.  
- CNNs significantly improved performance by learning hierarchical patterns.  
- Transfer learning with MobileNetV2 yielded the best results, demonstrating the power of pre-trained models for small datasets.  

## **Future Improvements**  
- Implement **Generative Adversarial Networks (GANs)** for synthetic data generation.  
- Experiment with **Vision Transformers (ViT)** for improved classification accuracy.  

## **Technologies Used**  
- **Python, TensorFlow, Keras, Scikit-Learn, OpenCV, NumPy, Matplotlib**  
- **Jupyter Notebook & Google Colab**  

## **How to Use**  
1. Clone the repository  
2. Install dependencies (`pip install -r requirements.txt`)  
3. Run `cs156-pipeline2.ipynb` to train and evaluate models  
