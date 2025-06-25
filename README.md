# Detection of Non-small Cell Lung Cancer using Histopathological Images by Deep Learning

## Abstract


This research presents a comprehensive approach to detect and classify Non-small Cell Lung Cancer (NSCLC) using histopathological images and deep learning techniques. The study compares multiple state-of-the-art CNN architectures for binary classification (benign vs malignant) and explores cancer grading using clustering algorithms.

## Key Features

- **Multi-model Comparison**: Evaluates 6 different deep learning architectures
- **High Accuracy**: Achieves up to 99.07% validation accuracy with EfficientNetB2
- **Cancer Grading**: Implements K-means clustering for cancer stage classification
- **Comprehensive Dataset**: Uses LC25000 histopathological image dataset
- **Transfer Learning**: Leverages pre-trained models for improved performance

## Dataset

**LC25000 Dataset**
- 25,000 histopathological images
- 5 classes total (750 lung tissue images)
- Categories: Benign, Adenocarcinoma, Squamous Cell Carcinoma
- Image resolution: 768×768 pixels
- Source: Kaggle

## Models Implemented

### 1. **EfficientNetB2** (Best Performance)
- **Validation Accuracy**: 99.07%
- **Parameters**: 36,998,396 total (29,229,827 trainable)
- **Input Size**: 260×260 pixels
- **Key Features**: Compound scaling, Mobile Inverted Bottleneck Convolution

### 2. **VGG19**
- **Test Accuracy**: 96%
- **Parameters**: 20,158,531 total (20,157,507 trainable)
- **Input Size**: 224×224 pixels
- **Architecture**: 19 layers with 3×3 convolutions

### 3. **Inception V3**
- **Test Accuracy**: 89.956%
- **Parameters**: 21,802,784 total (21,768,352 trainable)
- **Input Size**: 299×299 pixels
- **Key Features**: Multiple inception modules, factorized convolutions

### 4. **ResNet50**
- **Parameters**: 23,593,859 total (6,147 trainable)
- **Architecture**: 50 layers with residual connections
- **Key Features**: Skip connections, batch normalization

### 5. **DenseNet169**
- **Architecture**: 169 layers with dense connections
- **Key Features**: Feature reuse, gradient flow improvement

### 6. **Custom 7-Layer CNN**
- **Architecture**: 7 convolutional layers with max pooling
- **Purpose**: Baseline comparison model

## Cancer Grading Approach

### K-Means Clustering Method
1. **Feature Extraction**: Uses pre-trained VGG16 for feature extraction
2. **Dimensionality Reduction**: PCA to reduce features to 2 principal components
3. **Clustering**: K-means algorithm groups similar cancer patterns
4. **Visualization**: Scatter plots show cancer grade clusters

## Requirements

### Software Requirements
- Google Colab
- Python 3.0+
- TensorFlow/Keras
- OpenCV
- scikit-learn
- matplotlib
- numpy
- pandas

### Hardware Requirements
- GPU (recommended)
- Minimum 4 GB RAM
- Windows/Linux/MacOS

# Results Summary

| Model           | Test Accuracy | Precision | Recall | F1-Score |
|----------------|---------------|-----------|--------|----------|
| EfficientNetB2 | **99.07%**    | 99.1%     | 99.0%  | 99.0%    |
| VGG19           | 96.0%         | 96.2%     | 96.1%  | 96.1%    |
| ResNet50        | 94.5%         | 94.8%     | 94.3%  | 94.5%    |
| Inception V3    | 89.96%        | 89.96%    | 90.14% | 90.02%   |
| DenseNet169     | 92.3%         | 92.5%     | 92.1%  | 92.3%    |
| Custom CNN      | 88.2%         | 88.5%     | 88.0%  | 88.2%    |

---

##  Key Findings

-  **EfficientNetB2** achieves the highest accuracy (**99.07%**) with fewer parameters.
-  Histopathological images offer better diagnostic performance than X-ray or CT scans.
-  **Transfer learning** substantially boosts classification accuracy.
-  **K-means clustering** shows potential for automated grading of cancer severity.

---

##  Clinical Significance

- **Pathologist Support**: Assists medical professionals, reducing manual workload.
- **Rural Healthcare**: Enables cancer screening in underserved regions.
- **Early Detection**: Facilitates quicker diagnoses and treatment decisions.
- **Standardization**: Promotes consistent classification across clinics and hospitals.

---

##  Future Enhancements

-  **Automated Grading**: Add supervised models to determine cancer stage or grade.
-  **Multi-Dataset Training**: Incorporate TCGA or similar datasets for generalization.
-  **Real-Time Processing**: Optimize pipeline for faster clinical deployment.
-  **Explainable AI**: Visualize model decisions to aid interpretability.

---

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## Authors

- **Dhurka Prasanna P** – dhurkaprasanna@gmail.com  
- **Janima K Radhakrishnan** – janimakr580@gmail.com  
- **Kurapati Sreenivas Aravind** – arvind.kurapati@gmail.com  
- **Pranav R Nambiar** – pranavradhesh@gmail.com  
- **Nalini Sampath** – s_nalini@blr.amrita.edu  

**Department of Computer Science & Engineering**  
Amrita School of Engineering, Bangalore  
Amrita Vishwa Vidyapeetham, India

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{prasanna2022detection,
  title={Detection of Non-small cell Lung Cancer using Histopathological Images by the approach of Deep Learning},
  author={Prasanna, Dhurka P and Radhakrishnan, Janima K and Aravind, Kurapati Sreenivas and Nambiar, Pranav R and Sampath, Nalini},
  booktitle={2022 2nd International Conference on Intelligent Technologies (CONIT)},
  pages={1--11},
  year={2022},
  organization={IEEE}
}
