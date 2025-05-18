# 🧠 FusionVision: Hybrid Vessel Classification with DINOv2 and Shape-Aware Feature Fusion

FusionVision is a research-grade, hybrid image classification system that enhances fine-grained **vessel recognition** by combining **deep learning embeddings from DINOv2** with **traditional shape and texture features** using OpenCV. The model is designed for imbalanced datasets with small sample sizes and high inter-class similarity—such as those used in kitchen prep or waste classification pipelines.

---

## 🚀 Introduction

This project introduces a **dual-branch classifier** that processes both:
- Visual embeddings from **DINOv2 (ViT-L/14)**
- Shape-aware descriptors including:
  - Canny + Sobel edge detection
  - HOG features
  - Symmetry analysis
  - GLCM-based texture metrics
  - Contour statistics & Hu moments

The fusion of **learned** and **handcrafted** features enhances both accuracy and interpretability, particularly in **low-data regimes** with over **80–90 vessel classes**.

---

## 🧩 Architecture Summary

### 🔀 DualClassifier

- **Visual Feature Branch**:
  - Uses pretrained DINOv2 embeddings
  - Passes through EnhancedResidualBlocks
  - Includes dropout, batch norm, and linear layers

- **Shape Feature Branch**:
  - Input includes Hu moments, contour stats, edge scores, symmetry metrics
  - Processed through residual blocks with squeeze-excitation

- **Fusion Layer**:
  - **Multi-head cross-attention (8 heads)**
  - Residual connections & layer normalization
  - Feature concatenation → final projection

- **Output**:
  - Single classification head for **vessel class prediction**
  - Top-3 predictions returned with associated confidence scores

---

## 🧠 EnhancedResidualBlock

- Designed to improve gradient flow and multiscale representation
- Supports:
  - Multi-branch structure
  - Squeeze-and-Excitation modules
  - Optional LayerNorm
  - ReLU → BatchNorm → Dropout

---

## 🧪 Feature Extraction Pipeline

Implemented in `folder_test.py` and `model_test.py`, the shape descriptor pipeline includes:

### 🔷 Preprocessing

- **CLAHE** for contrast enhancement
- **Canny + Sobel** edge maps
- **Contour detection** using OpenCV

### 🔶 Shape Descriptors

- Hu Moments
- Perimeter / Area ratios
- Compactness
- Edge density
- Normalized height/width
- Symmetry scores (flipped image comparison)

### 🔶 Texture Features

- **GLCM-based metrics**: Contrast, Correlation, Dissimilarity, Homogeneity
- **HOG features** with cell sizes: 8×8, 16×16

### 🔷 Multi-Scale Support

Images processed at:
- 0.5x scale
- 1.0x scale
- 2.0x scale

---

## 📈 Training Curves

### 🔁 Training Run 1 (Baseline)
- 30 epochs
- No fusion, CLIP only
- Accuracy capped at 60–70%


### ✅ Training Run 2 (Hybrid Model)
- 25 epochs
- Uses full DINOv2 + Shape fusion
- Achieved ~95–100% validation accuracy


---

## 🛠️ Training Configuration

**Script**: `model_training.py`

- epochs = 25
- batch_size = 32
- dropout_rate = 0.25
- optimizer = Adam(lr=0.001)
- loss_fn = CrossEntropyLoss()
- augmentations = HorizontalFlip(), RandomBrightnessContrast(), CLAHE()

## 🔧 Improvements Over Time
- ✅ **Added dropout & batch normalization** for regularization and stable convergence  
- ✅ **Introduced better skip connections** for gradient flow  
- ✅ **Balanced batches by class** to address data imbalance  
- ✅ **Implemented multi-head cross-attention** in the fusion layer  
- ✅ **Added deeper classification heads** for more representational power  

---

## 🔍 Inference Pipeline

The inference logic is encapsulated in `model_test.py` and `folder_test.py`.

### Steps:
1. Load pre-trained model weights and label encoder  
2. Preprocess image with **CLAHE**, **Canny**, **Sobel**  
3. Extract **DINOv2 features** and **handcrafted shape features**  
4. Fuse features and perform inference  

### Output Includes:
- Predicted class  
- Top-3 predictions with confidence  
- Annotated visualization  
- CSV results file  

---

## 📊 Results Snapshot

| Model Version       | Validation Accuracy | Top-3 Accuracy | Inference Speed |
|---------------------|---------------------|----------------|-----------------|
| CLIP-only           | 68%                 | 80%            | Fast            |
| DINOv2 + Shape      | 97.8%               | 99.5%          | Moderate        |

> The hybrid approach provides a dramatic improvement in precision and confidence compared to unimodal models.

---

## 🔗 Primary Libraries Used

- torch, torchvision: for model training and architecture
- timm: for loading DINOv2 vision transformer
- opencv-python: for shape and contour-based feature extraction
- albumentations: for data augmentation
- scikit-image, scikit-learn: for HOG, GLCM, and evaluation
- matplotlib, tqdm: for progress monitoring and plotting

  ---

## ✅ Summary of Key Advances

This project makes several architectural and practical advancements in fine-grained vessel classification:
- Combines deep learning and handcrafted features for hybrid modeling
- Introduces shape-aware processing alongside transformer embeddings
- Integrates squeeze-excitation, residual blocks, and cross-attention
- Supports multi-scale and multi-channel inputs
- Enables confidence-based predictions and interpretability

  ---

## 🧠 Model Improvements
- Replaced CLIP ViT with DINOv2 for better semantic visual representation
- Removed ingredient classification for focused vessel modeling
- Upgraded fusion layer with multi-head attention (8 heads)
- Enhanced residual blocks with Squeeze-and-Excitation modules
- Added layer normalization to stabilize training

  ---

## 📈 Training Upgrades
- Adopted early stopping and patience scheduling
- Improved augmentation with CLAHE, flipping, cropping
- Balanced sampling per class to fight dataset skew
- Augmented from 5 to 10 views per sample
- Increased model depth for improved representation

---

