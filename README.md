# Face Verification using Deep Learning

A deep learning-based face verification system that determines whether two face images belong to the same person. The project uses a Siamese network architecture with ResNet50 backbone and contrastive loss for training on the LFW (Labeled Faces in the Wild) dataset.

##  Project Overview

This project implements a face verification system that:
- Takes two face images as input
- Generates 128-dimensional embeddings for each face
- Computes the Euclidean distance between embeddings
- Determines if the faces belong to the same person based on a learned threshold

##  Features

- **Siamese Network Architecture**: Uses twin networks with shared weights to learn face embeddings
- **Transfer Learning**: Leverages pre-trained ResNet50 for feature extraction
- **Contrastive Loss**: Optimizes the distance between matching and non-matching face pairs
- **High Accuracy**: Achieves 80.80% accuracy on the LFW test dataset
- **Efficient Training**: Fine-tunes only the last two layers of ResNet50 for faster convergence

##  Dataset

The project uses the **LFW (Labeled Faces in the Wild)** dataset:
- **Training pairs**: `matchpairsDevTrain.csv` and `mismatchpairsDevTrain.csv`
- **Test pairs**: `matchpairsDevTest.csv` and `mismatchpairsDevTest.csv`
- **Images**: High-quality face images from `lfw-deepfunneled` directory
- **Classes**: Binary classification (1 = same person, 0 = different people)

##  Model Architecture

### Backbone Network
- **Base Model**: ResNet50 pre-trained on ImageNet
- **Frozen Layers**: Early layers (layer1, layer2) frozen to retain low-level features
- **Trainable Layers**: layer3 and layer4 fine-tuned for face-specific features

### Embedding Network
```python
Fully Connected Layers:
├── Linear(2048 → 512) + ReLU
├── Linear(512 → 256) + ReLU
└── Linear(256 → 128)  # Final embedding
```

### Loss Function
- **Contrastive Loss** with margin = 1.5
- Minimizes distance for matching pairs
- Maximizes distance (up to margin) for non-matching pairs

##  Getting Started

### Prerequisites
```bash
torch>=1.9.0
torchvision>=0.10.0
opencv-python
pandas
matplotlib
numpy
```

### Installation
```bash
git clone https://github.com/Damanish/few_shot_face_verification.git
cd face-verification-pytorch
pip install -r requirements.txt
```

### Dataset Setup
1. Download the LFW dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
2. Extract to your desired directory
3. Update the `lfw_dir` path in the notebook

##  Usage

### Training
```python
# Load and preprocess data
python train_model.py

# Or run the Jupyter notebook
jupyter notebook FaceVerification.ipynb
```

### Inference with Pre-trained Model
```python
import torch
from model import FaceVerificationModel

# Load the trained model
model = FaceVerificationModel()
model.load_state_dict(torch.load('face_verification_model.pth'))
model.eval()

# Verify two faces
similarity_score = verify_faces(image1_path, image2_path, model)
```

### Evaluation
The model uses a threshold of **0.77** for classification:
- Distance < 0.77 → Same person (Match)
- Distance ≥ 0.77 → Different people (Mismatch)

##  Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 80.80% |
| **Training Epochs** | 7 |
| **Final Loss** | 0.0301 |
| **Embedding Dimension** | 128 |
| **Decision Threshold** | 0.77 |

### Distance Distribution
The model learns to separate matching and non-matching pairs effectively, as shown in the distance distribution histogram in the notebook.

##  File Structure
```
face-verification-pytorch/
├── FaceVerification.ipynb          # Main training and evaluation notebook
├── face_verification_model.pth     # Pre-trained model checkpoint
├── README.md                       # Project documentation

```

##  Model Configuration

### Training Hyperparameters
- **Learning Rate**: 5e-5
- **Weight Decay**: 0.005
- **Batch Size**: 32
- **Optimizer**: Adam
- **Image Size**: 224×224
- **Normalization**: ImageNet statistics

### Data Augmentation
- Resize to 224×224
- Convert to tensor
- Normalize with ImageNet mean/std

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **LFW Dataset**: University of Massachusetts, Amherst
- **ResNet Architecture**: Microsoft Research
- **PyTorch Team**: For the excellent deep learning framework

## 📞 Contact

For questions or suggestions, please open an issue or contact [rdsbankawat@gmail.com](mailto:rdsbankawat@gmail.com).

---

**Note**: Make sure to download the LFW dataset separately and update the paths in the configuration before running the code.
