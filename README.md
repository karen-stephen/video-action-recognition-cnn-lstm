# CNN + LSTM Video Action Recognition in PyTorch

End-to-end video action recognition pipeline implemented using PyTorch.

## Architecture
Video → Frame Sampling → ResNet18 Feature Extraction → LSTM Temporal Modeling → Classification

## Steps
- Implemented custom PyTorch VideoDataset using OpenCV
- Sampled frames from videos and converted them to tensors
- Used pretrained ResNet18 for frame-level feature extraction
- Used LSTM for temporal sequence modeling
- Built custom VideoModel using nn.Module
- Implemented DataLoader batching and training loop
- Verified end-to-end training by overfitting on a tiny dataset

## Result
- Successfully trained CNN + LSTM pipeline end-to-end
- Verified correct predictions on tiny sample dataset

## Tech Stack
- Python
- PyTorch
- torchvision
- OpenCV
- NumPy
