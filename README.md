# Real vs. Fake Face Detection with Inception ResNet V1

This project leverages the power of deep learning to differentiate between real and fake face images. An Inception ResNet V1 model was trained on a dataset of 140,000 images, equally divided between real and fake faces, to achieve high accuracy in identifying fraudulent attempts at digital impersonation.

## Project Overview

The increasing sophistication of digital impersonation techniques poses significant security risks. This project aims to bolster the reliability of facial recognition systems by developing a model capable of distinguishing real faces from fakes with high precision.

## Dataset

The model was trained on a dataset consisting of 140,000 images, split evenly between real and fake faces. This dataset was specifically chosen for its balance and variety, providing a comprehensive foundation for training.

- **Dataset Link**: [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

## Model Details

- **Architecture**: Inception ResNet V1
- **Training Platform**: [Kaggle Notebook](https://www.kaggle.com/code/mahendranb7/deep-fake-training/notebook)
- **Training Data**: 70,000 real images and 70,000 fake images
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC-AUC

## Training Results

### Loss Function Graph

![Loss Function Over Epochs](https://github.com/mahe115/Deepfake-image-classification/blob/4d0e376897544fcd5142ed79f454319aab15a4a8/loss%20function%20graph.png)

### Accuracy Graph

![Accuracy Over Epochs](https://github.com/mahe115/Deepfake-image-classification/blob/ec1ec2fa74d364591687ba28fa9d7ef49541d131/accuracy%20graph.png)


## Installation

Before running this project, ensure Python 3.x is installed. Then, install the required dependencies:


## Usage

To replicate the training process or to train the model with new data, follow the instructions provided in the Kaggle notebook:

- [Training Notebook on Kaggle](https://www.kaggle.com/code/mahendranb7/deep-fake-training/notebook)

## Evaluation and Results

The model's performance is thoroughly evaluated using multiple metrics. The results highlight the model's capability to accurately classify real and fake faces. Detailed results and analysis are available in the training notebook.

## Ethical Considerations

We are committed to ethical AI practices. This project addresses potential biases in facial recognition technology and ensures diverse representation within the training dataset to mitigate discriminatory outcomes.

## Future Work

Future enhancements may include:
- Data augmentation to increase model robustness.
- Implementing ensemble learning for improved accuracy.
- Continuous updates to the model to counter new impersonation techniques.


## Acknowledgments

- Special thanks to Kaggle for hosting both the dataset and the platform for training.
- Appreciation to all contributors to the Inception ResNet V1 architecture and related technologies.
