# Age Prediction Model

This repository contains the code for an age prediction model trained on a dataset of face images. The model uses a pre-trained ResNet-18 architecture fine-tuned for age regression. The goal is to predict a person's age as accurately as possible given a photo of their face.

## Project Structure

- `notebookf0fbadb03b.py`: Main Python script that includes dataset preparation, model definition, training, and evaluation.
- `train.csv`: The CSV file containing the training data.
- `test.csv`: The CSV file for testing the model and generating submissions.
- `model.pth`: Saved model weights after training (not included in the repository, generated after training).

## Dataset

The dataset consists of images of human faces along with their corresponding age. Each entry in the `train.csv` and `test.csv` files points to an image file and its associated age.

## Model

The age prediction model is built on the ResNet-18 architecture, a convolutional neural network that is 18 layers deep. The model is pre-trained on ImageNet and fine-tuned on our specific age prediction task. The final layer is a fully connected layer with a single output, corresponding to the predicted age.

## Features

- **Data Augmentation**: To increase the diversity of the training data and improve generalization, data augmentation techniques such as random horizontal flipping, random rotation, and color jittering are used during training.
- **Batch Normalization and Dropout**: Included in the fully connected layer to reduce overfitting and improve convergence.
- **Learning Rate Scheduling**: The learning rate is scheduled to decrease over epochs to allow for finer adjustments to the model weights as training progresses.
- **Hyperparameter Tuning**: Basic grid search over learning rates and batch sizes to find the best combination for training.
- **Model Checkpointing**: The best model state is saved based on the lowest training loss, allowing for recovery and further use.

## Setup Instructions

To run the code, follow these steps:

1. Ensure you have Python 3.x installed along with the necessary libraries: `torch`, `torchvision`, `pandas`, `numpy`, and `PIL`.
2. Place the `train.csv` and `test.csv` in the same directory as the `notebookf0fbadb03b.py` script or update the paths in the script accordingly.
3. Run the script using the command: `python notebookf0fbadb03b.py`.

## Usage

Once you have trained the model, you can use it to predict ages on new images by running the `predict` function defined in the script.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset used for this project was provided by [Kaggle](https://www.kaggle.com).
- The model is built using the PyTorch library, which provides the pre-trained ResNet-18 architecture.
