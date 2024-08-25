# Bird Species Classification - Predictive Modeling Competition 🐣


Welcome to our group project for the predictive modeling competition! We’re using the Bird Species dataset from Kaggle to build a Convolutional Neural Network (CNN) that can classify 15 bird species commonly found in Quebec. The goal is to train, test, and validate our model, with the validation set reserved for the final evaluation.


# Project Overview

In this project, we’re focused on training a CNN to recognize different bird species from the provided images. We took special care to ensure that the training, testing, and validation sets all contain the same bird species, which helps maintain consistency across the model's evaluations. The validation set is reserved for the end of the process, providing a clear and unbiased measure of the model's performance.

Our approach involved optimizing the model with techniques like data augmentation and image resizing. These steps were crucial in both improving the model's accuracy and speeding up the training process.

# Installation

**Clone the repository:**
- git clone (https://github.com/fbrossard04/Bird-Classification-Project)
  
**Navigate to the project directory:**
- cd bird-classification
  
**Install the required packages:**
- pip install -r requirements.txt

# Dataset

This project was developed using the dataset:
https://www.kaggle.com/datasets/gpiosenka/100-bird-species

The dataset features images of 15 bird species from Quebec. We’ve split the data into training, testing, and validation sets, making sure each set has the same species.

File Structure:
birds.csv: Contains file paths and labels for the images.

Images are organized in directories by species.

# Model Architecture

For this project, we built on the EfficientNetB0 architecture, which is pre-trained on the ImageNet dataset. This model provides a strong foundation due to its efficiency and performance on a wide range of tasks.

To adapt the model for our specific needs, we added a few custom layers:

**Global Average Pooling:** This layer reduces the dimensionality of the model’s output, making it more manageable.

**Dense Layer:** A fully connected layer with ReLU activation to introduce non-linearity and learn complex patterns in the data.

**Dropout Layer:** This layer helps prevent overfitting by randomly dropping units during training.

**Output Layer:** The final layer uses softmax activation to produce probabilities for each of the 15 bird species.

# Training

During training, we used data augmentation to artificially increase the diversity of our training data. This involved applying random transformations like rotation, zoom, and flipping to the images, which helps the model generalize better.

We also resized the images to reduce their dimensions, which not only speeds up the training process but also helps the model learn more efficiently. Google Colab was utilized to leverage its powerful GPU resources, enabling faster training times.

Our training process was divided into two main steps:

**Initial Training:** We first trained the top layers of the model while keeping the pre-trained base model frozen. This allowed us to focus on the new layers we added.

**Fine-Tuning:** After the initial training, we unfroze the base model and fine-tuned the entire model to improve accuracy further.
Evaluation

Once the training was complete, we evaluated the model's performance using the reserved validation set. This step was crucial for assessing how well the model generalizes to new, unseen data. By using the validation set only at the end, we ensured that our performance metrics were reliable and reflected the model's true capabilities.

# Prediction
With the trained model, we can now predict the species of a given bird image. The model processes the image and outputs the most likely species label. Additionally, we developed functionality to compare the predicted classes with the actual labels in the validation set, providing a detailed accuracy report.

# Results
In the final results, we present the model's accuracy and loss, showing how these metrics evolved during training and validation. We also include a classification report that breaks down the model’s performance by species, giving a clear picture of where the model excels and where it might need improvement.
