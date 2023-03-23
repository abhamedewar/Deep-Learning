# Transfer Learning

#### Transfer Learning:
Transfer learning is a technique used in machine learning and artificial intelligence to reuse a pre-trained model to solve a different but related problem. There are several types of transfer learning:

#### Types of transfer learning:
1. Finetuning
2. Feature Extraction

#### Finetuning:
Finetuning is a type of transfer learning in which a pre-trained model is further trained on a new dataset for a specific task. The pre-trained model is typically a deep neural network that has been trained on a large dataset, such as ImageNet or COCO, for a related task, such as image classification or object detection. Finetuning involves taking the pre-trained model and training it on a smaller dataset that is specific to a new task. The training process in finetuning involves training all the model parameters.

#### Feature Extraction:
Feature extraction is a widely used transfer learning technique where a pre-trained model is employed to extract features from data. The pre-trained model is typically a deep neural network that has been trained on a large dataset for a related task. In this approach, the pre-trained model is utilized, and only the last layer of the network is modified, and its weights are updated, while keeping the other layers frozen. This is particularly useful when the new task has insufficient training data. 

#### Data Used:

The data that will be used for performing fine tunning and feature extraction will be image data of natural scenes around the world. The data contains around 25k images of size 150 x 150 distributed amoung 6 categories.

The six categories are buildings, forest, glacier, mountain, sea and street. The data is divided into training and test split. The training set consists of 14k images and the test set consists of 3k images.

#### Models used:

The models that will be used are: 
1. Resnet
2. Alexnet
3. Vgg
4. Squeezenet
5. Densenet
6. Inception







