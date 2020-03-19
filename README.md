# ASL_Translation
## Introduction
This project aims at bridging this gap between the hearing impaired and the common population
by providing a method to translate the American Sign Language (ASL) to English. By observing
the hands and gestures used in sign language the project identifies each sign and displays the
translation in English text.

## Methodology
 The project comprises of 3 main phases:
### 1. Skin Pixel Segmentation
Create a Region Of Interest (ROI) from the entire camera view to obtain just the region
with the hand in the frame. Threshold the ROI to segment the image in order to obtain just
the skin pixels in white and the rest of the image in black.
### 2. Data Generation
Once the segmented ROI is obtained data required for training the model is generated. Data
is divided into train, validation and test portions which can be used by the model to train
and validate its learning.
### 3. Model Generation
A Convoluted Neural Network (CNN) model is generated with network architecture as
specified in “Using Deep Convolutional Networks for Gesture Recognition in American
Sign Language” by Vivek Bheda and N. Dianna Radpour. State University of New York
at Buffalo. 


