# alexnet-sbs

# Authors
A group of ML fan students @ The Stony Brook School.

# Description
An implementation of the Alexnet. Fed with chest X-ray pictures, it will predict whether the patient has pneumonia or not.

# Tasks
Preprocess data & determine final data size (512x512 at the moment)(Frank)
Define layers(Tom, Lucien)
Define forward prop(?)
Define backprop(?)
Define cost function(?)
...(Add more here)

# Current data distribution
| #data |    NORMAL    |  PNEUMONIA   |
|-------|--------------|--------------|
| TRAIN | 1341 (84.7%) | 3875 (90.7%) |
| TEST  |  234 (14.8%) |  390 ( 9.1%) |
|  VAL  |    8 ( 0.5%) |    8 ( 0.2%) |
| TOTAL | 1583 (100.%) | 4273 (100.%) |

Current data distribution biased.

# Network definition

Frank: DATA PREPROCESSING
- convert raw data into batches of 4D numpy arrays with size [batch, h, w, 1]

Tom & Lucien : NETWORK BUILDING
- Output one-hot list of labels for data: [normal, bacteria, virus]
