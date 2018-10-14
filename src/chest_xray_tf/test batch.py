import numpy as np
import matplotlib.pyplot as plt

test = np.load('/Users/frank/Documents/GitHub/alexnet-sbs/dataSet/parsed_train/batch0.npy')
img = test[0][1]
plt.imshow(img)
plt.show()