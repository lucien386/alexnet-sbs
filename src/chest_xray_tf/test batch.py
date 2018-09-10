import numpy as np
import matplotlib.pyplot as plt

test = np.load('/Users/frank/Documents/Github/alexnet-sbs/dataSet/train/normal_batch/batch0.npy')
im = test[0].reshape([224, 224])
print(max(im.flatten()))
plt.imshow(im, cmap="gray")
plt.show()