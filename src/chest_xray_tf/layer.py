from layer_hf import convolutional_layer
from layer_hf import normal_full_layer

#X is a reshaped 4D tensor with size [batch (param), height (param), width (param), channel = 1]
#Output one-hot label: list (norma, bacteria, virus)
def sbs_alex_net(X, W, b, dropout):
	conv1 = convolutional_layer(X, shape = [])
