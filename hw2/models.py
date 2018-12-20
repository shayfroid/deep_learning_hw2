import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is:

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.

    If dropout is used, a dropout layer is added after every ReLU.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param: Dropout probability. Zero means no dropout.
        """
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.dropout = dropout

        blocks = []

        # ====== YOUR CODE: ======

        blocks.append(Linear(in_features, hidden_features[0] if hidden_features else num_classes))

        features = [feature for feature in hidden_features] + [num_classes]
        for i in range(len(features) - 1):
            # TODO fix error
            blocks.append(ReLU())

            if dropout:
                blocks.append(Dropout(dropout))

            blocks.append(Linear(features[i], features[i+1]))

        # ========================
        # print("\nlast block type: ", type(blocks[-1]))
        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        conv_num = 0
        for i in range(int(len(self.filters)/self.pool_every)):
            for j in range(self.pool_every):
                layers.append(torch.nn.Conv2d(in_channels, self.filters[conv_num], 3, stride=1, padding=1))
                in_channels = self.filters[conv_num]
                conv_num += 1
                layers.append(torch.nn.ReLU())
            layers.append(torch.nn.MaxPool2d((2, 2), dilation=1))
            in_h = int(in_h/2)
            in_w = int(in_w/2)
        self.in_size = (in_channels, in_h, in_w)

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        print(self.in_size)
        in_features = in_h*in_channels*in_w
        for hd in self.hidden_dims:
             layers.append(torch.nn.Linear(in_features,hd))
             in_features = hd
             layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(in_features, self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        fe = self.feature_extractor(x)
        fe = fe.view(fe.size(0), -1)
        out = self.classifier(fe)
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)
        
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # ====== YOUR CODE: ======
        conv_num = 0
        for i in range(int(len(self.filters)/self.pool_every)):
            for j in range(self.pool_every):
                layers.append(torch.nn.Conv2d(in_channels, self.filters[conv_num], 3, stride=1, padding=1))
                in_channels = self.filters[conv_num]
                conv_num += 1
                layers.append(torch.nn.ReLU())
            layers.append(torch.nn.FractionalMaxPool2d(3, output_ratio=0.70711))
        layers.append(torch.nn.Conv2d(in_channels, in_channels, 4))
        layers.append(torch.nn.ReLU())                  
        layers.append(torch.nn.Conv2d(in_channels, in_channels, 8))     
        layers.append(torch.nn.ReLU())                  
        layers.append(torch.nn.Conv2d(in_channels, 10, 1))

        # ========================
        seq = nn.Sequential(*layers)
        return seq
                          
    def _make_classifier(self):
        return None
                          
    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        fe = self.feature_extractor(x)
        print(fe.shape)
        #out = fe.view(fe.size(0), -1)
        #print(out.shape)
        # ========================
        return out
