import torch
import torch.nn as nn


class HeightMaxPool2d(nn.Module):
    """Max pooling only along height axis."""
    
    def __init__(self, size=(2, 1), stride=(2, 1)):
        """
        Constructor.
        Size and stride are as per given in paper.
        """
        super().__init__()
        self.max_pooling = nn.MaxPool2d(kernel_size=size, stride=stride)
    
    def forward(self, x):
        """Forward pass."""
        return self.max_pooling(x)


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM which will be fed with RoIRotated shared features"""

    def __init__(self, input_size, hidden_size, n_classes):
        """Constructor."""
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,  # Same as paper
            bidirectional=True  # Paper uses bidirectional LSTM
        )

        self.fc = nn.Linear(
            # Multiplied by 2 because LSTM is bidirectional
            in_features=hidden_size * 2,
            out_features=n_classes
        )
        
    
    def forward(self, x, lengths):
        """Forward pass."""
        # compacts all weights into a contiguous chuck of memory
        # Reference: https://stackoverflow.com/q/53231571/5353128
        self.lstm.flatten_parameters()
        total_length = x.size(1)

        # To optimize the LSTM computations.
        # Reference: https://stackoverflow.com/q/51030782/5353128
        # https://github.com/pytorch/pytorch/issues/43227
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # output shape: (T, b, h * 2) [seq_len, batch_size, num_directions*hidden_size]
        output, _ = self.lstm(packed_input)
        padded_input, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=total_length, batch_first=True)

        b, T, h = padded_input.size()
        t_rec = padded_input.reshape(T * b, h)
        output = self.fc(t_rec)  # (T * b, n_classes)
        output = output.reshape(b, T, -1)
        output = nn.functional.log_softmax(output, dim=-1)  # for CTC loss

        return output


class CRNN(nn.Module):
    """
    CRNN is Convolutional Recurrent Neural Network. CRNN is typically used for
    text recognition part of OCR. In FOTS also, CRNN is used in text
    recognition branch. CRNN model uses a convolutional neural network (CNN)
    to extract visual features, which are reshaped and fed to a long short term
    memory network (LSTM). The output of the LSTM is then mapped to character
    labels space with a Dense/fully connected layer.

    To train CRNN module, CTC (Connectionist Temporal Classification) loss is used.
    More on CTC loss is in it's own implementation.

    Structure of text recognition branch (CRNN) is given in paper.

    Type                Kernel              Out
                        [size, stride]      Channels
    ---------------------------------------------------
    conv_bn_relu        [3, 1]              64
    ---------------------------------------------------
    conv_bn_relu        [3, 1]              64
    ---------------------------------------------------
    height-max-pool     [(2, 1), (2, 1)]    64
    ---------------------------------------------------
    conv_bn_relu        [3, 1]              128
    ---------------------------------------------------
    conv_bn_relu        [3, 1]              128
    ---------------------------------------------------
    height-max-pool     [(2, 1), (2, 1)]    128
    ---------------------------------------------------
    conv_bn_relu        [3, 1]              256
    ---------------------------------------------------
    conv_bn_relu        [3, 1]              256
    ---------------------------------------------------
    height-max-pool     [(2, 1), (2, 1)]    256
    ---------------------------------------------------
    bi-directional      lstm                256
    ---------------------------------------------------
    fully-connected                     |S| = n_classes
    ---------------------------------------------------
    """

    def __init__(self, feature_height, n_channels, n_classes, hidden_size):
        """
        Constructor.

        The feature height is kept constant (=8) in the paper. It is height
        of text detection features which are RoIRotated. Based on the aspect
        ration, it's width will vary.
        """
        super().__init__()
        
        # Based on the given settings, build CNN model
        self.cnn = nn.Sequential(
            # conv_bn_relu
            nn.Conv2d(
                in_channels=n_channels, out_channels=64, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            # conv_bn_relu
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            # height-max-pool 
            HeightMaxPool2d(),

            # conv_bn_relu
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            # conv_bn_relu
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            # height-max-pool 
            HeightMaxPool2d(),

            # conv_bn_relu
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # conv_bn_relu
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # height-max-pool 
            HeightMaxPool2d()
        )
    
        self.lstm = BidirectionalLSTM(256, hidden_size, n_classes)

    def forward(self, x, lengths):
        """CRNN forward pass."""
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x = self.lstm(x, lengths)

        return x
