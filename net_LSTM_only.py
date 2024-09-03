import mxnet as mx

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class StairSafetyNetLSTM(mx.gluon.nn.HybridBlock):
    def __init__(self, LSTM_layer_size, LSTM_num_layers, LSTM_dropout, sequence_len):
        super().__init__()
        self.norm_block = mx.gluon.nn.BatchNorm(axis = 2)
        self.LSTM_block = mx.gluon.rnn.LSTM(LSTM_layer_size, LSTM_num_layers, input_size=64*48*17, dropout=LSTM_dropout, bidirectional=True)
        self.fc_block_1 = mx.gluon.nn.Dense(128, activation="relu")
        self.fc_block_2 = mx.gluon.nn.Dense(64, activation="relu")
        self.fc_block_3 = mx.gluon.nn.Dense(32, activation="relu")
        self.fc_block_4 = mx.gluon.nn.Dense(2)

        self.sequence_len = sequence_len

    def hybrid_forward(self, x):
        x_norm = self.norm_block(x)
        y1 = self.LSTM_block(x_norm)
        y2 = self.fc_block_1(y1)
        y3 = self.fc_block_2(y2)
        y4 = self.fc_block_3(y3)
        y5 = self.fc_block_4(y4)
        return y5

if __name__ == "__main__":
    device=mx.gpu()
    net = StairSafetyNetLSTM(32, 2, 4)
    net.initialize(mx.init.Xavier(), ctx=device)
    x = mx.nd.ones((4, 4, 64*48*17), ctx=device)
    print(net.hybrid_forward(x))