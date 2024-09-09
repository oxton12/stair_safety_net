import mxnet as mx

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class StairSafetyNetLSTM(mx.gluon.nn.HybridBlock):
    def __init__(self, LSTM_layer_size, LSTM_num_layers, dropout_rate, sequence_len):
        super().__init__()
        self.LSTM_block = mx.gluon.rnn.LSTM(LSTM_layer_size, LSTM_num_layers, input_size=64*48*17, dropout=dropout_rate, layout='NTC')
        self.fc_block_1 = mx.gluon.nn.Dense(32, activation="relu")
        self.drop_1 = mx.gluon.nn.Dropout()
        self.fc_block_2 = mx.gluon.nn.Dense(2)

        self.sequence_len = sequence_len

    def hybrid_forward(self, x):
        y1 = self.LSTM_block(x)
        y2 = self.fc_block_1(y1)
        y2_drop = self.drop_1(y2)
        y3 = self.fc_block_2(y2_drop)
        return mx.nd.sigmoid(y3)

if __name__ == "__main__":
    device=mx.gpu()
    net = StairSafetyNetLSTM(32, 2, 4)
    net.initialize(mx.init.Xavier(), ctx=device)
    x = mx.nd.ones((4, 4, 64*48*17), ctx=device)
    print(net.hybrid_forward(x))