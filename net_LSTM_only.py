import mxnet as mx

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class StairSafetyNetLSTM(mx.gluon.nn.HybridBlock):
    def __init__(self, LSTM_layer_size, LSTM_num_layers, dropout_rate, sequence_len):
        super().__init__()
        #self.norm_block = mx.gluon.nn.LayerNorm(axis=2)
        self.LSTM_block = mx.gluon.rnn.LSTM(LSTM_layer_size, LSTM_num_layers, input_size=64*48*17, dropout=dropout_rate, layout='NTC')
        self.fc_block_1 = mx.gluon.nn.Dense(8, activation="relu")
        self.fc_block_2 = mx.gluon.nn.Dense(2)

        self.fc_block_2.weight.wd_mult = 0

        self.sequence_len = sequence_len

    def hybrid_forward(self, x):
        #x_norm = self.norm_block(x)
        y1 = self.LSTM_block(x)
        y2 = self.fc_block_1(y1)
        y3 = self.fc_block_2(y2)
        return y3

if __name__ == "__main__":
    device=mx.gpu()
    net = StairSafetyNetLSTM(32, 2, 4)
    net.initialize(mx.init.Xavier(), ctx=device)
    x = mx.nd.ones((4, 4, 64*48*17), ctx=device)
    print(net.hybrid_forward(x))