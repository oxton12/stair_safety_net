import mxnet as mx

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class StairSafetyNetLSTM(mx.gluon.nn.HybridBlock):
    def __init__(self, LSTM_layer_size, LSTM_num_layers, dropout_rate, sequence_len):
        super().__init__()
        #self.norm_block = mx.gluon.nn.LayerNorm(axis=2)
        self.LSTM_block = mx.gluon.rnn.LSTM(LSTM_layer_size, LSTM_num_layers, input_size=64*48*17, dropout=dropout_rate, layout='NTC')
        self.fc_block_1 = mx.gluon.nn.Dense(64, activation="relu")
        self.drop_block_1 = mx.gluon.nn.Dropout(8/64)
        self.fc_block_2 = mx.gluon.nn.Dense(64, activation="relu")
        self.drop_block_2 = mx.gluon.nn.Dropout(8/64)
        self.fc_block_3 = mx.gluon.nn.Dense(32, activation="relu")
        self.drop_block_3 = mx.gluon.nn.Dropout(5/32)
        self.fc_block_4 = mx.gluon.nn.Dense(32, activation="relu")
        self.drop_block_4 = mx.gluon.nn.Dropout(5/32)
        self.fc_block_5 = mx.gluon.nn.Dense(2)
        self.softmax = mx.nd.softmax


    def hybrid_forward(self, x):
        #x_norm = self.norm_block(x)
        y1 = self.LSTM_block(x)
        y2 = self.fc_block_1(y1)
        y2_drop = self.drop_block_1(y2)
        y3 = self.fc_block_2(y2_drop)
        y3_drop = self.drop_block_2(y3)
        y4 = self.fc_block_3(y3_drop)
        y4_drop = self.drop_block_3(y4)
        y5 = self.fc_block_4(y4_drop)
        y5_drop = self.drop_block_4(y5)
        y6 = self.fc_block_5(y5_drop)
        y7 = self.softmax(y6)
        return y7

if __name__ == "__main__":
    device=mx.gpu()
    net = StairSafetyNetLSTM(32, 2, 4)
    net.initialize(mx.init.Xavier(), ctx=device)
    x = mx.nd.ones((4, 4, 64*48*17), ctx=device)
    print(net.hybrid_forward(x))