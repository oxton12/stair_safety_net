import mxnet as mx

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class StairSafetyNet(mx.gluon.nn.HybridBlock):
    def __init__(self, LSTM_layer_size, LSTM_num_layers, sequence_len, context):
        super().__init__()
        model_json = 'model/Ultralight-Nano-SimplePose.json'
        model_params = "model/Ultralight-Nano-SimplePose.params"

        symbol = mx.symbol.load(model_json)
        input = mx.sym.var('data', dtype='float32')
        posenet = mx.gluon.SymbolBlock(symbol, input)
        posenet.collect_params().load(model_params, ctx=context)
        for param in posenet.collect_params().values():
            param.grad_req = 'null'

        self.pose_block = posenet
        self.LSTM_block = mx.gluon.rnn.LSTM(LSTM_layer_size, LSTM_num_layers)
        self.fc_block = mx.gluon.nn.Dense(2)

        self.sequence_len = sequence_len

    def hybrid_forward(self, x):
        if x.shape[0] != self.sequence_len:
            raise mx.base.MXNetError("Incompatible input shape")
        y1 = self.pose_block(x)
        y2 = y1.reshape(-1, 17, 3072)
        y3 = self.LSTM_block(y2)
        y4 = self.fc_block(y3)
        return mx.nd.softmax(y4[-1])

if __name__ == "__main__":
    device=mx.gpu()
    net = StairSafetyNet(32, 2, 4, context=device)
    net.initialize(mx.init.Xavier(), ctx=device)
    x = mx.nd.ones((4, 3, 256, 192), ctx=device)
    print(net.hybrid_forward(x))