import mxnet as mx
from mxnet import gluon
from gluoncv import model_zoo, data
from gluoncv.data.transforms.pose import detector_to_simple_pose
import cv2

import os
import random
from pathlib import Path

from net_LSTM_only import StairSafetyNetLSTM
from get_mean_std import get_mean_std

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class SequenceAssembler():
    def __init__(self, data_path, label, seq_len, mean, std):
        self.data_path = data_path
        self.label = label
        self.seq_counter = 0
        self.prev_end = 0
        self.seq_max = 3000
        self.seq_len = seq_len
        self.elements_amount = mx.nd.load(self.data_path)[0].shape[0]
        self.step = 1
        self.ended = False
        max_step_by_amount = (self.elements_amount - seq_len) // (seq_len - 1) + 1
        max_step_hard = 3
        self.max_step = min(max_step_by_amount, max_step_hard)
        if self.max_step < 1:
            self.ended = True
        self.mean = mean
        self.std = std


    def get_sequence(self):
        if self.ended:
            raise IndexError("Unused sequences ended")
        data = mx.nd.load(self.data_path)[0]
        seq_start = self.prev_end
        seq_end = seq_start + self.seq_len * self.step
        sequence = data[seq_start : seq_end : self.step]
        #sequence_norm = normalize(sequence, self.mean, self.std)
        reshaped_sequence = sequence.reshape((1, self.seq_len, 17*64*48))
        self.seq_counter += 1
        self.prev_end = seq_end
        if (self.seq_counter == self.seq_max) or ((seq_end + self.seq_len * self.step) >= (self.elements_amount + self.step - 1)):
            self.seq_counter = 0
            self.prev_end = 0
            if self.step < self.max_step:
                self.step += 1
            else:
                self.ended = True

        return reshaped_sequence, self.label
    

def normalize(data, mean, std):
    if data.shape[1] != len(mean) or data.shape[1] != len(std):
        raise Exception
    data -= mean
    data /= std
    return data
    

def get_data_paths(path, context):
    labeled_paths = {}
    for root, dirs, files in os.walk(path):
        parent = Path(root).name
        action = mx.nd.array([[0, 1]]) if parent == "not_holding" else mx.nd.array([[1, 0]])
        mx_action = mx.nd.array(action).as_in_context(context)
        for file in files:
            labeled_paths[os.path.join(path, parent, file)] = mx_action
    return labeled_paths


def evaluate_test(test_labeled_paths, net, seq_len, mean, std):
    test_paths = list(test_labeled_paths.keys())
    acc, met = evaluate_val(test_paths, test_labeled_paths, net, seq_len, mean, std)
    return acc, met


def evaluate_val(data_paths, labeled_paths, net, seq_len, mean, std):
    metric_test = mx.metric.MSE()
    accuracy_test = mx.metric.Accuracy()
    val_sequence_assemblers = [SequenceAssembler(data_path, labeled_paths[data_path], seq_len, mean, std) 
                               for data_path in data_paths]
    while any(assembler.ended == False for assembler in val_sequence_assemblers):
        available_assemblers = [assembler for assembler in val_sequence_assemblers if assembler.ended == False]
        selected_assembler = random.choice(available_assemblers)
        sequence, label = selected_assembler.get_sequence()
        output = net.hybrid_forward(sequence)
        accuracy_test.update(label, output)
        metric_test.update(label, output)
    _, acc = accuracy_test.get()
    _, met = metric_test.get()
    return round(acc, 5), round(met, 5)


def train():
    TRAIN_SET = "ndarray-data/train"
    TEST_SET = "ndarray-data/test"
    EPOCHS = 200
    BATCH_SIZE = 32
    SEQUENCE_LEN = 8
    CV_FOLDS = 5
    lr = 0.0004
    epsilon = 2e-07
    LSTM_layer_size = 32
    LSTM_num_layeres = 3
    dropout = 1/16
    weight_decay = 0
    log_interval = 10

    device = mx.gpu()
    net = StairSafetyNetLSTM(LSTM_layer_size, LSTM_num_layeres, dropout, SEQUENCE_LEN)

    train_labeled_paths = get_data_paths(TRAIN_SET, device)
    test_labeled_paths = get_data_paths(TEST_SET, device)

    # cross-validation implementation
    cross_val_paths = list(train_labeled_paths.keys())
    random.shuffle(cross_val_paths)
    fold_size = int(len(cross_val_paths) / CV_FOLDS)
    data_folds = [cross_val_paths[i * fold_size : (i + 1) * fold_size] for i in range(CV_FOLDS - 1)]
    data_folds.append(cross_val_paths[(CV_FOLDS - 1) * fold_size :])

    #mean, std = get_mean_std(train_labeled_paths)
    mean, std = 0, 0

    net.initialize(mx.init.Xavier(), ctx=device)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'epsilon': epsilon, 'wd': weight_decay})
    metric = mx.metric.MSE()
    L2 = gluon.loss.L2Loss()
    accuracy = mx.metric.Accuracy()

    for epoch in range(EPOCHS):
        accuracy.reset()
        metric.reset()
        batch_count = 0
        val_fold_id = epoch % CV_FOLDS
        val_fold = data_folds[val_fold_id]
        train_paths = [data_path for fold in data_folds if fold != val_fold for data_path in fold]
        batch = mx.nd.empty((0, SEQUENCE_LEN, 17*64*48), ctx=device)
        label_batch = mx.nd.empty((0, 2), ctx=device)
        train_sequence_assemblers = [SequenceAssembler(data_path, train_labeled_paths[data_path], SEQUENCE_LEN, mean, std) 
                               for data_path in train_paths]

        while any(assembler.ended == False for assembler in train_sequence_assemblers):
            available_assemblers = [assembler for assembler in train_sequence_assemblers if assembler.ended == False]
            selected_assembler = random.choice(available_assemblers)
            sequence, label = selected_assembler.get_sequence()

            try:
                batch = mx.nd.concat(batch, sequence, dim=0)
                label_batch = mx.nd.concat(label_batch, label, dim=0)
            except mx.base.MXNetError:
                batch = sequence
                label_batch = label
            if batch.shape[0] != BATCH_SIZE:
                continue

            with mx.autograd.record():
                output = net.hybrid_forward(batch)
                loss = L2(output, label_batch)
            mx.autograd.backward(loss, retain_graph=True)
            trainer.step(BATCH_SIZE)
            accuracy.update(label_batch, output)
            metric.update(label_batch, output)
            batch = mx.nd.empty((0, SEQUENCE_LEN, 17*64*48), ctx=device)
            batch_count += 1

            if batch_count % log_interval == 0:
                _, acc = accuracy.get()
                _, met = metric.get()
                #val_acc, val_met = evaluate_val(val_fold, train_labeled_paths, net, SEQUENCE_LEN, mean, std)
                print(f"""Epoch[{epoch + 1}] Batch[{batch_count}] train mse = {round(met, 5)} | train accuracy = {round(acc, 5)}""")


        if batch.shape[0] != 0:
            with mx.autograd.record():
                output = net.hybrid_forward(batch)
                loss = L2(output, label_batch)
            mx.autograd.backward(loss)
            trainer.step(batch.shape[0])
            accuracy.update(label_batch, output)
            metric.update(label_batch, output)

        _, acc = accuracy.get()
        _, met = metric.get()
        test_acc, test_met = evaluate_test(test_labeled_paths, net, SEQUENCE_LEN, mean, std)
        print(" ")
        print(f"Epoch[{epoch + 1}] train mse={round(met, 5)} | test mse={test_met} | test accuracy={test_acc}")
        print(" ")

    net.save("stair_safety_LSTM_only")


if __name__ == "__main__":
    train()