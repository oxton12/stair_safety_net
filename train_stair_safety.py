import mxnet as mx
from mxnet import gluon
from gluoncv import model_zoo, data
from gluoncv.data.transforms.pose import detector_to_simple_pose
import matplotlib.pyplot as plt
import numpy as np
import cv2

import os
import shutil
import random
from pathlib import Path

from net_LSTM_only import StairSafetyNetLSTM
from get_mean_std import get_mean_std

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class SequenceDataManager():
    def __init__(self, data_path, test_split, cv_folds, seq_len, max_step_hard, context):
        labeled_paths = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                path = os.path.join(root, file)
                parent = Path(root).name
                label = [[0, 1]] if parent == "not_holding" else [[1, 0]]
                mx_label = mx.nd.array(label).as_in_context(context)
                labeled_paths.append({"path": path, "label": mx_label})
        
        all_sequences = []
        #steps_info = {}
        #for i in range(1, max_step_hard + 1):z
            #steps_info[i] = 0
        for labeled_path in labeled_paths:
            path = labeled_path["path"]
            label = labeled_path["label"]
            data = mx.nd.load(path)[0]
            elements_amount = data.shape[0]
            max_step_by_amount = (elements_amount - seq_len) // (seq_len - 1) + 1
            max_step = min(max_step_hard, max_step_by_amount)
            if max_step < 1:
                continue
            seq_start = 0
            while(True):
                current_step = round(random.triangular(1, max_step, 0))
                seq_end = seq_start + seq_len * current_step
                while ((seq_end) >= (elements_amount + current_step - 1)):
                    current_step -= 1
                    seq_end = seq_start + seq_len * current_step
                for i in range(0, current_step):
                    sequence_info = {"path": path, "seq_start": seq_start, "seq_end": seq_end, "step": current_step, "label": label}
                    all_sequences.append(sequence_info)
                    #steps_info[current_step] += 1
                    if (seq_end + 1) < (elements_amount + current_step - 1):
                        seq_start += 1
                        seq_end += 1
                    else:
                        break
                seq_start = seq_end
                if (seq_end + seq_len * 1) >= (elements_amount + current_step - 1):
                    break
            
        #print(steps_info)
        self.test_sequences = random.sample(all_sequences, k=round(len(all_sequences) * test_split))
        train_sequences = [sequence for sequence in all_sequences if sequence not in self.test_sequences]
        random.shuffle(train_sequences)
        self.cv_folds = cv_folds
        fold_size = int(len(train_sequences) / cv_folds)
        self.data_folds = [train_sequences[i * fold_size : (i + 1) * fold_size] for i in range(cv_folds - 1)]
        self.data_folds.append(train_sequences[(cv_folds - 1) * fold_size :])
        self.val_fold_id = 0
        self.next_fold()


    def get_sequence(self, sequence_info):
        path = sequence_info["path"]
        seq_start = sequence_info["seq_start"]
        seq_end = sequence_info["seq_end"]
        step = sequence_info["step"]
        label = sequence_info["label"]

        data = mx.nd.load(path)[0]
        sequence = data[seq_start : seq_end : step]
        reshaped_sequence = sequence.reshape((1, len(sequence), 17*64*48))
        return reshaped_sequence, label
    

    def next_fold(self):
        if(self.val_fold_id == self.cv_folds):
            self.val_fold_id = 0
        self.val_sequences = self.data_folds[self.val_fold_id]
        self.train_sequences = [sequence for fold in self.data_folds if fold != self.val_sequences for sequence in fold]
        self.val_fold_id += 1
    

def normalize(data, mean, std):
    if data.shape[1] != len(mean) or data.shape[1] != len(std):
        raise Exception
    data -= mean
    data /= std
    return data


def evaluate_test(data_manager, net):
    metric = mx.metric.MSE()
    accuracy = mx.metric.Accuracy()
    for test_sequence_info in data_manager.test_sequences:
        sequence, label = data_manager.get_sequence(test_sequence_info)
        output = net.hybrid_forward(sequence)
        accuracy.update(label, output)
        metric.update(label, output)
    _, acc = accuracy.get()
    _, met = metric.get()
    return acc, met
            

def evaluate_val(data_manager, net):
    metric = mx.metric.MSE()
    accuracy = mx.metric.Accuracy()
    for val_sequence_info in data_manager.val_sequences:
        sequence, label = data_manager.get_sequence(val_sequence_info)
        output = net.hybrid_forward(sequence)
        accuracy.update(label, output)
        metric.update(label, output)
    _, acc = accuracy.get()
    _, met = metric.get()
    return acc, met


def get_precision_recall_graph(data_manager, net):
    confidences = [i/10 for i in range(11)]
    precisions = []
    recalls = []
    for conf in confidences:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for test_sequence_info in data_manager.test_sequences:
            sequence, label = data_manager.get_sequence(test_sequence_info)
            output = net.hybrid_forward(sequence)
            out_positive = output[0][0] >= conf
            label_positive = label[0][0] == 1
            if out_positive and label_positive:
                TP += 1
            elif out_positive and not label_positive:
                FP += 1
            elif not out_positive and not label_positive:
                TN += 1
            else:
                FN += 1
        if TP != 0:
            precisions.append(TP / (TP + FP))
            recalls.append(TP / (TP + FN))
        else:
            precisions.append(0)
            recalls.append(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(confidences, precisions, color='b')
    ax.plot(confidences, recalls, color='g')
    plt.xlabel("Confidence")
    plt.ylabel("Score")
    plt.xlim([0, 1])
    plt.xticks(np.arange(0, 1.1, step=0.1), labels=[str(i/10) for i in range(11)])
    plt.grid(True)
    plt.legend(["precision", "recall"])

    return plt


def train():
    DATA_PATH = "ndarray-data"
    EPOCHS = 1000
    BATCH_SIZE = 16
    SEQUENCE_LEN = 4
    TEST_SPLIT = 0.2
    CV_FOLDS = 5
    lr = 0.0004
    LR_DECAY = 0.4
    DECAY_ON = [100, 200, 300, 400, 500]
    EPSILON = 2e-07

    MAX_SEQ_STEP = 5

    LSTM_LAYER_SIZE = 64
    LSTM_NUM_LAUERS = 3
    DROPOUT = 1/16
    WEIGHT_DECAY = 0

    LOG_INTERVAL = 10

    MAX_ACC = 0.89
    MAX_ACC_MODEL_PATH = ""

    device = mx.gpu()
    net = StairSafetyNetLSTM(LSTM_LAYER_SIZE, LSTM_NUM_LAUERS, DROPOUT, SEQUENCE_LEN)

    data_manager = SequenceDataManager(DATA_PATH, TEST_SPLIT, CV_FOLDS, SEQUENCE_LEN, MAX_SEQ_STEP, device)

    #mean, std = get_mean_std(train_labeled_paths)
    mean, std = 0, 0

    net.initialize(mx.init.Xavier(), ctx=device)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'epsilon': EPSILON, 'wd': WEIGHT_DECAY})
    metric = mx.metric.MSE()
    L2 = gluon.loss.L2Loss()
    accuracy = mx.metric.Accuracy()

    for epoch in range(EPOCHS):
        accuracy.reset()
        metric.reset()
        batch_count = 0
        batch = mx.nd.empty((0, SEQUENCE_LEN, 17*64*48), ctx=device)
        label_batch = mx.nd.empty((0, 2), ctx=device)

        for train_sequence_info in data_manager.train_sequences:
            sequence, label = data_manager.get_sequence(train_sequence_info)

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

            if batch_count % LOG_INTERVAL == 0:
                _, acc = accuracy.get()
                _, met = metric.get()
                #val_acc, val_met = evaluate_val(val_fold, train_labeled_paths, net, SEQUENCE_LEN, mean, std)
                print(f"""Epoch[{epoch}] Batch[{batch_count}] train mse = {met:e} | train accuracy = {round(acc, 3)}""")


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
        val_acc, val_met = evaluate_val(data_manager, net)
        print(" ")
        print(f"Epoch[{epoch}] val mse={val_met:e} | val accuracy={round(val_acc, 3)}")
        print(" ")
        data_manager.next_fold()

        if val_acc >= 0.8:
            test_acc, test_met = evaluate_test(data_manager, net)
            print(" ")
            print(f"Epoch[{epoch}] | test mse={test_met:e} | test accuracy={round(test_acc, 3)}")
            print(" ")

            if test_acc >= MAX_ACC:
                dir_path = os.path.join("saved-models", f"epoch_{epoch}_acc_{round(test_acc, 3)}")
                os.mkdir(dir_path)
                net.save(os.path.join(dir_path, "stair_safety_LSTM_only"))
                MAX_ACC = test_acc
                if MAX_ACC_MODEL_PATH != "":
                    shutil.rmtree(MAX_ACC_MODEL_PATH)
                MAX_ACC_MODEL_PATH = dir_path
                plt = get_precision_recall_graph(data_manager, net)
                plt.savefig(os.path.join(dir_path, "precision_recall"))

        if epoch in DECAY_ON:
            lr = lr * LR_DECAY
            trainer.set_learning_rate(lr)
            print(f"lr = {lr}")


if __name__ == "__main__":
    train()