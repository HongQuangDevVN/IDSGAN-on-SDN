import matplotlib as mpl
mpl.use('TkAgg')
import pandas as pd
import numpy as np
import torch as th
from torch.autograd import Variable as V
from torch import nn, optim
from preprocessing import Preprocess_BlackBox, CreateBatch_BlackBox
from model.model_class import Blackbox_IDS
import matplotlib.pyplot as plt
from datetime import datetime

train_dataset = pd.read_csv("dataset/half_KDDTrain+.csv")
test_dataset = pd.read_csv("dataset/KDDTest+.csv")
train_data, train_label, test_data, test_label = Preprocess_BlackBox(train_dataset, test_dataset)

INPUT_DIM = train_data.shape[1]
OUTPUT_DIM = 2
BATCH_SIZE = 32
MAX_EPOCH = 100
LEARNING_RATE = 0.001

ids_model = Blackbox_IDS(INPUT_DIM, OUTPUT_DIM)
opt = optim.Adam(ids_model.parameters(), LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()
train_losses, test_losses = [], []

def TrainBlackbox(data, label):
    ids_model.train()
    batch_data, batch_label = CreateBatch_BlackBox(data, label, BATCH_SIZE)
    run_loss = 0
    count=0
    for data, label in zip(batch_data, batch_label):
        ids_model.zero_grad()
        data = V(th.Tensor(data), requires_grad=True)
        label = V(th.LongTensor(label))
        out = ids_model(data)
        loss = loss_function(out, label)
        run_loss += loss.item()
        count+=1
        loss.backward()
        opt.step()
    return run_loss / count


def TestBlackbox(data, label):
    ids_model.eval()
    batch_data, batch_label = CreateBatch_BlackBox(data, label, BATCH_SIZE)
    run_loss = 0
    count=0
    with th.no_grad():
        for data, label in zip(batch_data, batch_label):
            data = th.Tensor(data)
            label = th.LongTensor(label)
            out = ids_model(data)
            loss = loss_function(out, label)
            run_loss += loss.item()
            count+=1
    return run_loss / count


def main():
    print("BlackBox IDS start training")
    print("-" * 100)

    for epoch in range(MAX_EPOCH):
        train_loss = TrainBlackbox(train_data, train_label)     #train and return loss
        train_losses.append(train_loss)                         #save loss
        test_loss = TestBlackbox(test_data, test_label)         #test and return loss
        test_losses.append(test_loss)                           #save loss
        print(f"{epoch} : {train_loss} \t {test_loss}")

    # training finished
    print("BlackBox IDS finished training")

    #draw the loss graph
    th.save(ids_model.state_dict(), 'model/save/BlackBox/IDS.pth')
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
