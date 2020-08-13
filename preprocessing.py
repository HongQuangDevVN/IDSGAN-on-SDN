import pandas as pd
import numpy as np


train_features = ["duration", "protocol_type", "src_bytes", "dst_bytes", "count", "srv_count", "is_guest_login", "root_shell", "num_failed_logins", "" "class"]
protocol_map = {'tcp': 1, 'udp': 2, 'icmp': 3}


# Function - Chia bộ dataset thành các Batch mỗi batch có size = batch_size
def CreateBatch_GAN(x, batch_size):
    # Comment - a là danh sách các số từ 0 -> len(x)
    a = list(range(len(x)))
    # Comment - Xáo trộn a lên, đảo lộn vị trí các phần từ của a
    np.random.shuffle(a)
    # Comment - Xáo trộn các phần tử trong x
    x = x[a]
    # Comment - Mảng các batch, mỗi batch có số phần tử là batch size
    batch_x = [x[batch_size * i: (i + 1) * batch_size, :] for i in range(len(x) // batch_size)]
    return batch_x


def CreateBatch_BlackBox(data, label, batch_size):
    a = list(range(len(data)))
    np.random.shuffle(a)
    data = data[a]
    label = label[a]

    batch_data = [data[batch_size * i: (i + 1) * batch_size, :].tolist() for i in range(len(data) // batch_size)]
    batch_label = [label[batch_size * i: (i + 1) * batch_size].tolist() for i in range(len(data) // batch_size)]
    return batch_data, batch_label


def Preprocess_GAN(train):
    train["protocol_type"]=train["protocol_type"].map(protocol_map)
    # Comment - Loc ra cac cot tuong ung voi cac features sdn trong tap train
    trash = list(set(train.columns) - set(train_features))
    for t in trash:
        del train[t]
    # Comment - Chuyen cac cot gia tri so ve gia tri min max
    numeric_columns = list(train.select_dtypes(include=['int', "float"]).columns)
    for c in numeric_columns:
        max_ = train[c].max()
        min_ = train[c].min()
        train[c] = train[c].map(lambda x: (x - min_) / (max_ - min_))


    # Comment - Gan nhan o dang so: 1: annomaly; 0: normaly
    train["class"] = train["class"].map(lambda x: 1 if x == "anomaly" else 0)
    # Comment - raw_attack la tat ca cac record co nhan "anomaly"
    raw_attack = np.array(train[train["class"] == 1])[:, :-1]
    # Comment - normal la tat ca cac record co nhan "nomaly"
    normal = np.array(train[train["class"] == 0])[:, :-1]
    # Comment - Lay label
    true_label = train["class"]

    del train["class"]

    return train, raw_attack, normal, true_label


def Preprocess_BlackBox(train, test):
    # Comment - Xu ly tap train
    # Chuyen tcp, udp, icmp sang 1 2 3
    train["protocol_type"]=train["protocol_type"].map(protocol_map)
    # Comment - Loc ra cac cot tuong ung voi cac features sdn trong tap train
    trash = list(set(train.columns) - set(train_features))
    for t in trash:
        del train[t]
    # Chuan hoa ve khoang [0-1]
    numeric_columns = list(train.select_dtypes(include=['int', "float"]).columns)
    for c in numeric_columns:
        max_ = train[c].max()
        min_ = train[c].min()
        train[c] = train[c].map(lambda x: (x - min_) / (max_ - min_))

    # Comment - Chuyen label sang dang numberic (anomaly=1, normaly=0)
    train["class"] = train["class"].map(lambda x: 1 if x == "anomaly" else 0)


    # Xy ly tap test
    test["protocol_type"]=test["protocol_type"].map(protocol_map)
    trash = list(set(test.columns) - set(train_features))
    for t in trash:
        del test[t]
    numeric_columns = list(test.select_dtypes(include=['int', "float"]).columns)
    for c in numeric_columns:
        max_ = test[c].max()
        min_ = test[c].min()
        test[c] = test[c].map(lambda x: (x - min_) / (max_ - min_))
    test["class"] = test["class"].map(lambda x: 1 if x == "anomaly" else 0)

    # return data
    train_data, train_label = np.array(train[train.columns[train.columns != "class"]]), np.array(train["class"])
    test_data, test_label = np.array(test[test.columns[test.columns != "class"]]), np.array(test["class"])
    return train_data, train_label, test_data, test_label
