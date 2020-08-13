import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
import pandas as pd
import torch as th
from torch.autograd import Variable as V
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocessing import Preprocess_GAN,CreateBatch_GAN
from model.model_class import Blackbox_IDS,Generator,Discriminator
import matplotlib.pyplot as plt

def compute_gradient_penalty(D, normal_t, attack_t):
    alpha = th.Tensor(np.random.random((normal_t.shape[0], 1)))
    between_n_a = (alpha * normal_t + ((1 - alpha) * attack_t)).requires_grad_(True)
    d_between_n_a = D(between_n_a)
    adv = V(th.Tensor(normal_t.shape[0], 1).fill_(1.0), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_between_n_a,
        inputs=between_n_a,
        grad_outputs=adv,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

train_dataset = pd.read_csv("dataset/other_half_KDDTrain+.csv")
test_dataset = pd.read_csv("dataset/KDDTest+.csv")

train_data,raw_attack,normal,true_label = Preprocess_GAN(train_dataset)

#DEFINE
BATCH_SIZE = 64 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10     # Gradient penalty lambda hyperparameter
MAX_EPOCH = 100 # How many generator iterations to train for
D_G_INPUT_DIM = len(train_data.columns)
G_OUTPUT_DIM = len(train_data.columns)
D_OUTPUT_DIM = 1
CLAMP = 0.01
LEARNING_RATE=0.0001

# Load BlackBox IDS model 
ids_model = Blackbox_IDS(D_G_INPUT_DIM,2)
param = th.load('model/save/BlackBox/IDS.pth')
ids_model.load_state_dict(param)


generator = Generator(D_G_INPUT_DIM,G_OUTPUT_DIM)
print(100*'=')
print(generator)

discriminator = Discriminator(D_G_INPUT_DIM,D_OUTPUT_DIM)
print(100*'=')
print(discriminator)


#thuật toán tối ưu. tương tự Gradient Descent. https://viblo.asia/p/thuat-toan-toi-uu-adam-aWj53k8Q56m
optimizer_G = optim.RMSprop(generator.parameters(), LEARNING_RATE)
optimizer_D = optim.RMSprop(discriminator.parameters(), LEARNING_RATE)

# Comment - Do trong không thể cho hết cả dataset vào được, nên dataset ra thành các batch (các phần nhỏ hơn, bằng nhau)
batch_attack = CreateBatch_GAN(raw_attack,BATCH_SIZE)
d_losses,g_losses = [],[] #lưu lại để vẽ biểu đồ loss
ids_model.eval()

generator.train()
discriminator.train()

cnt = -5
print("IDSGAN start training")
print("-"*100)
for epoch in range(MAX_EPOCH):
    # Comment - Mỗi train epoch tạo batch 1 lần
    normal_batch = CreateBatch_GAN(normal,BATCH_SIZE)
    epoch_g_loss = 0.
    epoch_d_loss = 0.
    c=0
    for nb in normal_batch:
        normal_b = th.Tensor(nb)
        #  Train Generator
        for p in discriminator.parameters():
            p.requires_grad = False

        optimizer_G.zero_grad()

        # random_traffic - Lay tu raw_attack ngau nhien n=BATCH_SIZE phan tu
        random_attack_traffic = raw_attack[np.random.randint(0,len(raw_attack),BATCH_SIZE)]
        # random_traffic_noised - Lay random_traffic + noise la gia tri random tu 0 - 1

        ###!! random_traffic_noised - Nhu vay random_traffic_noised co the la gia tri > 1
        random_traffic_noised = random_attack_traffic + np.random.uniform(0,1,(BATCH_SIZE,D_G_INPUT_DIM))

        z = V(th.Tensor(random_traffic_noised))
        adversarial_traffic = generator(z)
        
        D_pred= discriminator(adversarial_traffic) #điểm của generated output
        g_loss = -th.mean(D_pred)
        g_loss.backward()
        optimizer_G.step()

        epoch_g_loss += g_loss.item()
        # Train Discriminator
        for p in discriminator.parameters():
            p.requires_grad = True

        for c in range(CRITIC_ITERS):
            optimizer_D.zero_grad()
            for p in discriminator.parameters():
                p.data.clamp_(-CLAMP, CLAMP)

            temp_data = raw_attack[np.random.randint(0,len(raw_attack),BATCH_SIZE)] + np.random.uniform(0,1,(BATCH_SIZE,D_G_INPUT_DIM))
            z = V(th.Tensor(temp_data))
            adversarial_traffic = generator(z).detach()
            ids_input = th.cat((adversarial_traffic,normal_b))

            l = list(range(len(ids_input)))
            np.random.shuffle(l)
            ids_input = V(th.Tensor(ids_input[l]))
            ids_pred = ids_model(ids_input)
            ids_pred_label = th.argmax(nn.Sigmoid()(ids_pred),dim = 1).detach().numpy()

            pred_normal = ids_input.numpy()[ids_pred_label==0]
            pred_attack = ids_input.numpy()[ids_pred_label==1]

            if len(pred_attack) == 0:
                cnt += 1
                break

            D_normal = discriminator(V(th.Tensor(pred_normal)))
            D_attack= discriminator(V(th.Tensor(pred_attack)))

            loss_normal = th.mean(D_normal)
            loss_attack = th.mean(D_attack)
            #gradient_penalty = compute_gradient_penalty(discriminator, normal_b.data, adversarial_traffic.data)
            d_loss =  loss_attack - loss_normal #+ LAMBDA * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
            epoch_d_loss += d_loss.item()

    d_losses.append(epoch_d_loss/CRITIC_ITERS)
    g_losses.append(epoch_g_loss)
    print(f"{epoch} : {epoch_g_loss} \t {epoch_d_loss/CRITIC_ITERS}")
'''
    if cnt >= 100:
        print("Not exist predicted attack traffic")
        break
'''

print("IDSGAN finish training")

th.save(generator.state_dict(), 'model/save/GAN/generator.pth')
th.save(discriminator.state_dict(), 'model/save/GAN/discriminator.pth')

plt.plot(d_losses,label = "D_loss")
plt.plot(g_losses, label = "G_loss")
plt.legend()
plt.show()

