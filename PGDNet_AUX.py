import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from PGD_AUXFunctions import find_capacity
from random import randint, seed


# ------------------------------------- functions ----------------------------------------------------

def monotone_arr(arr):
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            arr[i] = arr[i - 1]
    return arr


def min_rate_vs_capacity(ch_n_valid, min_rate_v_arr, h_valid, k, sigma, name):
    """
    :param ch_n_valid: Number of channels to work on
    :param min_rate_v_arr: Sum rate data along PGD iterations
    :param h_valid: Channels to work on
    :param k:  Number of PGD iterations
    :param name: graph's name
    """
    n_arr = []
    if type(name) == int:
        seed(name)

    for i in range(4):
        n_arr.append(randint(0, ch_n_valid - 1))

    srv1 = monotone_arr(min_rate_v_arr[:, n_arr[0]].detach().numpy())
    srv2 = monotone_arr(min_rate_v_arr[:, n_arr[1]].detach().numpy())
    srv3 = monotone_arr(min_rate_v_arr[:, n_arr[2]].detach().numpy())
    srv4 = monotone_arr(min_rate_v_arr[:, n_arr[3]].detach().numpy())
    c_arr = []
    for i in range(len(n_arr)):
        _, c = find_capacity(h_valid[n_arr[i]].tolist(), sigma)
        c_arr.append(c)

    t_K = np.linspace(start=1, stop=k, num=k)
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(t_K, srv1, linestyle='dotted', color='green')
    plt.axhline(y=c_arr[0], linestyle='--', label=f'channel 1 capacity', color='green')
    plt.plot(t_K, srv2, linestyle='dotted', color='orange')
    plt.axhline(y=c_arr[1], linestyle='--', label=f'channel 2 capacity', color='orange')
    plt.plot(t_K, srv3, linestyle='dotted', color='blue')
    plt.axhline(y=c_arr[2], linestyle='--', label=f'channel 3 grid capacity', color='blue')
    plt.plot(t_K, srv4, linestyle='dotted', color='black')  # label=f'Valid sum-rate channel: {n_arr[3]}'
    plt.axhline(y=c_arr[3], linestyle='--', label='channel 4 grid capacity', color='black')  # label=f'channel {
    # n_arr[3]} capacity'
    plt.xlabel('Number of Iterations')
    plt.ylabel('Min Rate')
    plt.legend(loc='best')
    # plt.title('Comparison of Different Valid Channels')
    plt.grid()
    # plt.show()
    plt.savefig(f'capacity_{name}.jpg')
    plt.close()


def train_valid_loss(epochs, train_loss, valid_loss, name):
    """
    :param epochs: Number of training epochs
    :param train_loss: Train loss vector
    :param valid_loss: Validation loss vector
    :param name: graph's name
    :return: Plots a graph of train/valid rate as a function of epochs
    """
    t_epochs = np.linspace(start=1, stop=epochs, num=epochs)
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(t_epochs, train_loss, '-', label='Train Loss')
    plt.plot(t_epochs, valid_loss, '-', label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Negative Rate)')
    plt.legend(loc='best')
    # plt.title('Train Loss vs Validation Loss')
    plt.grid()
    # plt.show()
    plt.savefig(f'Loss_{name}.jpg')
    plt.close()


def train_nn(epochs, H_train, H_valid, batch_size, model, K, optimizer, scheduler):
    """
    :param epochs: Number of training epochs
    :param H_train: Training channels
    :param H_valid: Validation channels
    :param batch_size: Number of training channels in each batch
    :param model: ANN
    :param K: Number of PGD iterations for forward method
    :param optimizer: Optimizer for learnable parameters
    :param scheduler: Optimizer's learning rate scheduler
    :return: Train loss, Validation loss, State of the best model
    """
    best_min_rate = 0
    state = {}
    train_loss = []
    valid_loss = []
    t0 = time()
    for i in range(epochs):
        t2 = time()
        H_b = H_train[torch.randperm(H_train.size()[0])]
        for b in range(0, len(H_b), batch_size):
            data = H_b[b:b + batch_size]
            x_n_arr, p_k, min_rate_arr = model.forward(K, data)
            a_train = [model.min_rate(x_n, data) for x_n in x_n_arr]
            b_train = [sum(r) for r in a_train]
            c_train = [np.log2(2 + k) * b_train[k] for k in
                       range(len(b_train))]  # log(2+k) because we count from 0, not 1
            loss = -torch.sum(torch.stack(c_train)) / batch_size  # weighted loss to enhance the last PGD iterations

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # batch_min_rate = model.min_rate(p_k, data)
            # loss = -batch_min_rate.mean()
            # print(f"Epoch {i + 1}, Batch {int(b / batch_size) + 1}: Loss: {float(loss)}")  # (b / batch_size)

        # average training loss after every epoch
        _, p_k_t, min_rate_t_arr = model.forward(K, H_b)
        min_rate_t = model.min_rate(p_k_t, H_b)
        loss_t = -min_rate_t.mean().detach().numpy()
        train_loss.append(loss_t)
        # print(f'Epoch {i + 1} train loss = {loss_t}')

        # average validation loss after every epoch
        _, p_k_v, min_rate_v_arr = model.forward(K, H_valid)  # learned P_K and sum rate along PGD iterations
        min_rate_v = model.min_rate(p_k_v, H_valid)
        loss_v = -min_rate_v.mean().detach().numpy()

        print(f'Epoch {i + 1} validation loss = {loss_v}')
        if scheduler:
            scheduler.step()  # loss_v

        if -loss_v >= best_min_rate:
            print('Improvement')
            best_min_rate = -loss_v
            state['epoch'] = i + 1
            state['best_valid_rate'] = -loss_v
            state['parameters'] = model.steps

        valid_loss.append(loss_v)

        t3 = time()
        print(f'Epoch {i + 1} time is {(t3 - t2) / 60} minutes')

    t1 = time()
    print(f'Training and estimation time = {(t1 - t0) / 60} minutes')
    return train_loss, valid_loss, state


def rayleighFading(ch, N):
    """
    Generate Rayleigh flat-fading channel samples
    Parameters:
    ch: number of channels to generate
    N : number of samples to generate
    Returns:
    abs_h : Rayleigh flat fading samples
    """
    # 1 tap complex gaussian filter
    h_real = torch.rand(ch, N)
    h_imag = torch.rand(ch, N)
    h = torch.complex(h_real, h_imag) / np.sqrt(2)
    return h


def train_general_net(epochs, A_train, A_valid, batch_size, model, K, optimizer, scheduler, M, N, T):
    """

    :param epochs: num of epochs to train
    :param A_train: training data
    :param A_valid: validation data
    :param batch_size: how many channels to forward pass
    :param model: PGDNet
    :param K: how many forward iterations
    :param optimizer: step sizes optimizer
    :param scheduler: optimizer scheduler
    :param M: number of relays
    :param N: number of end users
    :param T: number of transmitters
    :return: losses and best step sizes
    """
    best_min_rate = 0
    state = {}
    train_loss_arr = []
    valid_loss_arr = []
    t0 = time()
    for i in range(epochs):
        t2 = time()
        A_b = A_train[torch.randperm(A_train.size()[0])]
        G_b = A_b[:, :T * M]
        H_b = A_b[:, T * M:]
        epoch_train_loss = 0
        epoch_valid_loss = 0
        count = 0
        train_loss = []

        for b in range(0, len(H_b), batch_size):
            count += 1
            G_data = G_b[b:b + batch_size].reshape(T, M)
            H_data = H_b[b:b + batch_size].reshape(M, N)
            x_n_arr, p_k, min_rate_arr = model.forward(K, G_data, H_data)

            a_train = [model.min_rate(G_data, H_data, x_n[-T:] if T > 1 else x_n[-1:], x_n[:-T]) for x_n in x_n_arr]
            b_train = [torch.sum(r[1]) for r in a_train]
            c_train = [np.log2(2 + k) * b_train[k] for k in range(len(b_train))]  # log(2+k) because we count from 0, not 1
            train_loss.append(-torch.sum(torch.stack(c_train)) / batch_size)  # weighted loss to enhance the last PGD iterations

            epoch_train_loss += model.min_rate(G_data, H_data, p_k[-T:] if T > 1 else p_k[-1:], p_k[:-T])[1].item()

            if count == 100:
                loss = torch.sum(torch.stack(train_loss)) / 100
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                count = 0
                train_loss = []

        # average training loss after every epoch
        epoch_train_loss /= -len(H_b)
        train_loss_arr.append(epoch_train_loss)

        # average validation loss after every epoch
        G_valid = A_valid[:, :T * M]
        H_valid = A_valid[:, T * M:]
        for b in range(0, len(H_valid), batch_size):
            G_vb = G_valid[b:b + batch_size].reshape(T, M)
            H_vb = H_valid[b:b + batch_size].reshape(M, N)
            x_n_arr, p_k, min_rate_arr = model.forward(K, G_vb, H_vb)
            epoch_valid_loss += model.min_rate(G_vb, H_vb, p_k[-T:] if T > 1 else p_k[-1:], p_k[:-T])[1].item()
        epoch_valid_loss /= -len(H_valid)

        print(f'Epoch {i + 1} validation loss = {epoch_valid_loss}')
        if scheduler:
            scheduler.step()  # loss_v

        if -epoch_valid_loss >= best_min_rate:
            print('Improvement')
            best_min_rate = -epoch_valid_loss
            state['epoch'] = i + 1
            state['best_valid_rate'] = -epoch_valid_loss
            state['parameters'] = model.steps

        valid_loss_arr.append(epoch_valid_loss)

        t3 = time()
        print(f'Epoch {i + 1} time is {(t3 - t2) / 60} minutes')

    t1 = time()
    print(f'Training and estimation time = {(t1 - t0) / 60} minutes')
    return train_loss_arr, valid_loss_arr, state


def create_normalized_tensor(m, n):
    A = torch.rand(m, n)
    norm_A = torch.norm(A, dim=1, keepdim=True)
    normalized_A = A / norm_A
    return normalized_A
