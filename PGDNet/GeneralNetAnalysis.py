from math import sqrt
import numpy as np
import torch
from PGDNet_AUX import *
from GeneralNet import PGDNet
import torch.optim as optim
import pickle
import os

# MANET topology and channels
T = 1  # transmitters
M = 3  # relays
N = 3  # end-users
sigma = 1  # noise variance
w_std = sqrt(sigma)
h_std = 1  # channel std
ch_train = 1000
ch_valid = 200

# Initial power allocation
phi = create_normalized_tensor(T, N)
p = create_normalized_tensor(M, N)

# Initial step sizes
K = 40
steps = torch.tensor([0.001] * K, requires_grad=True)  # 0.01

# ADAM optimizer parameters
lr = 1e-3
betas = (0.6, 0.8)

model = PGDNet(T, M, N, phi, p, steps, sigma)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
epochs = 50
batch_size = 1
scheduler = False

# Channels generation
A_train = rayleighFading(ch_train, M * (T + N), h_std)
A_valid = rayleighFading(ch_valid, M * (T + N), h_std)

csi = True

# PGDNet training
train_loss, valid_loss, state = train_general_net(epochs, A_train, A_valid, batch_size, model, K, optimizer, scheduler,
                                                  M, N, T, csi, h_std, w_std)
state['ADAM learning rate'] = lr
state['betas'] = betas
state['p0'] = p
state['phi0'] = phi
state['A_train'] = A_train
state['A_valid'] = A_valid
state['T'] = T
state['M'] = M
state['N'] = N
state['K'] = K
state['sigma'] = sigma
state['h_std'] = h_std
# -------------------------------------- Loss Graph and Data Saving ----------------------------------------------------
if csi:
    dir_path = f"C:\\Users\\User\\Desktop\\MS.c\\MS.c\\research\\results\\General Net\\(T,M,N)={(T, M, N)} " \
               f"noise = {10 * np.log10(sigma)} dB, h_var={h_std ** 2}"
else:
    dir_path = f"C:\\Users\\User\\Desktop\\MS.c\\MS.c\\research\\results\\General Net estimation\\(T,M,N)={(T, M, N)} "\
               f"noise = {10 * np.log10(sigma)} dB, h_var={h_std ** 2}"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
os.chdir(dir_path)

name = f'M = {M}, N = {N}'
train_valid_loss(epochs, train_loss, valid_loss, name=name)

state_path = 'state.pickle'
with open(state_path, 'wb') as file:
    pickle.dump(state, file)

# -------------------------------------- Valid Min Rate Using Best Model -----------------------------------------------
state_path = r"C:\Users\User\Desktop\MS.c\MS.c\research\results\General Net\M = 3, N = 3, T = 1, noise = 0.0 " \
             r"dB\state.pickle "
file = open(state_path, 'rb')
state = pickle.load(file)
best_model = PGDNet(T, M, N, phi, p, state['parameters'], sigma)
iters = 2000
step_sizes = torch.tensor([0.001] * iters, requires_grad=False)
phi = state['phi0']
p = state['p0']
classical_model = PGDNet(T, M, N, phi, p, step_sizes, sigma)

A_valid = state['A_valid']
G_valid = A_valid[:, :T * M]
H_valid = A_valid[:, T * M:]

mean_valid_learn = np.zeros((len(A_valid), K))
mean_valid_classical = np.zeros((len(A_valid), iters))

for i in range(len(A_valid)):
    if csi:
        x_n_arr, p_k, min_rate_arr = best_model.forward(K, G_valid[i:i + batch_size].reshape(T, M),
                                                        H_valid[i:i + batch_size].reshape(M, N))

        x_n_arr_classical, p_k_classical, min_rate_arr_classical = classical_model.forward(iters, G_valid[
                                                                                                  i:i + batch_size].reshape(
            T, M),
                                                                                           H_valid[
                                                                                           i:i + batch_size].reshape(M,
                                                                                                                     N))
    else:
        G_data = G_valid[i:i + batch_size].reshape(T, M)
        H_data = H_valid[i:i + batch_size].reshape(M, N)
        pilot_mat = torch.fft.fft(torch.eye(M)) / np.sqrt(M)
        H_est = torch.zeros_like(H_data)
        if T == 1:
            G_est = torch.reshape(BC_estimation(pilot_mat=pilot_mat, h=G_data.reshape((-1, 1)), h_std=h_std,
                                                w=AWGN(batch_size, M, w_std ** 2).T, w_std=w_std), G_data.size())
        else:
            G_est = torch.zeros_like(G_data)
            for g in range(len(G_data)):
                G_est[g] = torch.reshape(
                    MAC_estimation(pilot_mat=pilot_mat, h=G_data[g].reshape((-1, 1)), h_std=h_std,
                                   w=AWGN(batch_size, M, w_std ** 2).T, w_std=w_std), G_data[g].size())

        for h in range(H_data.size()[1]):
            H_est[:, h] = torch.reshape(
                MAC_estimation(pilot_mat=pilot_mat, h=H_data[:, h].reshape((-1, 1)), h_std=h_std,
                               w=AWGN(batch_size, M, w_std ** 2).T, w_std=w_std), H_data[:, h].size())

        x_n_arr, p_k, min_rate_arr = best_model.forward(K, G_est, H_est)

        x_n_arr_classical, p_k_classical, min_rate_arr_classical = classical_model.forward(iters, G_est, H_est)

    mean_valid_learn[i] = np.reshape(min_rate_arr.detach().numpy(), -1)
    mean_valid_classical[i] = np.reshape(min_rate_arr_classical.detach().numpy(), -1)

mean_valid_learn = np.mean(mean_valid_learn, axis=0)
mean_valid_classical = np.mean(mean_valid_classical, axis=0)

# ---------------------------- Classical vs. PGDNet --------------------------------------------------------------------

tc = np.linspace(start=0, stop=iters, num=iters)
tl = np.linspace(start=0, stop=K, num=K)
plt.figure(figsize=(8, 6), dpi=80)
plt.plot(tc, monotone_arr(mean_valid_classical), label='Classic PGD iterations', linestyle='dotted')
plt.plot(tl, monotone_arr(mean_valid_learn), label='PGDNet iterations', linestyle='solid')

plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Min Rate", fontsize=14)
plt.grid()
plt.legend(loc='best')
plt.savefig(f'mean min rate pgd {"CSI" if csi else "noisy"} 133.jpeg')
plt.close()
print('done')
print('****************************************************************************')

