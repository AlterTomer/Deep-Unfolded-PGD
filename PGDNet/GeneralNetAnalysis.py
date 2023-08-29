import torch
from math import sqrt
from PGDNet_AUX import *
from GeneralNet import PGDNet
import torch.optim as optim
import pickle
import os

T = 2  # transmitters
M = 3  # relays
N = 3  # end-users
ch_train = 1000  # 1000
ch_valid = 200


phi = create_normalized_tensor(T, N)
p = create_normalized_tensor(M, N)
# phi = torch.ones(T, N) * sqrt(1 / N)
# p = torch.ones(M, N) * sqrt(1 / N)

K = 40
steps = torch.tensor([0.01] * K, requires_grad=True)
sigma = 1
lr = 1e-3
betas = (0.6, 0.8)

model = PGDNet(T, M, N, phi, p, steps, sigma)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, betas=betas)
epochs = 100
batch_size = 1
scheduler = False



A_train = rayleighFading(ch_train, M * (T + N))
A_valid = rayleighFading(ch_valid, M * (T + N))


# PGDNet training
train_loss, valid_loss, state = train_general_net(epochs, A_train, A_valid, batch_size, model, K, optimizer, scheduler,
                                                  M, N, T)
state['learning rate'] = lr
state['betas'] = betas
state['p0'] = p
state['phi0'] = phi
state['A_train'] = A_train
state['A_valid'] = A_valid
# -------------------------------------- Loss Graph and Data Saving ----------------------------------------------------
dir_path = f"C:\\Users\\User\\Desktop\\MS.c\\MS.c\\research\\results\\General Net\\M = {M}, N = {N}, T = {T}, noise = {10 * np.log10(sigma)} dB"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
os.chdir(dir_path)

name = f'M = {M}, N = {N}'
train_valid_loss(epochs, train_loss, valid_loss, name=name)

state_path = 'state.pickle'
with open(state_path, 'wb') as file:
    pickle.dump(state, file)
# -------------------------------------- Valid Min Rate Using Best Model -----------------------------------------------


best_model = PGDNet(T, M, N, phi, p, state['parameters'], sigma)

iters = 2000
step_sizes = torch.tensor([0.01] * iters, requires_grad=False)
classical_model = PGDNet(T, M, N, phi, p, step_sizes, sigma)

G_valid = A_valid[:, :T * M]
H_valid = A_valid[:, T * M:]
mean_valid_learn = np.zeros((len(A_valid), K))
mean_valid_classical = np.zeros((len(A_valid), iters))


for i in range(len(A_valid)):
    x_n_arr, p_k, min_rate_arr = best_model.forward(K, G_valid[i:i + batch_size].reshape(T, M),
                                                    H_valid[i:i + batch_size].reshape(M, N))

    x_n_arr_classical, p_k_classical, min_rate_arr_classical = classical_model.forward(iters, G_valid[i:i + batch_size].reshape(T, M),
                                                                                       H_valid[
                                                                                       i:i + batch_size].reshape(M, N))
    mean_valid_learn[i] = np.reshape(min_rate_arr.detach().numpy(), -1)
    mean_valid_classical[i] = np.reshape(min_rate_arr_classical.detach().numpy(), -1)

mean_valid_learn = np.mean(mean_valid_learn, axis=0)
mean_valid_classical = np.mean(mean_valid_classical, axis=0)

# ---------------------------- Classical vs. PGDNet --------------------------------------------------------------------

tc = np.linspace(start=0, stop=iters, num=iters)
tl = np.linspace(start=0, stop=K, num=K)

plt.figure(figsize=(8, 6), dpi=80)
plt.plot(tc, mean_valid_classical, label='Classic PGD iterations', linestyle='dotted')
plt.plot(tl, mean_valid_learn, label='PGDNet iterations', linestyle='solid')

plt.xlabel("Iterations")
plt.ylabel("Min Rate")
plt.grid()
plt.legend(loc='best')
plt.savefig('mean min rate pgd.jpeg')
plt.close()
print('done')
print('****************************************************************************')
