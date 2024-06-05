import torch
from torch.utils.data import DataLoader
from PGDNet_AUX import rayleighFading, create_normalized_tensor, BC_estimation, MAC_estimation, AWGN
from GNN_AUX import *
from GNN_Module import *
import pickle
import os



T = 1  # transmitters
M = 3  # relays
N = 3  # end-users
sigma = 1
w_std = sigma ** 0.5
h_std = 1
ch_train = 1000
ch_valid = 200
batch_size = 1
csi = False

power_allocations = create_normalized_tensor(T + M + N, N)
file = open(r"C:\Users\User\Desktop\MS.c\MS.c\research\results\General Net\M = 3, N = 3, T = 1, noise = 0.0 "
            r"dB\state.pickle", 'rb')
PGD_state = pickle.load(file)

A_train = PGD_state['A_train']
A_valid = PGD_state['A_valid']
if not csi:
    G_train = A_train[:, :T * M]
    H_train = A_train[:, T * M:]
    A_train_est = torch.zeros_like(A_train)
    for i in range(len(A_train)):
        G_data = G_train[i:i + batch_size].reshape(T, M)
        H_data = H_train[i:i + batch_size].reshape(M, N)
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
        A_train_est[i] = torch.cat((G_est, H_est.resize(1, M * N)), dim=1)

# Create a list of data samples
train_data = [(create_manet_graph(A_train[i], power_allocations, T, M, N), power_allocations, A_train[i] if csi else A_train_est[i]) for i in range(ch_train)]
valid_data = [(create_manet_graph(A_valid[i], power_allocations, T, M, N), power_allocations, A_valid[i]) for i in range(ch_valid)]

# Create the dataset instance
train_dataset = CommunicationDataset(train_data)
valid_dataset = CommunicationDataset(valid_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

in_feats = N  # Define the input feature size (power allocations)
hidden_feats = 300
out_feats = N  # Define the output feature size (power allocations)
model = GNNModel(in_feats, hidden_feats, out_feats)

# Perform the forward pass

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 50
train_loss_arr, valid_loss_arr, state = train_gnn(model, optimizer, epochs, batch_size, train_loader, valid_loader, T, M, N, sigma=sigma)
state['in feats'] = in_feats
state['hidden feats'] = hidden_feats
state['out feats'] = out_feats
state['p0'] = power_allocations

dir_path = f"C:\\Users\\User\\Desktop\\MS.c\\MS.c\\research\\results\\GNN\\Noisy CSI\\(T,M,N) = {(T, M, N)}, noise = {10 * np.log10(sigma)} dB, h_var={h_std ** 2} "
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
os.chdir(dir_path)
state_path = 'state.pickle'
with open(state_path, 'wb') as file:
    pickle.dump(state, file)
train_valid_loss(epochs, train_loss_arr, valid_loss_arr, f'(T,M,N) = {(T, M, N)}, noise = {10 * np.log10(sigma)} dB, '
                                                         f'h_var={h_std ** 2}')
