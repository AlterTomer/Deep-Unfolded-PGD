from torch.utils.data import DataLoader
from PGDNet_AUX import rayleighFading, create_normalized_tensor
from GNN_AUX import *
from GNN_Module import *
import pickle
import os


#
T = 1  # transmitters
M = 2  # relays
N = 2  # end-users
sigma = 1
ch_train = 1000  # 1000
ch_valid = 200
batch_size = 1

# phi = create_normalized_tensor(T, N)
# p = create_normalized_tensor(M, N)
power_allocations = create_normalized_tensor(T + M + N, N)
# power_allocations = torch.ones(T+M+N,N) * torch.tensor([0.5, np.sqrt(1 - (0.5 ** 2))])
# power_allocations = power_allocations.float()

# power_allocations = torch.cat([phi, p])

# A_train = torch.load('H_train53.pt')
# A_valid = torch.load('H_valid53.pt')
A_train = rayleighFading(ch_train, M * (T + N))
A_valid = rayleighFading(ch_valid, M * (T + N))

# Create a list of data samples
train_data = [(create_manet_graph(A_train[i], power_allocations, T, M, N), power_allocations, A_train[i]) for i in range(ch_train)]
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
# new_power_allocations = model(g)  # , torch.cat([p, phi])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 100
train_loss_arr, valid_loss_arr, state = train_gnn(model, optimizer, epochs, batch_size, train_loader, valid_loader, T, M, N, sigma=sigma)
state['in feats'] = in_feats
state['hidden feats'] = hidden_feats
model['out feats'] = out_feats
state['p0'] = power_allocations

dir_path = f"C:\\Users\\User\\Desktop\\MS.c\\MS.c\\research\\results\\GNN\\(T,M,N) = {(T, M, N)}, noise = {10 * np.log10(sigma)} dB test"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
os.chdir(dir_path)
state_path = 'state.pickle'
with open(state_path, 'wb') as file:
    pickle.dump(state, file)
train_valid_loss(epochs, train_loss_arr, valid_loss_arr, f'(T,M,N) = {(T, M, N)}, noise = {10 * np.log10(sigma)} dB')
