import torch
import dgl
import re
import math
import numpy as np
from time import time
import matplotlib.pyplot as plt


def create_manet_graph(channel_links, power_allocations, T, M, N):
    # Create the DGL graph object for your topology
    channel_links = channel_links.tolist()
    channel_links = torch.tensor([x for x in channel_links for _ in range(2)])
    graph = dgl.graph([])
    graph.add_nodes(T + M + N, data={'feat': power_allocations})
    # Add edges between transmitters and relays
    t_indices = []
    m_indices = []
    for t_idx in range(T):
        for m_idx in range(M):
            t_indices.append(t_idx)
            m_indices.append(T + m_idx)
    graph.add_edges(t_indices, m_indices)
    graph.add_edges(m_indices, t_indices)

    # Add edges between relays and end users
    m_indices = []
    n_indices = []
    for m_idx in range(M):
        for n_idx in range(N):
            m_indices.append(T + m_idx)
            n_indices.append(T + M + n_idx)
    graph.add_edges(m_indices, n_indices)
    graph.add_edges(n_indices, m_indices)
    # Set edge features to store the channel information (channel_links)
    graph.edata['feat'] = channel_links

    return graph


# ----------------------------------------------------------------------------------------------------------------------
def calc_relay_rate_bc(name, h, sigma, phi):
    """
    :param h: link gain between TX and the relay (float)
    :param sigma: link noise (float)
    :param phi: power allocation given by TX (dict - {'phi1': float, 'phi2': float, ..., 'phiN': float})
    :return: Rm1, Rm2, ..., RmN
    """
    phi = phi.clone()
    relay_match = re.search(r'\d+\.\d+|\d+', name)
    relay_number = float(relay_match.group()) if '.' in relay_match.group() else int(relay_match.group())
    r_dict = {}
    phi_dict = {}
    for j in range(phi.size()[1]):
        phi_dict[f'phi {j + 1}'] = phi[:, j]

    sort_phi = {k: v for k, v in sorted(phi_dict.items(), key=lambda item: item[1])}
    data = sort_phi.items()
    for key in sort_phi.keys():
        phi_match = re.search(r'\d+\.\d+|\d+', key)
        phi_number = float(phi_match.group()) if '.' in phi_match.group() else int(phi_match.group())
        smaller_phi = torch.tensor([value for phi, value in data if phi != key and phi < key])

        n = (abs(h[0][relay_number - 1]) ** 2) * sort_phi[key] ** 2
        if len(smaller_phi) > 0:
            d = sigma + (abs(h[0][relay_number - 1]) ** 2) * torch.sum(smaller_phi ** 2)
        else:  # the weakest message
            d = sigma

        sinr = n / d
        r_dict[f'r_{phi_number}'] = torch.log2(1 + sinr)

    return r_dict


# ----------------------------------------------------------------------------------------------------------------------
def calc_relay_rate_mac(name, h, sigma, phi, N):
    """

    :param h: link gains between all relays to this RX node (np array of floats)
    :param sigma: channel noise (float)
    :param phi: power allocation from each relay to each message (matrix of floats)
    :return: R1n, R2n, ..., RNn
    """
    p = phi.clone()
    user_match = re.search(r'\d+\.\d+|\d+', name)
    user_number = float(user_match.group()) if '.' in user_match.group() else int(user_match.group())
    g_names = [f'g{n + 1}_{user_number}' for n in range(N)]
    r_dict = {}

    h_vec = h[:, user_number - 1].unsqueeze(-1)
    p = p.to(h_vec.dtype)
    g_tensor = torch.abs(torch.matmul(p.T, h_vec) ** 2)
    g_list = list(zip(g_names, g_tensor))
    g_dict = {key: value for key, value in g_list}
    sort_g = {k: v for k, v in sorted(g_dict.items(), key=lambda item: item[1])}
    data = sort_g.items()

    for idx, key in enumerate(sort_g.keys()):
        g_match = re.search(r'\d+\.\d+|\d+', key)
        g_number = float(g_match.group()) if '.' in g_match.group() else int(g_match.group())
        smaller_g = torch.tensor([value for g, value in data if g != key and g < key])
        n = sort_g[key]
        if len(smaller_g) > 0:  # g_after
            d = sigma + torch.sum(smaller_g)  # g_after
        else:  # the weakest message
            d = sigma
        sinr = n / d
        r_dict[f'r_{g_number}'] = torch.log2(1 + sinr).float()

    return r_dict


# ----------------------------------------------------------------------------------------------------------------------
def calc_enduser_rate(name, h, sigma, p, N):
    """

    :param h: link gains between all relays to this RX node (np array of floats)
    :param sigma: channel noise (float)
    :param p: power allocation from each relay to each message (matrix of floats)
    :return: R1n, R2n, ..., RNn
    """
    p = p.clone()
    user_match = re.search(r'\d+\.\d+|\d+', name)
    user_number = float(user_match.group()) if '.' in user_match.group() else int(user_match.group())
    g_names = [f'g{n + 1}_{user_number}' for n in range(N)]
    r_dict = {}

    h_vec = h[:, user_number - 1].unsqueeze(-1)
    p = p.to(h_vec.dtype)
    g_tensor = torch.abs(torch.matmul(p.T, h_vec) ** 2)
    g_list = list(zip(g_names, g_tensor))
    g_dict = {key: value for key, value in g_list}
    sort_g = {k: v for k, v in sorted(g_dict.items(), key=lambda item: item[1])}
    data = sort_g.items()
    target_number = int(name.split()[1])

    index = None

    for i, element in enumerate(sort_g.keys()):
        number = int(element.split('_')[0][1:])
        if number == target_number:
            index = i
            break

    for idx, key in enumerate(sort_g.keys()):
        g_match = re.search(r'\d+\.\d+|\d+', key)
        g_number = float(g_match.group()) if '.' in g_match.group() else int(g_match.group())
        smaller_g = torch.tensor([value for g, value in data if g != key and g < key])
        n = sort_g[key]
        if len(smaller_g) > 0:  # g_after
            d = sigma + torch.sum(smaller_g)  # g_after
        else:  # the weakest message
            d = sigma
        sinr = n / d
        if idx >= index:
            r_dict[f'r_{g_number}'] = torch.log2(1 + sinr).float()
        else:
            r_dict[f'r_{g_number}'] = torch.tensor(float('inf'))

    return r_dict


# ----------------------------------------------------------------------------------------------------------------------
def min_rate(G, H, p, phi, sigma, relays, end_users, T, M, N):
    # -------------------- relays min rates -----------------------------
    relays_rates = {}
    for idx, relay in enumerate(relays):
        if T > 1:
            rate = calc_relay_rate_mac(name=relay, h=G, sigma=sigma, phi=phi, N=N)
        else:
            rate = calc_relay_rate_bc(name=relay, h=G, sigma=sigma, phi=phi)

        relays_rates[f'relay {idx + 1}'] = rate

    # -------------------- end users min rates -----------------------------
    end_users_rates = {}
    for idx, end_user in enumerate(end_users):
        rate = calc_enduser_rate(name=end_user, h=H, sigma=sigma, p=p, N=N)
        end_users_rates[f'end user {idx + 1}'] = rate

    # -------------------- min network rates ---------------------------------
    lowest_rate = math.inf
    lowest_tuple = None

    # Iterate over relay_rates dictionary
    for relay, relay_dict in relays_rates.items():
        for rate_key, rate_tensor in relay_dict.items():
            if rate_tensor.item() < lowest_rate:
                lowest_rate = rate_tensor.item()
                lowest_tuple = (rate_key, rate_tensor, relay)

    # Iterate over end_users_rates dictionary
    for end_user, end_user_dict in end_users_rates.items():
        for rate_key, rate_tensor in end_user_dict.items():
            if rate_tensor.item() < lowest_rate:
                lowest_rate = rate_tensor.item()
                lowest_tuple = (rate_key, rate_tensor, end_user)

    return lowest_tuple


# ----------------------------------------------------------------------------------------------------------------------

def train_gnn(model, optimizer, epochs, batch_size, train_loader, valid_loader, T, M, N, sigma):
    best_min_rate = 0
    state = {}
    train_loss_arr = []
    valid_loss_arr = []
    t0 = time()
    relays = [f'relay {m + 1}' for m in range(M)]
    end_users = [f'user {n + 1}' for n in range(N)]
    for i in range(epochs):
        t2 = time()
        epoch_train_loss = 0
        epoch_valid_loss = 0

        for idx, data in enumerate(train_loader):  # , user_power_allocations, channel_links
            graph = data[0]
            # power_allocations = graph.ndata['feat']
            channel_links = graph.edata['feat'][::2]
            new_power_allocations = model(graph)
            phi = new_power_allocations[0:T]
            p = new_power_allocations[T:T + M]
            G = channel_links[0:T * M].reshape(T, M)
            H = channel_links[T * M:].reshape(M, N)
            loss = -min_rate(G, H, p, phi, sigma, relays, end_users, T, M, N)[1]
            epoch_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # average training loss after every epoch
        epoch_train_loss /= len(train_loader)
        train_loss_arr.append(epoch_train_loss)

        # average validation loss after every epoch
        with torch.no_grad():
            for idx, data in enumerate(valid_loader):  # , user_power_allocations, channel_links
                graph = data[0]
                # power_allocations = graph.ndata['feat']
                channel_links = graph.edata['feat'][::2]
                new_power_allocations = model(graph)
                phi = new_power_allocations[0:T]
                p = new_power_allocations[T:T + M]
                G = channel_links[0:T * M].reshape(T, M)
                H = channel_links[T * M:].reshape(M, N)
                loss = -min_rate(G, H, p, phi, sigma, relays, end_users, T, M, N)[1]
                epoch_valid_loss += loss.item()
        epoch_valid_loss /= len(valid_loader)

        print(f'Epoch {i + 1} validation loss = {epoch_valid_loss}')

        if -epoch_valid_loss >= best_min_rate:
            print('Improvement')
            best_min_rate = -epoch_valid_loss
            state['epoch'] = i + 1
            state['best_valid_rate'] = -epoch_valid_loss
            state['parameters'] = model.state_dict()
            state['T,M,N'] = (T, M, N)


        valid_loss_arr.append(epoch_valid_loss)

        t3 = time()
        print(f'Epoch {i + 1} time is {(t3 - t2) / 60} minutes')

    t1 = time()
    print(f'Training and estimation time = {(t1 - t0) / 60} minutes')
    return train_loss_arr, valid_loss_arr, state


# ----------------------------------------------------------------------------------------------------------------------
def collate(batch):
    graphs, user_power_allocations, channel_links = zip(*batch)
    batched_graph = dgl.batch(graphs)
    graph_sizes = [g.number_of_nodes() for g in graphs]

    return batched_graph, torch.stack(user_power_allocations), torch.stack(channel_links), graph_sizes


# ----------------------------------------------------------------------------------------------------------------------

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
    plt.grid()
    # plt.show()
    plt.savefig(f'Loss_{name}.jpg')
    plt.close()
