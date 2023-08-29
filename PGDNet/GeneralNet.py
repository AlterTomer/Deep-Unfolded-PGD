import torch
import torch.nn as nn
import numpy as np
import re
import math


class BC_Relay:
    """
    this class represents a relay node between TX and RX
    """

    def __init__(self, sigma, phi, N, name):
        """
        :param sigma: link noise (float)
        :param phi: power allocation given by TX (dict - {'phi1': float, 'phi2': float, ..., 'phiN': float})
        :param N: number of messages (int)
        :param name: relay's name
        """
        self.sigma = sigma
        self.phi = phi
        self.N = N
        self.name = name

    def calc_relay_rate(self, h, sigma, phi):
        """
        :param h: link gain between TX and the relay (float)
        :param sigma: link noise (float)
        :param phi: power allocation given by TX (dict - {'phi1': float, 'phi2': float, ..., 'phiN': float})
        :return: Rm1, Rm2, ..., RmN
        """
        phi = phi.clone()
        relay_match = re.search(r'\d+\.\d+|\d+', self.name)
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

    def dr_df(self, smaller_phi, g_m, phi_n, phi_t, sigma):
        """

        :param smaller_phi: all the TX power allocations that are smaller than the one of the nth message
        :param g_m: TX -> relay m link gain
        :param phi_n: power allocation of TX to message n
        :param phi_t: derivative variable
        :param sigma: TX -> relay noise
        :return: partial derivative dRmn/dphi_t
        """
        nm, p_n = phi_n
        tm, p_t = phi_t
        if tm > nm:
            return 0

        elif tm == nm:
            all_phi = torch.cat((smaller_phi, p_n), dim=0)
            n = 2 * abs(g_m ** 2) * p_n
            d = sigma + (abs(g_m ** 2) * torch.sum(all_phi ** 2))
            return n / d

        else:  # tm < nm
            all_phi = torch.cat((smaller_phi, p_n), dim=0)
            n = -2 * abs(g_m ** 4) * (p_n ** 2) * p_t
            d = (sigma + (abs(g_m ** 2) * torch.sum(smaller_phi ** 2))) * (
                    sigma + (abs(g_m ** 2) * torch.sum(all_phi ** 2)))
            return n / d

    def grad_relay(self, h, sigma, phi, N):
        """
        :param h: link gain between TX and the relay (float)
        :param sigma: link noise (float)
        :param phi: power allocation given by TX (dict - {'phi1': float, 'phi2': float, ..., 'phiN': float})
        :param N: number of transmitted signals
        :return: grad_phi(R1m), grad_phi(R2m), ..., grad_phi(RNm)
        """
        phi = phi.clone()
        grad_dict = {}
        phi_dict = {}
        relay_match = re.search(r'\d+\.\d+|\d+', self.name)
        relay_number = float(relay_match.group()) if '.' in relay_match.group() else int(relay_match.group())
        for j in range(phi.size()[1]):  # for j in range(phi.size()[1]):
            phi_dict[f'phi {j + 1}'] = phi[:, j]  # phi[i, j]

        sort_phi = {k: v for k, v in sorted(phi_dict.items(), key=lambda item: item[1])}
        data = sort_phi.items()
        g_m = h[0][relay_number - 1]
        for idx, key in enumerate(sort_phi.keys()):
            grad = torch.zeros(N, 1)
            phi_match = re.search(r'\d+\.\d+|\d+', key)
            phi_number = float(phi_match.group()) if '.' in phi_match.group() else int(phi_match.group())
            smaller_phi = torch.tensor([value for phi, value in data if
                                        phi != key and phi < key])  # all power allocations of the weaker messages
            phi_n = (idx, sort_phi[key])

            for i, element in enumerate(sort_phi.keys()):
                t_match = re.search(r'\d+\.\d+|\d+', element)
                t = float(t_match.group()) if '.' in t_match.group() else int(t_match.group())
                phi_t = (i, sort_phi[element])  # i --> t - 1
                grad[t - 1] = self.dr_df(smaller_phi, g_m, phi_n, phi_t, sigma)
            grad_dict[f'r_{phi_number}'] = (grad / np.log(2)).float()  # torch.tensor(grad / np.log(2)).float()

        return grad_dict


# ----------------------------------------------------------------------------------------------------------------------

class MAC_Relay:
    """
    this class represents a relay with multiple inputs
    """

    def __init__(self, sigma, p, N, name):
        """
        :param sigma: channel noise (float)
        :param p: power allocation from each relay to each message (matrix of floats)
        :param N: number of messages (int)
        :param name: user's name
        """
        self.sigma = sigma
        self.p = p
        self.N = N
        self.name = name

    def calc_relay_rate(self, h, sigma, phi):
        """

        :param h: link gains between all relays to this RX node (np array of floats)
        :param sigma: channel noise (float)
        :param phi: power allocation from each relay to each message (matrix of floats)
        :return: R1n, R2n, ..., RNn
        """
        p = phi.clone()
        user_match = re.search(r'\d+\.\d+|\d+', self.name)
        user_number = float(user_match.group()) if '.' in user_match.group() else int(user_match.group())
        g_names = [f'g{n + 1}_{user_number}' for n in range(self.N)]
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

    def dr_dp(self, g_after, h_sl, g_nl, g_tl, p_st, sigma):
        """

        :param g_after: all the g variables that are smaller than g_nl
        :param h_sl: link gain between relay s to end-user l
        :param g_nl: generalized power allocation for message n at end-user l
        :param g_tl: generalized power allocation for message t at end-user l
        :param p_st: power allocation of message t from relay s
        :param sigma: link noise between relay s to end-user l
        :return: partial derivative dRln/dps_t
        """

        if g_tl > g_nl:
            return 0

        elif g_tl == g_nl:
            if g_after.size()[0] == 0:
                all_g = g_nl
            else:
                all_g = torch.cat((g_after, g_nl), dim=0)
            n = 2 * abs(h_sl ** 2) * p_st
            d = sigma + torch.sum(all_g)
            return n / d

        else:  # tl < nl
            all_g = torch.cat((g_after, g_nl), dim=0)
            n = -2 * abs(h_sl ** 2) * p_st * g_nl
            d = (sigma + torch.sum(g_after)) * (sigma + torch.sum(all_g))
            return n / d

    def assure_unique(self, dict, epsilon=1e-10):
        new_dict = {}

        previous_value = None
        for key, value in dict.items():
            if value == previous_value:
                value = value + epsilon
            new_dict[key] = value
            previous_value = value
        return new_dict

    def grad_relay(self, h, sigma, phi, N):
        p = phi.clone()
        a, b = p.size()
        user_match = re.search(r'\d+\.\d+|\d+', self.name)
        user_number = float(user_match.group()) if '.' in user_match.group() else int(user_match.group())
        g_names = [f'g{n + 1}_{user_number}' for n in range(self.N)]
        grad_dict = {}

        h_vec = h[:, user_number - 1].unsqueeze(-1)
        p = p.to(h_vec.dtype)
        g_tensor = torch.abs(torch.matmul(p.T, h_vec) ** 2)
        g_list = list(zip(g_names, g_tensor))
        g_dict = {key: value for key, value in g_list}

        sort_g = self.assure_unique({k: v for k, v in sorted(g_dict.items(), key=lambda item: item[1])})
        data = sort_g.items()

        g_arr = list(sort_g.values())
        for idx, key in enumerate(sort_g.keys()):
            grad = torch.zeros(a, b)
            g_match = re.search(r'\d+\.\d+|\d+', key)
            g_number = float(g_match.group()) if '.' in g_match.group() else int(g_match.group())
            smaller_g = torch.tensor([value for g, value in data if g != key and g < key])  # all power allocations of the weaker messages
            g_nl = sort_g[key]
            for t in range(a * b):
                row = t // b
                col = t % b
                h_sl = h_vec[row]
                g_tl = g_arr[col]
                p_st = p[row][col]
                grad[row][col] = self.dr_dp(smaller_g, h_sl, g_nl, g_tl, p_st, sigma)  # g_after

            grad_dict[f'r_{g_number}'] = (grad / np.log(2)).float()  # torch.tensor(grad / np.log(2)).float()

        return grad_dict


# ----------------------------------------------------------------------------------------------------------------------

class EndUser:
    """
    this class represents an end-user (RX node)
    """

    def __init__(self, sigma, p, N, name):
        """
        :param sigma: channel noise (float)
        :param p: power allocation from each relay to each message (matrix of floats)
        :param N: number of messages (int)
        :param name: user's name
        """
        self.sigma = sigma
        self.p = p
        self.N = N
        self.name = name

    def calc_enduser_rate(self, h, sigma, p):
        """

        :param h: link gains between all relays to this RX node (np array of floats)
        :param sigma: channel noise (float)
        :param p: power allocation from each relay to each message (matrix of floats)
        :return: R1n, R2n, ..., RNn
        """
        p = p.clone()
        user_match = re.search(r'\d+\.\d+|\d+', self.name)
        user_number = float(user_match.group()) if '.' in user_match.group() else int(user_match.group())
        g_names = [f'g{n + 1}_{user_number}' for n in range(self.N)]
        r_dict = {}

        h_vec = h[:, user_number - 1].unsqueeze(-1)
        p = p.to(h_vec.dtype)
        g_tensor = torch.abs(torch.matmul(p.T, h_vec) ** 2)
        g_list = list(zip(g_names, g_tensor))
        g_dict = {key: value for key, value in g_list}
        sort_g = {k: v for k, v in sorted(g_dict.items(), key=lambda item: item[1])}
        data = sort_g.items()
        target_number = int(self.name.split()[1])

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

    def dr_dp(self, g_after, h_sl, g_nl, g_tl, p_st, sigma):
        """

        :param g_after: all the g variables that are smaller than g_nl
        :param h_sl: link gain between relay s to end-user l
        :param g_nl: generalized power allocation for message n at end-user l
        :param g_tl: generalized power allocation for message t at end-user l
        :param p_st: power allocation of message t from relay s
        :param sigma: link noise between relay s to end-user l
        :return: partial derivative dRln/dps_t
        """

        if g_tl > g_nl:
            return 0

        elif g_tl == g_nl:
            if g_after.size()[0] == 0:
                all_g = g_nl
            else:
                all_g = torch.cat((g_after, g_nl), dim=0)
            n = 2 * abs(h_sl ** 2) * p_st
            d = sigma + torch.sum(all_g)
            return n / d

        else:  # tl < nl
            all_g = torch.cat((g_after, g_nl), dim=0)
            n = -2 * abs(h_sl ** 2) * p_st * g_nl
            d = (sigma + torch.sum(g_after)) * (sigma + torch.sum(all_g))
            return n / d

    def assure_unique(self, dict, epsilon=1e-10):
        new_dict = {}

        previous_value = None
        for key, value in dict.items():
            if value == previous_value:
                value = value + epsilon
            new_dict[key] = value
            previous_value = value
        return new_dict

    def grad_enduser(self, h, sigma, p, N):
        p = p.clone()
        a, b = p.size()
        user_match = re.search(r'\d+\.\d+|\d+', self.name)
        user_number = float(user_match.group()) if '.' in user_match.group() else int(user_match.group())
        g_names = [f'g{n + 1}_{user_number}' for n in range(self.N)]
        grad_dict = {}

        h_vec = h[:, user_number - 1].unsqueeze(-1)
        p = p.to(h_vec.dtype)
        g_tensor = torch.abs(torch.matmul(p.T, h_vec) ** 2)
        g_list = list(zip(g_names, g_tensor))
        g_dict = {key: value for key, value in g_list}

        sort_g = self.assure_unique({k: v for k, v in sorted(g_dict.items(), key=lambda item: item[1])})
        data = sort_g.items()

        g_arr = list(sort_g.values())
        for idx, key in enumerate(sort_g.keys()):
            grad = torch.zeros(a, b)
            g_match = re.search(r'\d+\.\d+|\d+', key)
            g_number = float(g_match.group()) if '.' in g_match.group() else int(g_match.group())
            smaller_g = torch.tensor(
                [value for g, value in data if g != key and g < key])  # all power allocations of the weaker messages
            g_nl = sort_g[key]
            for t in range(a * b):
                row = t // b
                col = t % b
                h_sl = h_vec[row]
                g_tl = g_arr[col]
                p_st = p[row][col]
                grad[row][col] = self.dr_dp(smaller_g, h_sl, g_nl, g_tl, p_st, sigma)  # g_after

            grad_dict[f'r_{g_number}'] = (grad / np.log(2)).float()  # torch.tensor(grad / np.log(2)).float()

        return grad_dict


# ----------------------------------------------------------------------------------------------------------------------

class PGDNet(nn.Module):
    """
    this class represent the entire communication network
    """

    def __init__(self, T, M, N, phi, p, steps, sigma):
        """
        :param T: number of transmitters
        :param M: number of relays
        :param N: number of end-users (number of messages)
        :param phi: power allocation transmitters -> relays
        :param p: power allocation relays -> end-users
        :param steps: PGD step sizes
        :param sigma: noise power

        """
        super(PGDNet, self).__init__()
        self.T = T
        self.M = M
        self.N = N
        self.phi = phi
        self.p = p
        self.steps = nn.Parameter(steps, requires_grad=True)
        self.sigma = sigma
        self.relay_names = [f'relay {m + 1}' for m in range(M)]
        self.user_names = [f'user {n + 1}' for n in range(N)]
        self.relays = self.create_relays()
        self.end_users = self.create_endusers()

    def create_relays(self):
        relays = []
        if self.T > 1:
            for m in range(self.M):
                relays.append(MAC_Relay(self.sigma, self.phi, self.N, self.relay_names[m]))
        else:
            for m in range(self.M):
                relays.append(BC_Relay(self.sigma, self.phi, self.N, self.relay_names[m]))

        return relays

    def create_endusers(self):
        end_users = []
        for n in range(self.N):
            end_users.append(EndUser(self.sigma, self.p, self.N, self.user_names[n]))

        return end_users

    def calc_rates(self, G, H, phi, p):
        relays = self.relays
        end_users = self.end_users

        # -------------------- relays min rates -----------------------------
        relays_rates = {}
        for idx, relay in enumerate(relays):
            rate = relay.calc_relay_rate(h=G, sigma=self.sigma, phi=phi)
            relays_rates[f'relay {idx + 1}'] = rate

        # -------------------- end users min rates -----------------------------
        end_users_rates = {}
        for idx, end_user in enumerate(end_users):
            rate = end_user.calc_enduser_rate(h=H, sigma=self.sigma, p=p)
            end_users_rates[f'end user {idx + 1}'] = rate
        # -------------------- min network rates ---------------------------------
        rates = {}

        for relay_key in relays_rates.keys():
            relay_rate = relays_rates[relay_key]
            for rate_key in relay_rate.keys():
                rate_value = relay_rate[rate_key]
                if rate_key not in rates or rate_value < rates[rate_key]:
                    rates[rate_key] = rate_value

        for end_user_key in end_users_rates.keys():
            end_user_rate = end_users_rates[end_user_key]
            for rate_key in end_user_rate.keys():
                rate_value = end_user_rate[rate_key]
                if rate_key not in rates or rate_value < rates[rate_key]:
                    rates[rate_key] = rate_value
        return rates

    def min_rate(self, G, H, phi, p):
        relays = self.relays
        end_users = self.end_users

        # -------------------- relays min rates -----------------------------
        relays_rates = {}
        for idx, relay in enumerate(relays):
            rate = relay.calc_relay_rate(h=G, sigma=self.sigma, phi=phi)
            relays_rates[f'relay {idx + 1}'] = rate

        # -------------------- end users min rates -----------------------------
        end_users_rates = {}
        for idx, end_user in enumerate(end_users):
            rate = end_user.calc_enduser_rate(h=H, sigma=self.sigma, p=p)
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

    def calc_grad(self, G, H, phi, p):
        relays = self.relays
        end_users = self.end_users
        rate = self.min_rate(G, H, phi, p)

        # -------------------- relays grad ---------------------------------
        relays_grad = {}
        for idx, relay in enumerate(relays):
            grad = relay.grad_relay(h=G, sigma=self.sigma, phi=phi, N=self.N)
            relays_grad[f'relay {idx + 1}'] = grad

        # -------------------- end users grad ---------------------------------
        endusers_grad = {}
        for idx, end_user in enumerate(end_users):
            grad = end_user.grad_enduser(h=H, sigma=self.sigma, p=p, N=self.N)
            endusers_grad[f'end user {idx + 1}'] = grad

        # ------------------- final grads according to min rates ---------------
        rate_type, _, device = rate

        # Fetch the gradient based on rate_type and device
        if device in endusers_grad:
            grad = endusers_grad[device][rate_type]
        elif device in relays_grad:
            grad = relays_grad[device][rate_type]
        else:
            grad = None

        # Return a tuple containing rate type, gradient, and device
        min_grad = (rate_type, grad, device)

        return min_grad

    def proj(self, x):
        eps = 1e-6
        if x.size()[0] == 1:
            row_norms = torch.norm(abs(x))
        else:
            row_norms = torch.norm(abs(x), dim=1)
        # x_proj = torch.max(torch.full_like(x, eps), x / row_norms.unsqueeze(-1))
        x[x < 0] = 0
        x_proj = x / row_norms.unsqueeze(-1)

        return x_proj

    def forward(self, num_iter, G, H):
        min_rate_arr = torch.zeros(num_iter, 1)
        phi_k = self.proj(self.phi)
        p_k = self.proj(self.p)

        T = self.phi.size()[0]
        M, N = self.p.size()

        x_n = torch.cat((p_k, phi_k))
        steps = self.steps
        x_n_arr = []

        for i in range(num_iter):
            min_rate_arr[i] = self.min_rate(G=G, H=H, phi=x_n[-T:] if T > 1 else x_n[-1:], p=x_n[:-T])[1]  # x_n[-T:] if T > 1 else x_n[-1:].unsqueeze(0)
            # PGD steps
            g = self.calc_grad(G=G, H=H, phi=x_n[-T:] if T > 1 else x_n[-1:], p=x_n[:-T])
            user = g[2]
            if 'relay' in user:
                grad_phi = g[1]
                if T == 1:
                    grad_phi = g[1].T
                grad_p = torch.zeros(M, N)
                grad = torch.cat((grad_p, grad_phi), dim=0)
            elif 'end user' in user:
                grad_phi = torch.zeros(T, N)
                grad_p = g[1]
                grad = torch.cat((grad_p, grad_phi), dim=0)

            y_n_1 = x_n - steps[i] * (-grad)
            x_n_1 = self.proj(y_n_1)
            x_n = x_n_1.clone()
            x_n_arr.append(x_n)
            # print(f'iter {i + 1} grad = {-grad}')

            if i == 0:
                if 'relay' in user:
                    phi_k = x_n[-T:] if T > 1 else x_n[-1:]
                elif 'end user' in user:
                    p_k = x_n[:-T]
            else:
                if min_rate_arr[i] > min_rate_arr[i - 1]:  # check if the PGD step improved the rate
                    if 'relay' in user:
                        phi_k = x_n[-T:] if T > 1 else x_n[-1:]  # update the improved power allocations
                    #
                    elif 'end user' in user:
                        p_k = x_n[:-T]  # update the improved power allocations

        return x_n_arr, torch.cat((p_k, phi_k)), min_rate_arr  # x_n --> p_k
