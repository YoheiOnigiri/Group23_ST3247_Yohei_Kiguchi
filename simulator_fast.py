import numpy as np
from numba import njit

# This function simulates the SIR model with rewiring on a random graph,
# and computes summary statistics.
@njit
def simulate_fast(beta, gamma, rho, N=200, p_edge=0.05, n_infected0=5, T=200):
    adj = np.zeros((N, N), dtype=np.bool_)
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.random() < p_edge:
                adj[i, j] = True
                adj[j, i] = True

# 0 = susceptible, 1 = infected, 2 = recovered
    state = np.zeros(N, dtype=np.int8)
    infected_chosen = 0
    while infected_chosen < n_infected0:
        idx = np.random.randint(N)
        if state[idx] == 0:
            state[idx] = 1
            infected_chosen += 1

# Arrays to store summary statistics
    infected_fraction = np.zeros(T + 1, dtype=np.float64)
    rewire_counts = np.zeros(T + 1, dtype=np.int64)

# Initial summary statistic at time 0    
    inf_count_initial = 0
    for i in range(N):
        if state[i] == 1:
            inf_count_initial += 1
    infected_fraction[0] = inf_count_initial / N

# Main simulation loop
    for t in range(1, T + 1):
        
        new_infections = np.zeros(N, dtype=np.bool_)
        for i in range(N):
            if state[i] == 1:
                for j in range(N):
                    if adj[i, j] and state[j] == 0:
                        if np.random.random() < beta:
                            new_infections[j] = True

# Update states for new infections
        for j in range(N):
            if new_infections[j]:
                state[j] = 1

        for i in range(N):
            if state[i] == 1:
                if np.random.random() < gamma:
                    state[i] = 2

# Identify SI edges for potential rewiring
        si_s = np.zeros(N * N, dtype=np.int32)
        si_i = np.zeros(N * N, dtype=np.int32)
        si_count = 0

# Collect all SI edges
        for i in range(N):
            if state[i] == 0:
                for j in range(N):
                    if adj[i, j] and state[j] == 1:
                        si_s[si_count] = i
                        si_i[si_count] = j
                        si_count += 1

# Rewire SI edges with probability rho
        rewire_count = 0
        for idx in range(si_count):
            if np.random.random() < rho:
                s_node = si_s[idx]
                i_node = si_i[idx]

                if not adj[s_node, i_node]:
                    continue

                adj[s_node, i_node] = False
                adj[i_node, s_node] = False

                candidates = np.zeros(N, dtype=np.int32)
                cand_count = 0
                for k in range(N):
                    if k != s_node and not adj[s_node, k]:
                        candidates[cand_count] = k
                        cand_count += 1

                if cand_count > 0:
                    new_partner = candidates[np.random.randint(cand_count)]
                    adj[s_node, new_partner] = True
                    adj[new_partner, s_node] = True
                    rewire_count += 1

# Compute summary statistics at time t
        inf_count = 0
        for i in range(N):
            if state[i] == 1:
                inf_count += 1
        infected_fraction[t] = inf_count / N
        rewire_counts[t] = rewire_count

# Compute degree histogram (capped at 30)
    degree_histogram = np.zeros(31, dtype=np.int64)
    for i in range(N):
        deg = 0
        for j in range(N):
            if adj[i, j]:
                deg += 1
        if deg > 30:
            deg = 30
        degree_histogram[deg] += 1

    return infected_fraction, rewire_counts, degree_histogram