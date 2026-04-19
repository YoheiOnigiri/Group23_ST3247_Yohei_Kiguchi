import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from simulator_fast import simulate_fast
import time

def calc_lse_loglik(params, target_mu, target_cov, n_sims):
    """
    Calculate Synthetic Likelihood using LogSumExp (Log-space & Center of Mass version)
    """
    beta, gamma, rho = params
    
    # Boundary check for uniform prior distribution
    if not (0.05 <= beta <= 0.50 and 0.02 <= gamma <= 0.20 and 0.0 <= rho <= 0.8):
        return -np.inf
    
    logliks = np.zeros(n_sims)
    degrees_arr = np.arange(31)
    
    for s in range(n_sims):
        sim_infected, sim_rewires, sim_degrees = simulate_fast(beta, gamma, rho)
        
        # 1. Peak Infected Fraction (Keep linear)
        sim_peak = np.max(sim_infected)
        
        # 2. Total Rewires (Log-space)
        sim_rewires_tot = np.sum(sim_rewires)
        sim_log_rewires = np.log(sim_rewires_tot) if sim_rewires_tot > 0 else np.log(1e-6)
        
        # 3. Degree Variance (Log-space)
        sim_mean_deg = np.sum(degrees_arr * sim_degrees) / 200
        sim_var_deg = np.sum((degrees_arr**2) * sim_degrees) / 200 - sim_mean_deg**2
        sim_log_var_deg = np.log(sim_var_deg) if sim_var_deg > 0 else np.log(1e-6)
        
        # 4. Weighted Mean Time (Center of mass & Log-space)
        sum_inf = np.sum(sim_infected)
        if sum_inf > 0:
            times_arr = np.arange(len(sim_infected))
            sim_weighted_time = np.sum(times_arr * sim_infected) / sum_inf
        else:
            sim_weighted_time = 1e-6
        sim_log_time = np.log(sim_weighted_time)
        
        # Summary statistics vector aligned with the target scale
        sim_stats = np.array([sim_peak, sim_log_rewires, sim_log_var_deg, sim_log_time])
        
        try:
            ll = multivariate_normal.logpdf(sim_stats, mean=target_mu, cov=target_cov)
            logliks[s] = ll
        except np.linalg.LinAlgError:
            logliks[s] = -np.inf
            
    if np.all(np.isinf(logliks)):
        return -np.inf
    else:
        return logsumexp(logliks) - np.log(n_sims)


# def run_hybrid_adaptive_mcmc(init_params, target_mu, target_cov, stages, init_proposal_cov):
#     """
#     Adaptive MCMC manager across multiple stages (exploration and final sampling)
#     """
#     current_params = np.copy(init_params)
#     proposal_cov = np.copy(init_proposal_cov)
    
#     final_valid_chain = None

#     full_history = []
#     stage_boundaries = []
#     current_global_iter = 0
    
#     for stage in stages:
#         print(f"\n--- Starting {stage['name']} ---")
#         start_time = time.time()
        
#         n_iter = stage['iters']
#         n_sim = stage['n_sim']
#         inflation = stage['inflation']
        
#         # Inflate target covariance matrix to facilitate exploration
#         stage_target_cov = target_cov * inflation
        
#         chain = np.zeros((n_iter, 3))
#         chain[0] = current_params

#         full_history.append(current_params.copy())
#         current_global_iter += 1

#         accept_count = 0
        
#         current_loglik = calc_lse_loglik(current_params, target_mu, stage_target_cov, n_sim)
        
#         for i in range(1, n_iter):
#             # A. Sampling from proposal distribution
#             proposed_step = np.random.multivariate_normal(mean=np.zeros(3), cov=proposal_cov)
#             proposed_params = current_params + proposed_step
            
#             # B. Likelihood evaluation
#             proposed_loglik = calc_lse_loglik(proposed_params, target_mu, stage_target_cov, n_sim)
            
#             # C. Metropolis-Hastings acceptance step
#             log_ratio = proposed_loglik - current_loglik
            
#             if np.isfinite(proposed_loglik) and np.log(np.random.rand()) < log_ratio:
#                 current_params = proposed_params
#                 current_loglik = proposed_loglik
#                 accept_count += 1
                
#             chain[i] = current_params

#             full_history.append(current_params.copy())
#             current_global_iter += 1
            
#             if i % 1000 == 0:
#                 print(f"Iter: {i}/{n_iter} | Acceptance Rate: {accept_count/i:.2%}")
                
#         stage_boundaries.append(current_global_iter)
        
#         # D. Update proposal covariance matrix for the next stage (Adaptive)
#         burn_in = int(n_iter * 0.2)
#         valid_chain = chain[burn_in:]
        
#         # Optimal scaling based on Gelman et al. (1996)
#         proposal_cov = np.cov(valid_chain.T) * (2.38**2 / 3) 
#         proposal_cov += np.diag([1e-6, 1e-6, 1e-6]) # Tiny value for numerical stability
        
#         elapsed = time.time() - start_time
#         print(f"{stage['name']} completed in {elapsed:.1f} seconds.")
#         print(f"Final Acceptance Rate: {accept_count/n_iter:.2%}")
        
#         # Save the chain from the final stage
#         final_valid_chain = valid_chain
        
#     return np.array(full_history), stage_boundaries, final_valid_chain



def run_hybrid_adaptive_mcmc(init_params, target_mu, target_cov, stages, init_proposal_cov):
    """
    Adaptive MCMC manager across multiple stages (Sticky-Trap Proof / MCWM version)
    """
    current_params = np.copy(init_params)
    proposal_cov = np.copy(init_proposal_cov)
    
    final_valid_chain = None
    full_history = []
    stage_boundaries = []
    current_global_iter = 0
    
    for stage in stages:
        print(f"\n--- Starting {stage['name']} ---")
        start_time = time.time()
        
        n_iter = stage['iters']
        n_sim = stage['n_sim']
        inflation = stage['inflation']
        
        # Inflate target covariance matrix
        stage_target_cov = target_cov * inflation
        
        chain = np.zeros((n_iter, 3))
        chain[0] = current_params
        full_history.append(current_params.copy())
        current_global_iter += 1

        accept_count = 0
        
        for i in range(1, n_iter):
            # A. Sampling from proposal distribution
            proposed_step = np.random.multivariate_normal(mean=np.zeros(3), cov=proposal_cov)
            proposed_params = current_params + proposed_step
            
            # B. Likelihood evaluation
            # ★必殺技: 過去の「まぐれ当たり」をリセットするため、現在地も毎回フェアーに再評価する！
            fresh_current_loglik = calc_lse_loglik(current_params, target_mu, stage_target_cov, n_sim)
            proposed_loglik = calc_lse_loglik(proposed_params, target_mu, stage_target_cov, n_sim)
            
            # C. Metropolis-Hastings acceptance step
            log_ratio = proposed_loglik - fresh_current_loglik
            
            if np.isfinite(proposed_loglik) and np.log(np.random.rand()) < log_ratio:
                current_params = proposed_params
                accept_count += 1
                
            chain[i] = current_params
            full_history.append(current_params.copy())
            current_global_iter += 1
            
            if i % 1000 == 0:
                print(f"Iter: {i}/{n_iter} | Acceptance Rate: {accept_count/i:.2%}")
                
        stage_boundaries.append(current_global_iter)
        
        # D. Update proposal covariance matrix for the next stage (Adaptive)
        burn_in = int(n_iter * 0.2)
        valid_chain = chain[burn_in:]
        
        # ★安全装置: 最低でも10回動いていないと共分散が崩壊するため更新をスキップ
        if accept_count > 10:
            proposal_cov = np.cov(valid_chain.T) * (2.38**2 / 3) 
            proposal_cov += np.diag([1e-5, 1e-5, 1e-5]) 
        else:
            print(f"⚠️ Acceptance too low ({accept_count}). Keeping previous proposal_cov.")
        
        elapsed = time.time() - start_time
        print(f"{stage['name']} completed in {elapsed:.1f} seconds.")
        print(f"Final Acceptance Rate: {accept_count/n_iter:.2%}")
        
        final_valid_chain = valid_chain
        
    return np.array(full_history), stage_boundaries, final_valid_chain