data {
  int<lower=1> N;                   
  matrix[N, 4] sim;       //1 - targ, 2 - cohort or rhyme or dist, 3 - dist, 4  - dist    
  array[N] int<lower=1> timestep;         
  array[N] simplex[4] y;            
  
  int<lower=1> num_groups;      // 5 : 1=First Intact, 2=First Noisy, 3=Noisy after Intact, 4=Intact after Noisy Unexplained, 5=Intact after Noisy Explained  
  array[N] int<lower=1, upper=num_groups> group_id; 
}

parameters {
  vector[num_groups] log_tau;             
  real log_phi;            
}

model {
  // Priors
  log_tau ~ normal(0, 1);
  log_phi ~ normal(log(50), 1);  

  vector[4] log_prior;
  vector[4] log_posterior_unnorm;
  vector[4] posterior;
  vector[num_groups] tau = exp(log_tau);
  real phi = exp(log_phi);
  
  // Fit the model to find the best tau for each group
  for (n in 1:N) {
    if (timestep[n] == 1) {
      log_prior = rep_vector(log(0.25), 4);
    }
    
    for (i in 1:4) {
      // Use the specific tau for this row's block type
      // old version: likelihood[i] = exp(sim[n, i] / tau[group_id[n]]);
      log_posterior_unnorm[i] = (sim[n, i] / tau[group_id[n]]) + log_prior[i];
    }
    
    // old version: posterior = likelihood .* current_prior;
    // old version: posterior = posterior / sum(posterior);
    posterior = softmax(log_posterior_unnorm);
    
    y[n] ~ dirichlet(posterior * phi);
    log_prior = log(posterior);
  }
}

generated quantities {
  // --- 1. GENERATE PREDICTIONS & LOG-LIKELIHOOD ---
  matrix[N, 4] predicted_posterior;
  array[N] simplex[4] y_rep; // predicted fixation props
  vector[N] log_lik;  // store the pointwise log-likelihood for LOO-CV
  vector[num_groups] tau = exp(log_tau);
  real phi = exp(log_phi);
  
  { // Use a local block to hide temporary loop variables from the output
    vector[4] log_prior_g;
    vector[4] log_posterior_unnorm_g;
    vector[4] posterior_g;
    
    
    for (n in 1:N) {
      if (timestep[n] == 1) {
        log_prior_g = rep_vector(log(0.25), 4);
      }
      
      for (i in 1:4) {
        // Use the specific tau for this row's experimental block
        //likelihood_g[i] = exp(sim[n, i] / tau[group_id[n]]);
        log_posterior_unnorm_g[i] = (sim[n, i] / tau[group_id[n]]) + log_prior_g[i];
      }
      
      //posterior_g = likelihood_g .* current_prior_g;
      //posterior_g = posterior_g / sum(posterior_g);
      posterior_g = softmax(log_posterior_unnorm_g);
      
      // Save the 4 probabilities to our output matrix
      for (i in 1:4) {
        predicted_posterior[n, i] = posterior_g[i];
      }
      
      // Save simulated gaze props
      y_rep[n] = dirichlet_rng(posterior_g * phi);
      
      // Calculate the pointwise log-likelihood of the observed data (y[n])
      // given the predicted posterior and the precision parameter (phi)
      log_lik[n] = dirichlet_lpdf(y[n] | posterior_g * phi);
      
      // Pass posterior forward to the next timestep
      log_prior_g = log(posterior_g);
      
    }
  }

  // --- 2. HYPOTHESIS TESTING CONTRASTS ---
  // (1=First Intact, 2=First Noisy, 3=Noisy after Intact, 
  // 4=Intact after Noisy Unexplained, 5=Intact after Noisy Explained)
  
  real initial_noise_effect = tau[2] - tau[1];
  real initial_noise_effect_log = log_tau[2] - log_tau[1];
  real noise_residual_effect = tau[4] - tau[1];
  real noise_residual_effect_log = log_tau[4] - log_tau[1];
  real noise_contrast_effect = tau[3] - tau[2];
  real noise_contrast_effect_log = log_tau[3] - log_tau[2];
  real explain_away_effect = tau[5] - tau[4]; 
  real explain_away_effect_log = log_tau[5] - log_tau[4]; 
  real explained_noise_residual_effect = tau[5] - tau[1]; 
  real explained_noise_residual_effect_log = log_tau[5] - log_tau[1];
   
}
