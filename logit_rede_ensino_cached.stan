
    data {
      int<lower=0> N;          
      int<lower=0> K;          
      matrix[N, K] X;         
      array[N] int<lower=0, upper=1> y; // CORRIGIDO: nova sintaxe de array
    }
    parameters {
      vector[K] beta;           
      real beta0;               
    }
    model {
      // Priors
      beta0 ~ normal(-1, 1.5); 
      beta ~ normal(0.5, 1.0);
      
      // Likelihood
      vector[N] Z = beta0 + X * beta;
      y ~ bernoulli_logit(Z);
    }
    generated quantities {
        // Para prever probabilidades no conjunto completo
        vector[N] p_privada = inv_logit(beta0 + X * beta);
    }
    