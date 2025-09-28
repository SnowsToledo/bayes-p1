
// logit_rede_ensino.stan (Sintaxe Corrigida)

data {
  int<lower=0> N;          // Número total de observações
  int<lower=0> K;          // Número de preditores (notas)
  matrix[N, K] X;         // Matriz de preditores (notas padronizadas)

  // **CORREÇÃO AQUI:** array[N] int<lower=0, upper=1> y;
  array[N] int<lower=0, upper=1> y; // Variável resposta (0=Pública, 1=Privada)
}

parameters {
  vector[K] beta;           // Coeficientes das notas
  real beta0;               // Intercepto
}

model {
  // 1. Priors
  beta0 ~ normal(-1, 1.5); 
  beta ~ normal(0.5, 1.0);

  // 2. Likelihood
  vector[N] Z = beta0 + X * beta;
  y ~ bernoulli_logit(Z);
}

generated quantities {
    vector[N] p_privada_media = inv_logit(beta0 + X * beta);
}
