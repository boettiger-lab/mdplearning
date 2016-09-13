
bayes_update_model_belief <- function(model_prior, x_t, x_t1, a_t, transition){
  n_models <- length(transition)
  P <- vapply(1:n_models, function(m) transition[[m]][x_t,x_t1,a_t], numeric(1))
  model_prior * P / sum(model_prior * P)
}


