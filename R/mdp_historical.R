#' mdp_historical
#' 
#' @inheritParams mdp_compute_policy
#' @param state sequence of states observed historically
#' @param action sequence of historical actions taken at time of observing that state
#' @return a list with component "df", a data.frame showing the historical state, 
#' historical action, and what action would have been recommended by MDP; and an
#' array showing the evolution of the belief over models during each subsequent observation
#' @export
mdp_historical <- function(transition, reward, discount, model_prior = NULL,
                            state, action, ...){
  
  Tmax = length(state)
  n_models <- length(transition)
  recommended <- numeric(Tmax)
  if(is.null(model_prior)) 
    model_prior <- rep(1, n_models) / n_models
  belief <- array(NA, dim = c(Tmax, n_models))
  belief[1,] <- model_prior
  
  for(t in 1:(Tmax-1)){
    out <- mdp_online(transition, reward, discount, model_prior, state[t], action[t], state[t+1], ...)
    recommended[t] <- out$action
    belief[t+1,] <- out$posterior
  }
  
  list(df = data.frame(time = 1:Tmax, state, action, recommended), posterior = belief)
}






