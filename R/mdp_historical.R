#' mdp_historical
#'
#' @inheritParams mdp_compute_policy
#' @inheritParams mdp_planning
#' @param state sequence of states observed historically
#' @param action sequence of historical actions taken at time of observing that state
#' @param model_names optional vector of names for columns in model posterior distribution. 
#' Will be taken from names of transition list if none are provided here. 
#' @return a list with component "df", a data.frame showing the historical state,
#' historical action, and what action would have been optimal by MDP; and a
#' data.frame showing the evolution of the belief over models during each subsequent observation
#' @export
mdp_historical <- function(transition, reward, discount, model_prior = NULL,
                            state, action, model_names, ...){

  
  if(any(is.na(model_names)))
    model_names <- names(transition)
  
  Tmax <- length(state)
  n_models <- length(transition)
  optimal <- numeric(Tmax)
  optimal[1] <- NA # initial recommendation is meaningless
  if(is.null(model_prior))
    model_prior <- rep(1, n_models) / n_models
  belief <- array(NA, dim = c(Tmax, n_models))
  belief[1,] <- model_prior

  for(t in 2:Tmax){
    out <- mdp_online(transition, reward, discount, belief[t-1,],
                      state[t-1], action[t-1], state[t], ...)
    optimal[t] <- out$action
    belief[t,] <- out$posterior
  }

  posterior <- as.data.frame(belief)
  if(!any(is.na(model_names))) names(posterior) <- model_names
  
  list(df = data.frame(time = 1:Tmax, state, action, optimal),
       posterior = posterior)
}
