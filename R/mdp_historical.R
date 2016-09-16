#' mdp_historical
#'
#' @inheritParams mdp_compute_policy
#' @inheritParams mdp_planning
#' @param state sequence of states observed historically
#' @param action sequence of historical actions taken at time of observing that state
#' @return a list with component "df", a data.frame showing the historical state,
#' historical action, and what action would have been optimal by MDP; and a
#' data.frame showing the evolution of the belief over models during each subsequent observation
#' @export
mdp_historical <- function(transition, reward, discount, model_prior = NULL,
                            state, action, ...){

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

  list(df = data.frame(time = 1:Tmax, state, action, optimal),
       posterior = as.data.frame(belief))
}
