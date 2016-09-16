
#' mdp learning
#'
#'
#' Simulate learning under the mdp policy
#' @inheritParams mdp_compute_policy
#' @inheritParams mdp_planning
#' @param true_transition actual transition used to drive simulation.
#' @return a list, containing: data frame "df" with the state, action and a value at each time step in the simulation,
#' and a data.frame "posterior", in which the t'th row shows the belief state at time t.
#' @export
#'
#' @examples
#' source(system.file("examples/K_models.R", package="mdplearning"))
#' transition <- lapply(models, `[[`, "transition")
#' reward <- models[[1]]$reward
#'
#' ## example where true model is model 1
#' out <- mdp_learning(transition, reward, discount, x0 = 10,
#'                     Tmax = 20, true_transition = transition[[1]])
#' ## Did we learn which one was the true model?
#' out$posterior[20,]
#'
#' ## Simulate MDP strategy under observation uncertainty
#' out <- mdp_learning(transition = transition, reward, discount, x0 = 10,
#'                true_transition = transition[[1]],
#'                Tmax = 20, observation = models[[1]]$observation)
mdp_learning <- function(transition, reward, discount, model_prior = NULL,
                    x0,
                    Tmax = 20,
                    true_transition,
                    observation = NULL,
                    a0 = 1,
                    ...){

  n_states <- dim(true_transition)[1]
  n_models <- length(transition)
  state <- obs <- action <- value <- numeric(Tmax+1)
  state[2] <- x0
  action[1] <- a0
  time <- 2:(Tmax+1)

  if(is.null(model_prior))
    model_prior<- rep(1, n_models) / n_models
  belief <- array(NA, dim = c(Tmax+2, length(transition)))
  belief[2,] <- model_prior

  for(t in time){
    out <- mdp_compute_policy(transition, reward, discount, belief[t,],
                                Tmax = Tmax+3 - t, ...)
    ## Use imperfect observations if requested
    obs[t] <- state[t]
    if(!is.null(observation))
      obs[t] <- sample(1:dim(observation)[2], 1, prob =
                         observation[state[t], , action[t-1]])
    ## Select action, determine value, transition to next state
    action[t] <- out$policy[obs[t]]
    value[t] <- reward[state[t], action[t]] * discount^(t-1)
    state[t+1] <- sample(1:n_states, 1, prob =
                           true_transition[state[t], , action[t]])
    belief[t+1,] <- bayes_update_model_belief(belief[t,], state[t],
                                              state[t+1], action[t],
                                              transition)

  }

  list(df = data.frame(time = 1:Tmax, state = state[time],
                   obs = obs[time], action = action[time],
                   value = value[time]),
       posterior = as.data.frame(belief[time,]))
}
