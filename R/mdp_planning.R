#' mdp planning
#'
#'
#' Simulate MDP under a given policy
#' @inheritParams mdp_compute_policy
#' @param x0 initial state
#' @param Tmax length of time to simulate
#' @param observation NULL by default, simulate perfect observations
#' @param a0 previous action before starting, irrelivant unless actions influence observations and true_observation is not null
#' @param policy a vector of length n_obs, whose i'th entry is the index of the optimal action given the system is in (observed) state i.
#' @param ... additional arguments to \code{\link{mdp_compute_policy}}
#' @return a data frame "df" with the state, action and a value at each time step in the simulation
#' @export
#'
#' @examples
#' source(system.file("examples/K_models.R", package="mdplearning"))
#' transition <- lapply(models, `[[`, "transition")
#' reward <- models[[1]]$reward
#'
#' df <- mdp_compute_policy(transition, reward, discount, model_prior = c(0.5, 0.5))
#' out <- mdp_planning(transition[[1]], reward, discount, x0 = 10,
#'                Tmax = 20, policy = df$policy)
#'
#' ## Simulate MDP strategy under observation uncertainty
#' out <- mdp_planning(transition[[1]], reward, discount, x0 = 10,
#'                Tmax = 20, policy = df$policy, 
#'                observation = models[[1]]$observation)

#'
mdp_planning <- function(transition, reward, discount, model_prior = NULL,
                    x0, Tmax = 20, 
                    observation = NULL, a0 = 1,
                    policy,
                    ...){
  
 
  n_states <- dim(transition)[1]
  state <- obs <- action <- value <- numeric(Tmax+1)
  state[2] <- x0
  action[1] <- a0
  time <- 2:(Tmax+1)
  
  for(t in time){
    obs[t] <- state[t]  
    if(!is.null(observation))
      obs[t] <- sample(1:dim(observation)[2], 1, prob = 
                         observation[state[t], , action[t-1]])
      ## Select action, determine value, transition to next state
      action[t] <- policy[obs[t]]
      value[t] <- reward[state[t], action[t]] * discount^(t-1)
      state[t+1] <- sample(1:n_states, 1, prob = 
                             transition[state[t], , action[t]])
  }
  
  data.frame(time = 1:Tmax, state = state[time], 
             obs = obs[time], action = action[time], 
             value = value[time])
}
