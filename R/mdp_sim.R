#' mdp sim
#'
#'
#' Simulate learning under the mdp policy
#' @details If no observation matrix is given, simulation is based on perfect observation.  Otherwise, simulation includes imperfect measurement
#' If a policy is given, this policy is used to determine the actions throughout the simulation and no learning takes place.
#' Otherwise, the simulation will compute the optimal (fully observed) MDP solution over the given transition models and model prior,
#' and continue to learn by updating the belief over models and the resulting optimal MDP policy.
#' @inheritParams compute_mdp_policy
#' @param x0 initial state
#' @param Tmax length of time to simulate
#' @param true_transition actual transition used to drive simulation. If a fixed policy is given, then true_transition = transition
#'  if it is not specified (i.e. avoids having to declare transition separately).
#' @param observation NULL by default, simulate perfect observations
#' @param a0 previous action before starting, irrelivant unless actions influence observations and true_observation is not null
#' @param policy a vector of length n_obs, whose i'th entry is the index of the optimal action given the system is in (observed) state i.
#' @return a list, containing: data frame "df" with the state, action and a value at each time step in the simulation,
#' and an array "posterior", in which the t'th row shows the belief state at time t.
#' @export
#'
#' @examples
#' source(system.file("examples/K_models.R", package="mdplearning"))
#' transition <- lapply(models, `[[`, "transition")
#' reward <- models[[1]][["reward"]]
#'
#' ## example where true model is model 1
#' out <- mdp_sim(transition, reward, discount, x0 = 10,
#'                     Tmax = 20, true_transition = transition[[1]])
#' ## Did we learn which one was the true model?
#' out$posterior[20,]
#'
#' ## simulate a fixed policy
#' df <- compute_mdp_policy(transition, reward, discount, model_prior = c(0.5, 0.5))
#' out <- mdp_sim(transition[[1]], reward, discount, x0 = 10,
#'                Tmax = 20,
#'                policy = df$policy)
#'
#' ## Simulate MDP strategy under observation uncertainty
#'
#' out <- mdp_sim(transition = transition, reward, discount, x0 = 10,
#'                true_transition = transition[[1]],
#'                Tmax = 20, observation = models[[1]][["observation"]])
mdp_sim <- function(transition, reward, discount,
                    model_prior = NULL,
                    x0, 
                    Tmax = 20, 
                    true_transition = transition,
                    observation = NULL,
                    a0 = 1,
                    policy = NULL,
                    max_iter = 500, 
                    epsilon = 1e-5, 
                    type = c("policy iteration", 
                             "value iteration", 
                             "finite time")){
  type <- match.arg(type)

  n_states <- dim(true_transition)[1]
  state <- obs <- action <- value <- numeric(Tmax+1)

  if(is.null(policy)){
    if(is.null(model_prior))
      model_prior<- rep(1, length(transition)) / length(transition)
    q <- array(NA, dim = c(Tmax+2, length(transition)))
    q[2,] <- model_prior
  }

  state[2] <- x0
  action[1] <- a0

  time <- 2:(Tmax+1)
  for(t in time){

    ## if no policy is given, then we learn over the models and update
    if(is.null(policy)){
      if(type == "finite time"){ 
        max_iter <- Tmax + 3 - t
      }
      out <- compute_mdp_policy(transition, reward, discount, 
                                q[t,], max_iter = max_iter, 
                                epsilon = epsilon, type = type)
    } else {
      out <- list(policy = policy)
    }

    ## Use perfect observations unless observation matrix is given
    if(!is.null(observation)){
      obs[t] <- sample(1:dim(observation)[2], 1, 
                       prob = observation[state[t], , action[t-1]])
    } else{
      obs[t] <- state[t]
    }

    ## Select action, determine value, transition to next state
    action[t] <- out$policy[obs[t]]
    value[t] <- reward[state[t], action[t]] * discount^(t-1)
    prob <- true_transition[state[t], , action[t]]
    state[t+1] <- sample(1:n_states, 1, prob = prob)

    ## Update belief
    if(is.null(policy))
      q[t+1,] <- bayes_update_model_belief(q[t,], state[t], 
                                           state[t+1], action[t], 
                                           transition)
  }

  df <- data.frame(time = 1:Tmax, state = state[time], 
                   obs = obs[time], action = action[time], 
                   value = value[time])
  if(is.null(policy))
    list(df = df, posterior = q[time,])
  else
    df
}

bayes_update_model_belief <- function(model_prior, x_t, x_t1, a_t, transition){
  n_models <- length(transition)
  P <- vapply(1:n_models, function(m) transition[[m]][x_t,x_t1,a_t], numeric(1))
  model_prior * P / sum(model_prior * P)
}

