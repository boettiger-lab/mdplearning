states <- 0:20
actions <- states
obs <- states
sigma_g <- 0.1
sigma_m <- sigma_g
reward_fn <- function(x,h) pmin(x,h)
discount <- 0.95
K1 <- function(x, h, r = 1, K = 15){
  s <- pmax(x - h, 0)
  s * exp(r * (1 - s / K) )
}
K2 <- function(x, h, r = 1, K = 10){
  s <- pmax(x - h, 0)
  s * exp(r * (1 - s / K) )
}

fisheries <- function(states = 0:23,
         actions = states,
         observed_states = states,
         reward_fn = function(x,a) pmin(x,a),
         f = ricker,
         sigma_g = 0.1,
         sigma_m = sigma_g){

  ## Transition and Reward Matrices
  n_s <- length(states)
  n_a <- length(actions)
  n_z <- length(observed_states)

  transition <- array(0, dim = c(n_s, n_s, n_a))
  reward <- array(0, dim = c(n_s, n_a))
  observation <- array(0, dim = c(n_s, n_z, n_a))

  for (k in 1:n_s) {
    for (i in 1:n_a) {
      nextpop <- f(states[k], actions[i])
      if(nextpop <= 0){
        transition[k, , i] <- c(1, rep(0, n_s - 1))
      } else if(sigma_g > 0){
        x <- dlnorm(states, log(nextpop), sdlog = sigma_g)
        if(sum(x) == 0){ ## nextpop is computationally zero
          transition[k, , i] <- c(1, rep(0, n_s - 1))
        } else {
          N <- plnorm(states[n_s], log(nextpop), sigma_g)
          x <- x * N / sum(x)
          x[n_s] <- 1 - N + x[n_s]
          transition[k, , i] <- x
        }
      } else {
        stop("sigma_g not > 0")
      }
      reward[k, i] <- reward_fn(states[k], actions[i])
    }
  }

  for (k in 1:n_a) {
    if(sigma_m <= 0){
      observation[, , k] <- diag(n_s)
    } else {
      for (i in 1:n_s) {
        if(states[i] <= 0){
        observation[i, , k] <- c(1, rep(0, n_z - 1))
        } else {
          x <- dlnorm(observed_states, log(states[i]), sdlog = sigma_m)
          ## Normalize using CDF
          N <- plnorm(observed_states[n_s], log(states[i]), sigma_m)
          x <- x * N / sum(x)
          x[n_s] <- 1 - N + x[n_s]
          observation[i, , k] <- x
        }
      }
    }
  }
  list(transition = transition, 
       observation = observation, 
       reward = reward)
}


models <- lapply(list(K1, K2), function(f) 
                 fisheries(states, actions, obs, 
                           reward_fn, f, sigma_g, sigma_m))

rm("fisheries")
