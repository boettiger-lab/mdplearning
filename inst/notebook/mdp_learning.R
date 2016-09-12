
value_iteration <- function(transition, utility, qt = rep(1, length(transition))/length(transition), Tmax){

  n_models <- length(transition)
  n_states <- dim(transition[[1]])[1]
  n_actions <- dim(transition[[1]])[3]

  ## Initialize value-to-go (integrated over models)
  Vtplus <- numeric(n_states)
  ## Initialize policy & final value
  policy <- numeric(n_states)
  ## Intialize value-to-go over states i for model j
  V_model <- array(dim=c(n_states, n_models))

  for (t in (Tmax - 1):1) {
    # We define a matrix Q that stores the updated action values for
    # all states (rows), actions (columns)
    Q <- array(0, dim = c(n_states, n_actions))
    for (i in 1:n_actions) {
      # For each action we fill for all states values (rows)
      # the value associated with that action (ith column) into matrix Q
      # The utility of the ith action recorded for each state is
      # added to the discounted average (product) of the transition matrix of the ith
      # action by the running action value Vt

      ## In the case of parametric/model uncertainty, we compute this value for each model..
      for(j in 1:n_models){
        V_model[,j] <- transition[[j]][,,i] %*% Vtplus
      }

      ## and then average over models weighted by qt
      Q[,i] <- utility[, i] + discount * V_model %*% qt

    } # end of the actions loop

    # Find the optimal action value at time t is the maximum of Q
    Vt <- apply(Q, 1, max)

    # After filling vector Vt of the action values at all states, we
    # update the vector Vt+1 to Vt and we go to the next step standing
    # for previous time t-1, since we iterate backward
    Vtplus <- Vt

  } # end of the time loop

  # Find optimal action for each state
  for (k in 1:n_states) {
    # We look for each state which column of Q corresponds to the
    # maximum of the most recent value, Vt (at time t + 1).
    # If the index vector is longer than 1, (i.e. if there is more
    # than one equally optimal value) we chose the smaller action (min harvest)
    policy[k] <- min(which(Q[k, ] == Vt[k]))
  }

  list(policy = policy, value = Vt)
}




value_of_policy <- function(policy, transition, utility, qt = rep(1, length(transition))/length(transition), Tmax){

  if(is.array(transition)){
    transition <- list(transition)
  }

  n_models <- length(transition)
  n_states <- dim(transition[[1]])[1]
  n_actions <- dim(transition[[1]])[3]

  Vt <- numeric(n_states)
  V_model <- array(dim=c(n_states, n_models))

  for (t in (Tmax - 1):1) {
    Q <- array(0, dim = c(n_states, n_actions))
    for (i in 1:n_actions) {
      for(j in 1:n_models){
        V_model[,j] <- transition[[j]][,,i] %*% Vt
      }
      Q[,i] <- utility[, i] + discount * V_model %*% qt
    }

    for(i in 1:n_states)
      Vt[i] <- Q[i,policy[i]]
    #Vt <- apply(Q, 1, max)
  }
  Vt
}


sim_policy <- function(transition, utility, policy, x0, Tmax){
  state <- action <- value <- numeric(Tmax)
  time <- 1:(Tmax-1)
  state[1] <- x0
  for(t in time){

    action[t] <- policy[state[t]]
    value[t] <- utility[state[t], action[t]] * discount^(t-1)
    prob <- transition[state[t], , action[t]]
    state[t+1] <- sample(1:n_states, 1, prob = prob)


  }

  data.frame(time, state, action, value)
}



bayes_update_qt <- function(qt, x_t, x_t1, a_t, transition){

  n_models <- length(transition)
  P <- vapply(1:n_models, function(m) transition[[m]][x_t,x_t1,a_t], numeric(1))
  qt * P / sum(qt * P)

}


mdp_learning <- function(initial_state_index, transition, utility,
                         qt = rep(1, length(transition)) / length(transition),
                         Tmax = 20, true_transition){

  n_states <- dim(transition[[1]])[1]
  state <- numeric(Tmax)
  action <- numeric(Tmax)
  value <- numeric(Tmax)
  q <- array(NA, dim = c(Tmax, length(transition)))
  q[1,] <- qt
  state[1] <- initial_state_index


  for(t in 1:(Tmax-1)){

    ## Determine optimal action
    out <- value_iteration(transition, utility, q[t,], Tmax-t)
    ## Take proposed action, collect discounted reward
    action[t] <- out$policy[state[t]]
    value[t] <- utility[state[t], action[t]] * discount^(t-1)
    ## draw new state based on true model probability
    prob <- true_transition[state[t], , action[t]]
    state[t+1] <- sample(1:n_states, 1, prob = prob)

    ## Update belief
    q[t+1,] <- bayes_update_qt(q[t,], state[t], state[t+1], action[t], transition)


  }

  # Final action and associated value
  # action[t+1] <- out$policy[state[t+1]]
  # value[t+1] <- utility[state[t+1], action[t+1]]


  list(df = data.frame(time = 1:Tmax, state = state, action = action, value = value), posterior = q)
}
