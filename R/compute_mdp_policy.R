#' compute mdp policy
#'
#' @param transition list of transition matrices, one per model
#' @param reward the utility matrix U(x,a) of being at state x and taking action a
#' @param discount the discount factor (1 is no discounting)
#' @param model_prior the prior belief over models, a numeric of length(transitions). Uniform by default
#' @param max_iter maximum number of iterations to perform
#' @param epsilon convergence tolerance
#' @param type consider converged when policy converges or when value converges?
#'
#' @return a data.frame with the optimal policy and (discounted) value associated with each state
#' @export
#'
#' @examples
#' source(system.file("examples/K_models.R", package="mdplearning"))
#' transition <- lapply(models, `[[`, "transition")
#' reward <- models[[1]][["reward"]]
#' df <- compute_mdp_policy(transition, reward, discount)
#' plot(df$state, df$state - df$policy, xlab = "stock", ylab="escapement")
compute_mdp_policy <- function(transition, reward, discount,
                               model_prior = rep(1, length(transition))/length(transition),
                               max_iter = 500, epsilon = 1e-5, type = c("policy iteration", "value iteration", "finite time")){

  type <- match.arg(type)

  n_models <- length(transition)
  n_states <- dim(transition[[1]])[1]
  n_actions <- dim(transition[[1]])[3]
  next_value <- numeric(n_states)
  next_policy <- numeric(n_states)
  V_model <- array(dim=c(n_states, n_models))
  converged <- FALSE
  t <- 1

  while(t < max_iter && converged == FALSE){
    Q <- array(0, dim = c(n_states, n_actions))
    for (i in 1:n_actions) {
      for(j in 1:n_models){
        V_model[,j] <- transition[[j]][,,i] %*% next_value
      }
      Q[,i] <- reward[, i] + discount * V_model %*% model_prior
    }
    value <- apply(Q, 1, max)
    policy <- apply(Q, 1, which.max)


    if(type == "value iteration"){
      if( sum( abs(value - next_value) ) < epsilon ){
        converged <- TRUE
      }
    } else if(type == "policy iteration"){
      if( sum( abs(policy - next_policy) ) < epsilon ){
        converged <- TRUE
      }
    }

    next_value <- value
    next_policy <- policy
    t <- t+1
    if(t == max_iter)
      message("Note: max number of iterations reached")
  }
  data.frame(state = 1:n_states, policy, value)
}



