#' mdp_value_of_policy
#'
#' Compute the expected net present (e.g. discounted) value of a (not-necessarily optimal) policy in a perfectly observed (MDP) system
#' @inheritParams compute_mdp_policy
#' @param policy the policy for which we want to determine the expected value
#' @return the expected net present value of the given policy, for each state
#' @details transition can be a single transition matrix or a list of transition matrices
#' @examples
#' source(system.file("examples/K_models.R", package="mdplearning"))
#' transition <- lapply(models, `[[`, "transition")
#' reward <- models[[1]][["reward"]]
#' df <- compute_mdp_policy(transition, reward, discount)
#' v <- mdp_value_of_policy(df$policy, transition, reward, discount)
#' @export
mdp_value_of_policy <- function(policy, transition, reward, discount, 
                                model_prior = NULL, 
                                max_iter = 500, epsilon = 1e-5){


  if(is.null(model_prior)){
    model_prior<- rep(1, length(transition)) / length(transition)
  }

  if(is.array(transition)){
    transition <- list(transition)
  }
  n_models <- length(transition)
  n_states <- dim(transition[[1]])[1]
  n_actions <- dim(transition[[1]])[3]
  Vt <- numeric(n_states)
  next_value <- Vt
  V_model <- array(dim=c(n_states, n_models))
  converged <- FALSE
  t <- 1
  while(t < max_iter && converged == FALSE){
    Q <- array(0, dim = c(n_states, n_actions))
    for (i in 1:n_actions) {
      for(j in 1:n_models){
        V_model[,j] <- transition[[j]][,,i] %*% Vt
      }
      Q[,i] <- reward[, i] + discount * V_model %*% model_prior
    }

    for(i in 1:n_states)
      Vt[i] <- Q[i,policy[i]]

    if( sum( abs(Vt - next_value) ) < epsilon ){
      converged <- TRUE
    }
    next_value <- Vt
    t <- t + 1
    if(t == max_iter)
      message("Note: max number of iterations reached")
  }
  Vt
}

