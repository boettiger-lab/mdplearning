

## Given state and prior, recommend next action

#' mdp online learning
#' 
#' Given previous state, previous action, and current state, update the model prior and propose best action
#' @inheritParams mdp_compute_policy
#' @inheritParams mdp_planning
#' @param prev_state the previous state of the system
#' @param prev_action the action taken after observing the previous state
#' @param state the most recent state observed
#' @return a list, with component 'action' giving the action recommended, and posterior, a 
#' vector of length(transition) giving the updated probability over models
#' @export
#' @details mdp_online provides a real-time updating mechanism given the latest observations.
#'  To learn the best model and compare proposed actions across historical data, use mdp_historical,
#'  which loops over mdp_online.  
#'  @examples 
#' source(system.file("examples/K_models.R", package="mdplearning"))
#' transition <- lapply(models, `[[`, "transition")
#' reward <- models[[1]]$reward
#' mdp_online(transition, reward, discount, c(0.5, 0.5), 10, 1, 12)
mdp_online <- function(transition, reward, discount, 
                       model_prior,
                       prev_state,
                       prev_action,
                       state,
                       ...){
  
  out <- mdp_compute_policy(transition, reward, discount, model_prior, ...)
  action <- out$policy[state]
  posterior <- bayes_update_model_belief(model_prior, prev_state, state, 
                                         prev_action, transition)
  
  list(action = action, posterior = posterior)
}
