source(system.file("examples/K_models.R", package="mdplearning"))
transition <- lapply(models, `[[`, "transition")
reward <- models[[1]][["reward"]]

df <- compute_mdp_policy(transition, reward, discount)

out <- mdp_sim(transition, reward, discount, x0 = 10, Tmax = 20, true_transition = transition[[1]])
final_belief <- out$posterior[20,]

v <- mdp_value_of_policy(df$policy, transition, reward, discount)

out <- mdp_sim(transition[[1]], reward, discount, x0 = 10, Tmax = 20, policy = df$policy)
