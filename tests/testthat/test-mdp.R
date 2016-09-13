source(system.file("examples/K_models.R", package="mdplearning"))
transition <- lapply(models, `[[`, "transition")
reward <- models[[1]][["reward"]]
observation <- models[[1]]$observation


############################
testthat::context("Compute mdp policy")

df <- mdp_compute_policy(transition, reward, discount)
df <- mdp_compute_policy(transition, reward, discount, 
                         type = "value iteration")
df <- mdp_compute_policy(transition, reward, discount, 
                         type = "finite time")



#################
testthat::context("Learning")

out <- mdp_learning(transition, reward, discount, x0 = 10, 
               Tmax = 20, true_transition = transition[[1]])

testthat::test_that("Test that we learned the correct model", {
  belief <- out$posterior[20,]
  testthat::expect_gt(belief[1], belief[2])
})


testthat::test_that("Simulation with learning, observation error present", {
  out <- mdp_learning(transition, reward, discount, x0 = 10, Tmax = 20, 
                      true_transition = transition[[1]], 
                      observation = observation)
  testthat::expect_is(out, "list")
})


#################
testthat::context("Simulation with fixed policy")

out <- mdp_planning(transition[[1]], reward, discount, x0 = 10, 
               Tmax = 20, policy = df$policy)


testthat::test_that("Simulation with fixed policy, finite time", {
  out <- mdp_planning(transition[[1]], reward, discount, x0 = 10, 
               Tmax = 20, policy = df$policy, observation = observation, 
               type = "finite time")
  testthat::expect_is(out, "data.frame")
               })

testthat::test_that("Simulation with fixed policy, 
                    observation error present", {
  out <- mdp_planning(transition[[1]], reward, discount, x0 = 10, 
               Tmax = 20, policy = df$policy, observation = observation)
  testthat::expect_is(out, "data.frame")
               })




##################
testthat::context("mdp_value_of_policy")


testthat::test_that("value of policy works as expected", {
  v <- mdp_value_of_policy(df$policy, transition, reward, discount)
               })


############ historical

## setup
df <- mdp_learning(transition, reward, discount, x0 = 10, Tmax = 20, 
              true_transition = transition[[1]])$df

mdp_historical(transition, reward, discount, 
               action = df$action, 
               state = df$state)


