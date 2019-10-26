library(mvtnorm)
library(MASS)

# Simulate data
kangschafer = function(n_obs) {
# Z are the true covariates
# t is the indicator for the respondents (treated)
# y is the outcome
# X are the observed covariates
# Returns Z, t y and X sorted in decreasing order by t
    Z = mvrnorm(n_obs, mu = rep(0, 4), Sigma = diag(4))
    p = 1 / (1 + exp(Z[, 1] + .5 * Z[, 2] + .25 * Z[, 3] + .1 * Z[, 4]))
    # p = 1 / (1 + exp(Z[, 1] + 2 * Z[, 2] + .25 * Z[, 3] + .1 * Z[, 4]))
    t = rbinom(n_obs, 1, p)
    Zt = cbind(Z, p, t)
    Zt = Zt[order(t), ]
    Z = Zt[, 1:4]
    p = Zt[, 5]
    t = Zt[, 6]
    y = 210 + 27.4 * Z[, 1] + 13.7 * Z[, 2] + 13.7 * Z[, 3] + 
        13.7 * Z[, 4] + rnorm(n_obs)
    X = cbind(exp(Z[, 1] / 2), (Z[, 2] / (1 + exp(Z[, 1]))) + 10, 
        (Z[, 1] * Z[, 3] / 25 + .6)^3, (Z[, 2] + Z[, 4] + 20)^2)    
    return(list(Z = Z, p = p, t = t, y = y, X = X)) 
}   
# set.seed(1)

for (i in 1:200){
    n_obs = 5000
    aux = kangschafer(n_obs)
    Z = aux$Z
    p = aux$p
    t = aux$t
    y = aux$y
    X = aux$X
    X2 = X^2
    X3 = X^3
    X4 = X^4
    X5 = X^5
    # Data frame    
    t_ind = t
    bal_covs = cbind(X, X2, X3, X4, X5)
    data_frame = as.data.frame(cbind(bal_covs, t_ind))
    names(data_frame) = c( 
        "X1",   "X2",   "X3",   "X4", 
        "X1^2", "X2^2", "X3^2", "X4^2", 
        "X1^3", "X2^3", "X3^3", "X4^3",
        "X1^4", "X2^4", "X3^4", "X4^4",
        "X1^5", "X2^5", "X3^5", "X4^5",
        "t_ind"
        )  
    data = cbind(data_frame, p, y)
    write.csv(data, paste(i,"kangschafer_approxbal_good.csv", sep=""), row.names = F, quote = F)
    # Treatment indicator 
    t_ind = "t_ind"

}

    




# mean y on all
mean(y)

# mean y on treated
sum(y  *  t) / sum(t)

# mean y on control
sum(y * (1 - t)) / sum((1 - t))

# weighted y
sum(data_frame_weights$weights  *  y)






