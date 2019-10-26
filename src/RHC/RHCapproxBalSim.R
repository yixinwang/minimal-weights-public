#####################
# preprocess
#####################


rm(list = ls())

# load optsub.RData Rosenbaum 2012 JCGS
load("optsub.RData")

# X is numbering labels of the observations (kept but unused here)

# treatment swang1
# outcome minimum survival datas 
# minlife = max(last contact - admission, 180*(survive in 180 days))

logminlife = log(pmax(d65$lstctdte - d65$sadmdte, 
    (2 - as.numeric(d65$death) ) * 180))

treat = as.numeric(d65$swang1) - 1

# leave out colnames with a lot of NAs, keep only one outcome minlife
d65dat = subset(d65, select = -c(cat2, sadmdte, dschdte, dthdte,
    lstctdte, swang1, adld3p, urin1, surv30day, z, p, m, me194,
    me394, m0, me394t)) 

d65dat = data.frame(cbind(d65dat, treat,
    logminlife))

# d65dat = data.frame(cbind(d65[,c("age", "sex", "race", "edu",
# "income",
# "ninsclas", "cat1")], 
# treat))
    
cov = subset(d65dat, select = c(
    age, sex, race, cat1, ca, 
    income, ninsclas,
    dnr1, aps1, surv2md1, 
    adld3padjusted, adld3miss, 
    das2d3pc, 
    temp1, hrt1, meanbp1, resp1, 
    wblc1, pafi1, ph1, crea1, alb1, scoma1)) 
# leave out cat1 and cat2adjusted


# convert categorical variables into binary dummy variables
# dim = 2998 * 83
# -1 leaves out intercept
dat = data.frame(cbind(model.matrix(treat~.,data = d65dat)[,-1], 
    treat))
cov = data.frame(cbind(model.matrix(treat~.,data = cov)[,-1]))

colnames(dat) = make.names(colnames(dat))

write.csv(dat, "RHC_approxBal.csv", row.names = F, quote = F)


#####################
# simulate
#####################


dat = read.csv("RHC_approxBal.csv")

# medicaidprivateprivate and medicaidprivatemedicard 
# turn out to be colinear
# with other covariates, delete
# dat = subset(dat, select = -c(medicaidprivatemedicaid, medicaidprivateprivate))
# cov = subset(dat, select = -c(X))

# dat = dat[, 23:39]
# cov = dat[,-1]

# simulate outcome
# remember to leave out treat
droptreat = c("treat", "X")
ctrlmodel = lm(logminlife~., 
    data = dat[dat[,"treat"] == 0, !(names(dat) %in% droptreat)])
keepctrl = row.names(summary(ctrlmodel)$coefficients)[
    summary(ctrlmodel)$coefficients[, 4] < 0.05]
keepctrl = c(keepctrl, "logminlife")
keepctrl = keepctrl[keepctrl != "treat"]
ctrlmodel = lm(logminlife~., 
    data = dat[dat[,"treat"] == 0, (names(dat) %in% keepctrl)])
# ctrlCoeff = summary(ctrlmodel)$coefficients[,1]
ctrlPredAll = predict(ctrlmodel, newdata = dat[, (names(dat) %in% keepctrl)], se.fit=TRUE)

trtmodel = lm(logminlife~., 
    data = dat[dat[,"treat"] == 1, !(names(dat) %in% droptreat)])
keeptrt = row.names(summary(trtmodel)$coefficients)[
    summary(trtmodel)$coefficients[, 4] < 0.05]
keeptrt = c(keeptrt, "logminlife")
keeptrt = keeptrt[keeptrt != "treat"]
trtmodel = lm(logminlife~., 
    data = dat[dat[,"treat"] == 0, (names(dat) %in% keeptrt)])
# trtCoeff = summary(trtmodel)$coefficients[,1]
trtPredAll = predict(trtmodel, newdata = dat[, (names(dat) %in% keeptrt)], se.fit=TRUE)

# simulate treatment
# dat leave out X and minlife
dropoutcome = c("logminlife", "X")
propmodel = glm(treat~., 
    data = dat[, !(names(dat) %in% dropoutcome)], 
    # data = dat,
    family = "binomial")
# keep significant variables
keepprop = row.names(summary(propmodel)$coefficients)[
    summary(propmodel)$coefficients[, 4] < 0.05]
keepprop = c(keepprop, "treat")
propmodel = glm(treat~., 
    data = dat[, (names(dat) %in% keepprop)], 
    # data = dat,
    family = "binomial")

# covlist = union(union(keepctrl, keeptrt), keepprop)
# cov = dat[, (names(dat) %in% covlist)]

# c controls the overlap
# c small -> bad overlap
# c large -> good overlap
simTY <- function(c){
    simTmean = model.matrix(propmodel) %*% 
        summary(propmodel)$coefficients[,1]
    simT = as.numeric((simTmean / c + runif(dim(dat)[1]) - 0.5) > 0)
    ctrlPred = ctrlPredAll$fit + rnorm(dim(dat)[1], sd = ctrlPredAll$residual.scale)
    trtPred = trtPredAll$fit + rnorm(dim(dat)[1], sd = trtPredAll$residual.scale)
    simY = simT * trtPred + (1 - simT) * ctrlPred
    simDat = data.frame(cbind(cov, simT, simY, trtPred, ctrlPred))
    # simDat = simDat[order(-simDat[, "simT"]),]
    return(simDat)
}

# check covariate balance of the simulated dataset
# consider only covariate columns, no treatment or outcome
checkcovbal <- function(dat){
    dropbal = c("simT", "simY", "trtPred", "ctrlPred")
    covbal = (colMeans(dat[dat$simT==0, !(names(dat) %in% dropbal)]) - 
    colMeans(dat[dat$simT==1, !(names(dat) %in% dropbal)])) / 
    apply(dat[, !(names(dat) %in% dropbal)], 2, sd)
    return(covbal)
}


setwd("sim_datasets")
# generate D datasets
D = 1000

dat = simTY(3)
covbal = checkcovbal(dat)
mean(covbal^2)
mean(abs(covbal))

covbal_bad = data.frame(matrix(ncol = length(covbal), nrow = D))
covbal_good = data.frame(matrix(ncol = length(covbal), nrow = D))
colnames(covbal_bad) = names(covbal)
colnames(covbal_good) = names(covbal)



for (i in 1:D){
    dat_bad = simTY(0.1)
    write.csv(dat_bad, paste(paste("RHC_approxBal_bad", i-1, sep="_"), ".csv", sep=""), row.names = F, quote = F)
    covbal_bad[i,] = checkcovbal(dat_bad)
    print(rowMeans(covbal_bad[i,]^2))
    print(rowMeans(abs(covbal_bad[i,])))
    
    dat_good = simTY(3)
    write.csv(dat_good, paste(paste("RHC_approxBal_good", i-1, sep="_"), ".csv", sep=""), row.names = F, quote = F)
    covbal_good[i,] = checkcovbal(dat_good)
    print(rowMeans(covbal_good[i,]^2))
    print(rowMeans(abs(covbal_good[i,])))
}

write.csv(covbal_bad, "RHC_approxBal_bad_covbal.csv", row.names = F, quote = F)
write.csv(covbal_good, "RHC_approxBal_good_covbal.csv", row.names = F, quote = F)




