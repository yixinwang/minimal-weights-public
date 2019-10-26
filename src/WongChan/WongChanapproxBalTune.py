import cvxpy as cp
import numpy as np
import numpy.random as npr
import pandas as pd
import math as ma
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker

from cvxpy.error import SolverError

def approxBalATT(X, T, Y, tol_vals, objective):
    # weight the control units so that they match the treated units

    n_ctrl = np.sum(1-T) # number of controls
    n_cov = X.shape[1] # number of covariates

    w = cp.Variable(n_ctrl)
    tol = cp.Parameter()
    if objective == "l1":
        obj = cp.Minimize(cp.norm((w-1/n_ctrl),1))
    elif objective == "l2":
        obj = cp.Minimize(cp.sum_squares(w-1/n_ctrl))
    elif objective == "entropy":
        obj = cp.Minimize(-cp.sum(cp.entr(w)))
    else:
        print("invalid objective")
        # return -1

    constraints = [cp.sum(w)==1]
    
    constraints += [0 <= w]
    for i in range(n_cov):
        constraints += [X[T==0][:,i]*w - \
                        np.mean(X[T==1][:,i]) <= \
                        tol * X[:,i].std()]
        constraints += [np.mean(X[T==1][:,i]) -\
                        X[T==0][:,i]*w <=\
                        tol * X[:,i].std()]
    prob = cp.Problem(obj, constraints)

    w_vals = []
    for tol_val in tol_vals:
        tol.value = tol_val
        try:
            result = prob.solve()
            if w.value is None:
                w.value = -1. * np.ones(n_ctrl)
            w_vals.append(w.value)
        except SolverError:
            w_vals.append(-1. * np.ones(n_ctrl))

    return w_vals


def approxBalATE(X, T, Y, tol_vals, objective):
    # weight the control and treated units so that they match the whole population

    n_ctrl = np.sum(1-T) # number of controls
    n_trt = np.sum(T)
    n_cov = X.shape[1] # number of covariates

    tol = cp.Parameter()

    # weight the control units to match the population
    w_c = cp.Variable(n_ctrl)
    if objective == "l1":
        obj_c = cp.Minimize(cp.norm((w_c - 1/n_ctrl),1))
    elif objective == "l2":
        obj_c = cp.Minimize(cp.sum_squares(w_c-1/n_ctrl))
    elif objective == "entropy":
        obj_c = cp.Minimize(-cp.sum(cp.entr(w_c)))
    else:
        print("invalid objective")
        # return -1

    constraints_c = [cp.sum(w_c)==1]
    
    constraints_c += [0 <= w_c]
    for i in range(n_cov):
        constraints_c += [X[T==0][:,i]*w_c - \
                        np.mean(X[:,i]) <= \
                        tol * X[:,i].std()]
        constraints_c += [np.mean(X[:,i]) -\
                        X[T==0][:,i]*w_c <=\
                        tol * X[:,i].std()]
    prob_c = cp.Problem(obj_c, constraints_c)

    w_c_vals = []
    for tol_val in tol_vals:
        tol.value = tol_val
        try:
            result_c = prob_c.solve()
            if w_c.value is None:
                w_c.value = -1. * np.ones(n_ctrl)
            w_c_vals.append(w_c.value)
        except SolverError:
            w_c_vals.append(-1. * np.ones(n_ctrl))
        

    # weight the treated units to match the population
    w_t = cp.Variable(n_trt)
    if objective == "l1":
        obj_t = cp.Minimize(cp.norm((w_t - 1/n_trt),1))
    elif objective == "l2":
        obj_t = cp.Minimize(cp.sum_squares(w_t-1/n_trt))
    elif objective == "entropy":
        obj_t = cp.Minimize(-cp.sum(cp.entr(w_t)))
    else:
        print("invalid objective")
        # return -1

    constraints_t = [cp.sum(w_t)==1]
    
    constraints_t += [0 <= w_t]
    for i in range(n_cov):
        constraints_t += [X[T==1][:,i]*w_t - \
                        np.mean(X[:,i]) <= \
                        tol * X[:,i].std()]
        constraints_t += [np.mean(X[:,i]) -\
                        X[T==1][:,i]*w_t <= \
                        tol * X[:,i].std()]
    prob_t = cp.Problem(obj_t, constraints_t)

    w_t_vals = []
    for tol_val in tol_vals:
        tol.value = tol_val
        try:
            result_t = prob_t.solve()
            if w_t.value is None:
                w_t.value = -1. * np.ones(n_trt)
            w_t_vals.append(w_t.value)
        except SolverError:
            w_t_vals.append(-1. * np.ones(n_trt))

    return w_c_vals, w_t_vals

def covBalATTBootstrapEval(w_c, X, T, metric, prop=0.5, smps=10):
    '''evaluate covariate balance with bootstrap samples t
    to choose tuning parameters'''

    if np.all(w_c > -1):
        n_ctrl = np.sum(1-T)
        n_subset = int(prop * n_ctrl)
        sd = np.std(X, axis=0)

        subsamples = [npr.choice(n_ctrl, n_subset, replace=True) for _ in range(smps)]

        cov_difs = [(w_c[subsample].dot(X[T==0][subsample]) / np.sum(w_c[subsample]) - \
            np.mean(X[T==1], axis = 0))/sd for subsample in subsamples]

        if metric == "l1":
            dif = [np.sum(np.abs(cov_dif)) for cov_dif in cov_difs]
        elif metric == "l2":
            dif = [np.sum(cov_dif**2) for cov_dif in cov_difs]
        elif metric == "linf":
            dif = [np.max(np.abs(cov_dif)) for cov_dif in cov_difs]
        else: 
            dif = -1
            print("invalid metric")
        return np.nanmean(dif)

    else:
        return 1e+16

def covBalATEBootstrapEval(w_c, w_t, X, T, metric, prop=0.5, smps=10):
    '''evaluate covariate balance with bootstrap samples t
    to choose tuning parameters'''

    if np.all(w_c > -1) and np.all(w_t > -1):

        n_all = len(T)
        w = np.zeros(n_all)
        w[T==0] = w_c
        w[T==1] = w_t
        n_subset = int(prop * n_all)
        subsamples = [npr.choice(n_all, n_subset, replace=True) for _ in range(smps)]
        sd = np.std(X, axis=0)

        cov_difs = []

        for subsample in subsamples:
            w_sub = w[subsample]
            T_sub = T[subsample]
            X_sub = X[subsample]
            cov_dif = np.array((w_sub[T_sub==0].dot(X_sub[T_sub==0])/np.sum(w_sub[T_sub==0]) - \
                            w_sub[T_sub==1].dot(X_sub[T_sub==1])/np.sum(w_sub[T_sub==1]))/sd)
            cov_difs.append(cov_dif)

        if metric == "l1":
            dif = [np.sum(np.abs(cov_dif)) for cov_dif in cov_difs]
        elif metric == "l2":
            dif = [np.sum(cov_dif**2) for cov_dif in cov_difs]
        elif metric == "linf":
            dif = [np.max(np.abs(cov_dif)) for cov_dif in cov_difs]
        else: 
            dif = -1
            print("invalid metric")
        return np.nanmean(dif)    

    else:
        return 1e+16

  

if __name__ == '__main__':
    datdir = "sim_datasets/"


    N = 100


    tol_vals = np.array([0, 1e-3, 2e-3, 5e-3, \
                1e-2, 2e-2, 5e-2, \
                1e-1, 2e-1, 5e-1, 1])


    for prop in [0.1]:
        for cov_qual in ["A", "B"]:
            ate_err_vs_covbal = pd.DataFrame({"tol":tol_vals})
            att_err_vs_covbal = pd.DataFrame({"tol":tol_vals})
            for objective in ["l1", "l2", "entropy"]:
                att_sqerrs = []
                ate_sqerrs = []
                att_covbal_l1s = []
                att_covbal_l2s = []
                att_covbal_linfs = []
                ate_covbal_l1s = []
                ate_covbal_l2s = []
                ate_covbal_linfs = []

                for i in range(N):
                    print(i)                   
                    filename = datdir + str(i) + 'WongChanSim' + cov_qual + '.csv'
                    data = pd.read_csv(filename, header=None)

                    cov, T, Y, Y1, Y0 = data.iloc[:,11:21].as_matrix().copy(), \
                        data.iloc[:,21].as_matrix().copy().astype(int), data.iloc[:,22].as_matrix().copy(), \
                        data.iloc[:,23].as_matrix().copy(), data.iloc[:,24].as_matrix().copy()

                    # only balance the mean of the covariates
                    X = np.column_stack([cov, cov**2])

                    true_att = np.mean(Y1[T==1]-Y0[T==1])
                    true_ate = np.mean(Y1-Y0)

                    att_w_c_vals = approxBalATT(X, T, Y, tol_vals, objective)

                    att_covbal_l1 = [covBalATTBootstrapEval(w_c, X, T, "l1", prop=prop) for w_c in att_w_c_vals]        
                    att_covbal_l2 = [covBalATTBootstrapEval(w_c, X, T, "l2", prop=prop) for w_c in att_w_c_vals]
                    att_covbal_linf = [covBalATTBootstrapEval(w_c, X, T, "linf", prop=prop) for w_c in att_w_c_vals]

                    att_covbal_l1s.append(att_covbal_l1)
                    att_covbal_l2s.append(att_covbal_l2)
                    att_covbal_linfs.append(att_covbal_linf)

                    att_sqerr = [((np.mean(Y[T==1]) - w_c.dot(Y[T==0])) - true_att)**2 \
                        for w_c in att_w_c_vals]
                    att_sqerrs.append(att_sqerr)

                    ate_w_c_vals, ate_w_t_vals = approxBalATE(X, T, Y, tol_vals, objective)

                    ate_covbal_l1 = [covBalATEBootstrapEval(w_c, w_t, X, T, "l1", prop=prop) \
                        for w_c, w_t in zip(ate_w_c_vals, ate_w_t_vals)]
                    ate_covbal_l2 = [covBalATEBootstrapEval(w_c, w_t, X, T, "l2", prop=prop) \
                        for w_c, w_t in zip(ate_w_c_vals, ate_w_t_vals)]
                    ate_covbal_linf = [covBalATEBootstrapEval(w_c, w_t, X, T, "linf", prop=prop) \
                        for w_c, w_t in zip(ate_w_c_vals, ate_w_t_vals)]

                    ate_covbal_l1s.append(ate_covbal_l1)
                    ate_covbal_l2s.append(ate_covbal_l2)
                    ate_covbal_linfs.append(ate_covbal_linf)

                    ate_sqerr = [((w_t.dot(Y[T==1]) - w_c.dot(Y[T==0])) - true_ate)**2 \
                        for w_t, w_c in zip(ate_w_t_vals, ate_w_c_vals)]
                    ate_sqerrs.append(ate_sqerr)


                att_sqerrs = np.array(att_sqerrs)
                att_covbal_l1s = np.array(att_covbal_l1s)
                att_covbal_l2s = np.array(att_covbal_l2s)
                att_covbal_linfs = np.array(att_covbal_linfs)

                att_sqerrs[att_sqerrs > 1e2] = np.nan
                att_covbal_l1s[att_covbal_l1s > 1e2] = np.nan
                att_covbal_l2s[att_covbal_l2s > 1e2] = np.nan
                att_covbal_linfs[att_covbal_linfs > 1e2] = np.nan

                ate_sqerrs = np.array(ate_sqerrs)
                ate_covbal_l1s = np.array(ate_covbal_l1s)
                ate_covbal_l2s = np.array(ate_covbal_l2s)
                ate_covbal_linfs = np.array(ate_covbal_linfs)

                ate_sqerrs[ate_sqerrs > 1e2] = np.nan
                ate_covbal_l1s[ate_covbal_l1s > 1e2] = np.nan
                ate_covbal_l2s[ate_covbal_l2s > 1e2] = np.nan
                ate_covbal_linfs[ate_covbal_linfs > 1e2] = np.nan

                np.savetxt("WC_att_sqerr_"+objective+"_"+cov_qual+"_"+str(int(prop*100))+".csv", att_sqerrs, delimiter=",")
                np.savetxt("WC_att_covbal_l1_"+objective+"_"+cov_qual+"_"+str(int(prop*100))+".csv", att_covbal_l1s, delimiter=",")
                np.savetxt("WC_att_covbal_l2_"+objective+"_"+cov_qual+"_"+str(int(prop*100))+".csv", att_covbal_l2s, delimiter=",")
                np.savetxt("WC_att_covbal_linf_"+objective+"_"+cov_qual+"_"+str(int(prop*100))+".csv", att_covbal_linfs, delimiter=",")


                np.savetxt("WC_ate_sqerr_"+objective+"_"+cov_qual+"_"+str(int(prop*100))+".csv", ate_sqerrs, delimiter=",")
                np.savetxt("WC_ate_covbal_l1_"+objective+"_"+cov_qual+"_"+str(int(prop*100))+".csv", ate_covbal_l1s, delimiter=",")
                np.savetxt("WC_ate_covbal_l2_"+objective+"_"+cov_qual+"_"+str(int(prop*100))+".csv", ate_covbal_l2s, delimiter=",")
                np.savetxt("WC_ate_covbal_linf_"+objective+"_"+cov_qual+"_"+str(int(prop*100))+".csv", ate_covbal_linfs, delimiter=",")


                att_err_vs_covbal[objective+"_err"] = np.nanmean(att_sqerrs,axis=0)
                att_err_vs_covbal[objective+"_sd"] = np.nanstd(att_sqerrs,axis=0)
                att_err_vs_covbal[objective+"_covbal_l1_mean"] = np.nanmean(att_covbal_l1s,axis=0)
                att_err_vs_covbal[objective+"_covbal_l1_sd"] = np.nanstd(att_covbal_l1s,axis=0)
                att_err_vs_covbal[objective+"_covbal_l2_mean"] = np.nanmean(att_covbal_l2s,axis=0)
                att_err_vs_covbal[objective+"_covbal_l2_sd"] = np.nanstd(att_covbal_l2s,axis=0)
                att_err_vs_covbal[objective+"_covbal_linf_mean"] = np.nanmean(att_covbal_linfs,axis=0)
                att_err_vs_covbal[objective+"_covbal_linf_sd"] = np.nanstd(att_covbal_linfs,axis=0)



                ate_err_vs_covbal[objective+"_err"] = np.nanmean(ate_sqerrs,axis=0)
                ate_err_vs_covbal[objective+"_sd"] = np.nanstd(ate_sqerrs,axis=0)
                ate_err_vs_covbal[objective+"_covbal_l1_mean"] = np.nanmean(ate_covbal_l1s,axis=0)
                ate_err_vs_covbal[objective+"_covbal_l1_sd"] = np.nanstd(ate_covbal_l1s,axis=0)
                ate_err_vs_covbal[objective+"_covbal_l2_mean"] = np.nanmean(ate_covbal_l2s,axis=0)
                ate_err_vs_covbal[objective+"_covbal_l2_sd"] = np.nanstd(ate_covbal_l2s,axis=0)
                ate_err_vs_covbal[objective+"_covbal_linf_mean"] = np.nanmean(ate_covbal_linfs,axis=0)
                ate_err_vs_covbal[objective+"_covbal_linf_sd"] = np.nanstd(ate_covbal_linfs,axis=0)

                # print("att")
                # print(att_sqerrs, att_covbal_l1s, att_covbal_l2s, att_covbal_linfs)
                # print("ate")
                # print(ate_sqerrs, ate_covbal_l1s, ate_covbal_l2s, ate_covbal_linfs)

                print(prop, cov_qual, objective)


                att_sqerr_exactbal = np.sqrt(np.nanmean(att_sqerrs[:,0]))
                att_sqerr_approxbal_l1 = np.sqrt(np.nanmean(np.array([att_sqerrs[(i,loc)] \
                    for i, loc in enumerate(np.nanargmin(att_covbal_l1s,axis=1))])))
                att_sqerr_approxbal_l2 = np.sqrt(np.nanmean(np.array([att_sqerrs[(i,loc)] \
                    for i, loc in enumerate(np.nanargmin(att_covbal_l2s,axis=1))])))
                att_sqerr_approxbal_linf = np.sqrt(np.nanmean(np.array([att_sqerrs[(i,loc)] \
                    for i, loc in enumerate(np.nanargmin(att_covbal_linfs,axis=1))])))

                print(att_sqerr_exactbal, att_sqerr_approxbal_l1, att_sqerr_approxbal_l2, att_sqerr_approxbal_linf)

                ate_sqerr_exactbal = np.sqrt(np.nanmean(ate_sqerrs[:,0]))
                ate_sqerr_approxbal_l1 = np.sqrt(np.nanmean(np.array([ate_sqerrs[(i,loc)] \
                    for i, loc in enumerate(np.nanargmin(ate_covbal_l1s,axis=1))])))
                ate_sqerr_approxbal_l2 = np.sqrt(np.nanmean(np.array([ate_sqerrs[(i,loc)] \
                    for i, loc in enumerate(np.nanargmin(ate_covbal_l2s,axis=1))])))
                ate_sqerr_approxbal_linf = np.sqrt(np.nanmean(np.array([ate_sqerrs[(i,loc)] \
                    for i, loc in enumerate(np.nanargmin(ate_covbal_linfs,axis=1))])))

                print(ate_sqerr_exactbal, ate_sqerr_approxbal_l1, ate_sqerr_approxbal_l2, ate_sqerr_approxbal_linf)

            att_err_vs_covbal.to_csv("WC_att_err_vs_covbal_"+cov_qual+".csv")
            ate_err_vs_covbal.to_csv("WC_ate_err_vs_covbal_"+cov_qual+".csv")

            ate_err_vs_covbal[ate_err_vs_covbal>1e2]=np.inf
            att_err_vs_covbal[att_err_vs_covbal>1e2]=np.inf

            fig1, ax1 = plt.subplots(figsize=(4,3))
            ax1.errorbar(ate_err_vs_covbal["tol"], \
                ate_err_vs_covbal["l1_covbal_l2_mean"], \
                # yerr=ate_err_vs_covbal["l1_covbal_l2_sd"],\
                marker="o", markersize=3, color="red", ls='--', lw=1, \
                label="Abs. Dev. Cov. Bal.")
            ax1.errorbar(ate_err_vs_covbal["tol"], \
                ate_err_vs_covbal["l1_err"], \
                # yerr=ate_err_vs_covbal["l1_sd"],\
                marker="^", markersize=3, color="red", lw=1, label="Abs. Dev. MSE")
            ax1.errorbar(ate_err_vs_covbal["tol"], \
                ate_err_vs_covbal["l2_covbal_l2_mean"], \
                # yerr=ate_err_vs_covbal["l2_covbal_l2_sd"],\
                marker="o", markersize=3, color="blue", ls='--', lw=1, label="Variance Cov. Bal.")
            ax1.errorbar(ate_err_vs_covbal["tol"], \
                ate_err_vs_covbal["l2_err"], \
                # yerr=ate_err_vs_covbal["l2_sd"],\
                marker="^", markersize=3, color="blue", lw=1, label="Variance MSE")
            ax1.errorbar(ate_err_vs_covbal["tol"], \
                ate_err_vs_covbal["entropy_covbal_l2_mean"], \
                # yerr=ate_err_vs_covbal["entropy_covbal_l2_sd"],\
                marker="o", markersize=3, color="orange", ls='--', lw=1, label="Neg. Ent. Cov. Bal.")
            ax1.errorbar(ate_err_vs_covbal["tol"], \
                ate_err_vs_covbal["entropy_err"], \
                # yerr=ate_err_vs_covbal["entropy_sd"],\
                marker="^", markersize=3, color="orange", lw=1, label="Neg. Ent. MSE")
            ax1.set_xlabel(r'$\delta$')
            ax1.set_ylabel('MSE / Cov. Bal.')
            ax1.set_xticks(np.array(ate_err_vs_covbal["tol"]))
            ax1.set_xscale('log')
            # ax1.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormateer('%.2e'))
            # ax1.set_xticklabels(np.array(ate_err_vs_covbal["tol"]), rotation='vertical')
            ax1.loglog()
            ax1.legend(fontsize="xx-small")
            plt.tight_layout()
            fig1.savefig("WC_ate_err_vs_covbal_"+cov_qual+".pdf")


            fig2, ax2 = plt.subplots(figsize=(4,3))
            ax2.errorbar(att_err_vs_covbal["tol"], \
                att_err_vs_covbal["l1_covbal_l2_mean"], \
                # yerr=att_err_vs_covbal["l1_covbal_l2_sd"],\
                marker="o", markersize=3, color="red", ls='--', lw=1, label="Abs. Dev. Cov. Bal.")
            ax2.errorbar(att_err_vs_covbal["tol"], \
                att_err_vs_covbal["l1_err"], \
                # yerr=att_err_vs_covbal["l1_sd"],\
                marker="^", markersize=3, color="red", lw=1, label="Abs. Dev. MSE")
            ax2.errorbar(att_err_vs_covbal["tol"], \
                att_err_vs_covbal["l2_covbal_l2_mean"], \
                # yerr=att_err_vs_covbal["l2_covbal_l2_sd"],\
                marker="o", markersize=3, color="blue", ls='--', lw=1, label="Variance Cov. Bal.")
            ax2.errorbar(att_err_vs_covbal["tol"], \
                att_err_vs_covbal["l2_err"], \
                # yerr=att_err_vs_covbal["l2_sd"],\
                marker="^", markersize=3, color="blue", lw=1, label="Variance MSE")
            ax2.errorbar(att_err_vs_covbal["tol"], \
                att_err_vs_covbal["entropy_covbal_l2_mean"], \
                # yerr=att_err_vs_covbal["entropy_covbal_l2_sd"],\
                marker="o", markersize=3, color="orange", ls='--', lw=1, label="Neg. Ent. Cov. Bal.")
            ax2.errorbar(att_err_vs_covbal["tol"], \
                att_err_vs_covbal["entropy_err"], \
                # yerr=att_err_vs_covbal["entropy_sd"],\
                marker="^", markersize=3, color="orange", lw=1, label="Neg. Ent. MSE")
            ax2.set_xlabel(r'$\delta$')
            ax2.set_ylabel('MSE / Cov. Bal.')
            ax2.set_xticks(np.array(att_err_vs_covbal["tol"]))
            ax2.set_xscale('log')
            # ax2.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2e'))
            # ax2.set_xticklabels(np.array(att_err_vs_covbal["tol"]), rotation='vertical')
            ax2.loglog()
            ax2.legend(fontsize="xx-small")
            plt.tight_layout()
            fig2.savefig("WC_att_err_vs_covbal_"+cov_qual+".pdf")
