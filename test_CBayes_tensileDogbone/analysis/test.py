import numpy as np
from scipy.stats import norm # The standard Normal distribution
from scipy.stats import gaussian_kde as GKDE # A standard kernel density estimator
import matplotlib.pyplot as plt

def QoI(lam,p): # defing a QoI mapping function
    q = lam**p
    return q

# Approximate the pushforward of the prior
N = int(1E4) # number of samples from prior
lam = np.random.uniform(low=-1,high=1,size=N) # sample set of the prior

# Evaluate the two different QoI maps on this prior sample set
qvals_linear = QoI(lam,1) # Evaluate lam^1 samples
qvals_nonlinear = QoI(lam,5) # Evaluate lam^5 samples
 
# Estimate push-forward densities for each QoI
q_linear_kde = GKDE( qvals_linear ) 
q_nonlinear_kde = GKDE( qvals_nonlinear )

def rejection_sampling(r):
    N = r.size # size of proposal sample set
    check = np.random.uniform(low=0,high=1,size=N) # create random uniform weights to check r against
    M = np.max(r)
    new_r = r/M # normalize weights 
    idx = np.where(new_r>=check)[0] # rejection criterion
    return idx

# Evaluate the observed density on the QoI sample set and then compute r
obs_vals_linear = norm.pdf(qvals_linear, loc=0.25, scale=0.1)
obs_vals_nonlinear = norm.pdf(qvals_nonlinear, loc=0.25, scale=0.1)

r_linear = np.divide(obs_vals_linear,q_linear_kde(qvals_linear))
r_nonlinear = np.divide(obs_vals_nonlinear,q_nonlinear_kde(qvals_nonlinear))

# Use rejection sampling for the CBayes posterior
samples_to_keep_linear = rejection_sampling(r_linear)
post_q_linear = qvals_linear[samples_to_keep_linear]
post_lam_linear = lam[samples_to_keep_linear]

samples_to_keep_nonlinear = rejection_sampling(r_nonlinear)
post_q_nonlinear = qvals_nonlinear[samples_to_keep_nonlinear]
post_lam_nonlinear = lam[samples_to_keep_nonlinear]

# compute normalizing constants
C_linear = np.mean(obs_vals_linear) 
C_nonlinear = np.mean(obs_vals_nonlinear)

sbayes_r_linear = obs_vals_linear/C_linear
sbayes_r_nonlinear = obs_vals_nonlinear/C_nonlinear

sbayes_samples_to_keep_linear = rejection_sampling(sbayes_r_linear)
sbayes_post_q_linear = qvals_linear[sbayes_samples_to_keep_linear]
sbayes_post_lam_linear = lam[sbayes_samples_to_keep_linear]

sbayes_samples_to_keep_nonlinear = rejection_sampling(sbayes_r_nonlinear)
sbayes_post_q_nonlinear = qvals_nonlinear[sbayes_samples_to_keep_nonlinear]
sbayes_post_lam_nonlinear = lam[sbayes_samples_to_keep_nonlinear]

# Compare the observed and the pushforwards of prior, Cbayes posterior and Sbayes posterior
plt.figure()
qplot = np.linspace(-1,1, num=100)
obs_vals_plot = norm.pdf(qplot, loc=0.25, scale=0.1)

postq_lin_kde = GKDE( post_q_linear )
sb_postq_lin_kde = GKDE( sbayes_post_q_linear )

oplot = plt.plot(qplot,obs_vals_plot, 'r-', linewidth=4, label="Observed")
prplot = plt.plot(qplot,q_linear_kde(qplot),'b-', linewidth=4, label="PF of prior")
poplot = plt.plot(qplot,postq_lin_kde(qplot),'k--', linewidth=4, label="PF of CBayes posterior")
sb_poplot = plt.plot(qplot,sb_postq_lin_kde(qplot),'g--', linewidth=4, label="PF of SBayes posterior")

plt.xlim([-1,1]), plt.xlabel("Quantity of interest"), plt.legend(), plt.show()

# Compare the observed and the pushforwards of prior, Cbayes posterior and Sbayes posterior
plt.figure()
qplot = np.linspace(-1,1, num=100)
obs_vals_plot = norm.pdf(qplot, loc=0.25, scale=0.1)

postq_nl_kde = GKDE( post_q_nonlinear )
sb_postq_nl_kde = GKDE( sbayes_post_q_nonlinear )

oplot = plt.plot(qplot,obs_vals_plot, 'r-', linewidth=4, label="Observed")
prplot = plt.plot(qplot,q_nonlinear_kde(qplot),'b-', linewidth=4, label="PF of prior")
poplot = plt.plot(qplot,postq_nl_kde(qplot),'k--', linewidth=4, label="PF of Cbayes posterior")
sb_poplot = plt.plot(qplot,sb_postq_nl_kde(qplot),'g--', linewidth=4, label="PF of SBayes posterior")

plt.xlim([-1,1]), plt.xlabel("Quantity of interest"), plt.legend(), plt.show()

# Let's compute some statistics, diagnostics and information gained.

print(np.mean(post_q_linear))
print(np.sqrt(np.var(post_q_linear)))
print(np.mean(r_linear))
print(np.mean(r_linear*np.log(r_linear)))

print(np.mean(post_q_nonlinear))
print(np.sqrt(np.var(post_q_nonlinear)))
print(np.mean(r_nonlinear))
print(np.mean(r_nonlinear*np.log(r_nonlinear)))

obs_vals_nonlinear_new = norm.pdf(qvals_nonlinear, loc=0.0, scale=0.1)

r_nonlinear_new = np.divide(obs_vals_nonlinear_new,q_nonlinear_kde(qvals_nonlinear))

samples_to_keep_nonlinear_new = rejection_sampling(r_nonlinear_new)
post_q_nonlinear_new = qvals_nonlinear[samples_to_keep_nonlinear_new]
post_lam_nonlinear_new = lam[samples_to_keep_nonlinear_new]

print(np.mean(post_q_nonlinear_new))
print(np.sqrt(np.var(post_q_nonlinear_new)))
print(np.mean(r_nonlinear_new))
print(np.mean(r_nonlinear_new*np.log(r_nonlinear_new)))

# Compare the observed and the pushforwards of prior and Cbayes posterior
plt.figure()
qplot = np.linspace(-1,1, num=100)
obs_vals_plot = norm.pdf(qplot, loc=0, scale=0.1)
postq_nl_kde = GKDE( post_q_nonlinear_new )

oplot = plt.plot(qplot,obs_vals_plot, 'r-', linewidth=4, label="Observed")
prplot = plt.plot(qplot,q_nonlinear_kde(qplot),'b-', linewidth=4, label="PF of prior")
poplot = plt.plot(qplot,postq_nl_kde(qplot),'k--', linewidth=4, label="PF of Cbayes posterio r")

plt.xlim([-1,1]), plt.xlabel("Quantity of interest"), plt.legend(), plt.show() 


plt.figure()
qplot = np.linspace(-0.25,0.25, num=100)
Ns = [1E4, 1E5, 1E6] # Compute the pushforward of the prior for different sample sizes
cs = ['b','r','g']
for N,c in zip(Ns,cs):
    lam_loop = np.random.uniform(low=-1,high=1,size=int(N)) 
    qvals_nonlinear_loop = QoI(lam_loop,5)
    q_nonlinear_kde_loop = GKDE( qvals_nonlinear_loop )
    plt.plot(qplot,q_nonlinear_kde_loop(qplot),c, linewidth=2, label="PF approx, N=" + str(int(N)))
plt.plot(qplot, 1/10*np.abs(qplot)**(-4/5),'k',label='Exact PF'), plt.ylim([0,12]), plt.legend(), plt.show()

# Classical counterexample to converse of Scheffe's Theorem
from scipy.optimize import bisect
plt.figure()
xplot = np.linspace(0,1,num=int(1E3))
N = int(1E4)
ns = [1,4,10]
cs = ['g','b','k']
alphas = [0.6,0.7,0.8]
u = np.random.uniform(low=0,high=1,size=N) 
lam_loop = np.zeros(N)
for n,c,a in zip(ns,cs,alphas):
    def f_n(x): # The pdf
        return (1-np.cos(2*np.pi*n*x))
    plt.subplot(1,2,1), plt.plot(xplot,f_n(xplot), c, linewidth=2, label='pdf for $f_n$, $n$='+str(n),alpha=a)
    for i in range(N): # Sample from r.v. with the pdf
        def F_n(x): # The cdf
            return (x-1/(2*np.pi*n)*np.sin(2*np.pi*n*x)-u[i])
        lam_loop[i] = bisect(F_n,0,1)
    plt.subplot(1,2,2), plt.hist(lam_loop, color=c, label='samples from $f_n$, $n$='+str(n),alpha=a)
plt.subplot(1,2,1), plt.legend(), plt.subplot(1,2,2), plt.legend()

