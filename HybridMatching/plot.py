"""
This code plot the pt distribution and chi distribution
Can be used for parton-gun data (have to change to 1D reweighting)
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import gzip
import json
import plotly.graph_objects as go
from matplotlib.colors import LogNorm
from scipy.stats import norm, skewnorm, linregress
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab
import scipy

import networkx as nx

#For weights
def Neff(N, beta=0.9998):
    """
    Calculates the efficient number of samples in each bin
    
    N (vactor): a one or two dimensional vector containing the number of samples in each bin 
    beta (float): the probability that a new sample in that bin is independent of the previous one
    """
    
    return (1-pow(beta, N))/(1-beta)

def findWeights2d(label, pt, En, lim1, lim2):
    """
    Find the weights for a batch of samples 2D reweighting
    
    label (vector): vector of labels
    pt (vector): vactor of pt values
    En (vector): vector of number of effective samples
    lim1 (vector): vector of bin limits for pt distribution
    lim2 (vector): vector of bin limits for chi distribution
    """
    w_list = []
    for val, p in zip(label,pt):
        i = np.argmax(np.array(p/1000) < lim1)
        j = np.argmax(np.array(val < lim2))
        if En[i-1][j-1] == 0:
            w_list.append(0)
            continue
        w_list.append(1/En[i-1][j-1])
    return torch.tensor(w_list)

def findWeights(label, pt, En, lim, lim2):
    """
    Find the weights for a batch of samples 1D reweighting
    
    label (vector): vector of labels
    pt (vector): vactor of pt values
    En (vector): vector of number of effective samples
    lim1 (vector): vector of bin limits for chi distribution
    """
    w_list = []
    for val in label:
        j = np.argmax(np.array(val < lim))
        if En[j-1] == 0:
            w_list.append(0)
            continue
        w_list.append(1/En[j-1])
    return torch.tensor(w_list)

#For Plotting
def plotchi(chi_list, pt_list):
    """
    Plot chi distributions
    """
    b=25
    #To find weights
    h1 = np.histogram2d(pt_list/1000, chi_list, range=([0,1.6], [0.1,1.]), bins=b) #2D
    #h1 = np.histogram(chi_list, range=[0.1,1.], bins=b) #1D
    w = Neff(h1[0])
    w_list = findWeights2d(chi_list, pt_list, w, h1[1], h1[2]) #2D
    #w_list = findWeights(chi_list, pt_list, w, h1[1], 0) #1D
    
    #Plots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    
    axes[0].hist(chi_list, range=[0.1,1.], bins=b)
    axes[0].set_title("$\chi$ distribution ")
    axes[0].set_xlabel("True $\chi$")
    axes[0].set_ylabel("Count")
    
    axes[1].hist(chi_list, weights=w_list, range=[0.1,1.], bins=b)
    axes[1].set_title("Re-weighted $\chi$ distribution")
    axes[1].set_xlabel("True $\chi$")
    axes[1].set_ylabel("Reweighet count")

    h = axes[2].hist2d(pt_list/1000, chi_list, weights=w_list, cmap="Greens", range=([0, 1.6], [0.1, 1.]), bins=b, norm=LogNorm())
    plt.colorbar(h[3], ax=axes[2])
    axes[2].set_title("Medium-modified $p_T$ versus True $\chi$, re-weighted")
    axes[2].set_ylabel("True $\chi$")
    axes[2].set_xlabel("Medium $p_T$ (TeV)")

    fig.tight_layout()
    plt.show()
        
def plotspect(pt_list, pt_v_list, weights_list):
    """
    Plot pT distributions and gaussian fits for parton-gun data
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
    
    _, bins, _ = axes[0].hist(pt_list/1000, bins=20, density=True)#, weights=weights_list)#, range=[0, 1.6], density=1)
    mu, sigma = norm.fit(pt_list/1000)
    mu2, sigma2 = norm.fit(pt_v_list/1000)
    
    params = skewnorm.fit(pt_list/1000, 1, loc=mu, scale=sigma)
    best_fit_line = skewnorm.pdf(bins, *params)
    axes[0].plot(bins, best_fit_line, label="$\mu$ = %f \n$\sigma$ = %f \nskewness = %f" %(params[1], params[2], params[0]))

    axes[0].set_title("$p_T$ spectrum for medium jets")
    axes[0].set_xlabel("$p_T$ (TeV)")
    axes[0].set_ylabel("Count Normed")
    axes[0].legend(loc='upper left', frameon=False)
    
    _, bins2, _ = axes[1].hist(pt_v_list/1000, bins=20, density=True)
    
    params2 = skewnorm.fit(pt_v_list/1000, 1, loc=mu2, scale=sigma2)
    best_fit_line2 = skewnorm.pdf(bins2, *params2)
    axes[1].plot(bins2, best_fit_line2,label="$\mu$ = %f \n$\sigma$ = %f \nskewness = %f" %(params2[1], params2[2], params2[0]))
    print("Gauss parameters, mu=%f, sigma=%f and skewness=%f" %(params2[1], params2[2], params2[0]))
    

    axes[1].set_title("$p_{T}$ spectrum for vacuum jets")
    axes[1].set_xlabel("$p_T$ (TeV)")
    axes[1].set_ylabel("Count Normed")
    axes[1].legend(loc='upper left', frameon=False)

    
    
    h2 = np.histogram(pt_list/1000, weights=weights_list, range=[0, 1.6], bins=20)
    h3 = np.histogram(pt_v_list/1000, weights=weights_list, range=[0, 1.6], bins=20)
    ratio = h2[0]/h3[0]
    for i in range(len(ratio)):
        if str(ratio[i])=="nan":
            ratio[i] = 0

    axes[2].bar(h2[1][:-1], ratio, width=h2[1][1]-h2[1][0], align='edge')
    axes[2].set_title("Nuclear modification factor")
    axes[2].set_xlabel("$p_T$ (TeV)")
    axes[2].set_ylabel("$R_{AA}$")
    
    fig.tight_layout()
    plt.show()
    
    def logfitfunc(x, A0, pt0, n0, n1):#, n2):
    #print(A0, pt0)
    return np.log(A0)+(np.log(pt0)-x)*(n0+n1*(x-np.log(pt0)))#+n2*(np.log(x)-np.log(np.log(pt0))))
    
def plotptspect(data, data2, weights):
    """
        Plot pT distribution for jeg spectrum data that is fitted to the function in Eq. 7.1
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
    
    #Fit
    h = np.histogram(np.log(data[data>100]), weights=weights[data>100])
    bincent = [h[1][i]+((h[1][i+1]-h[1][i])/2) for i in range(len(h[0]))]
    popt, pcov = curve_fit(logfitfunc, np.array(bincent)[bincent>np.log(100)], np.log(np.array(h[0])[bincent>np.log(100)]), bounds=(1e-10, np.inf))
    offset = min(np.log(h[0]))
    dist = h[1][1]-h[1][0]

    #Plot
    axes[0][0].bar(bincent, np.log(h[0])-offset, bottom=offset, width=dist)
    axes[0][0].plot(np.array(bincent)[bincent>np.log(100)], 
                    logfitfunc(np.array(bincent)[bincent>np.log(100)], *popt), label='$A_0$ = %f, $p_{T,0}$ = %f, \n$n_0$ = %f,$n_1$ = %f ' 
                    %(popt[0], popt[1], popt[2], popt[3]), color='orange') 
    axes[0][0].set_title('$p_T$ spectrum for medium jets')
    axes[0][0].set_xlabel('log($p_T$ (GeV))')
    axes[0][0].set_ylabel('log(Count)')
    axes[0][0].legend()
    
    #Fit
    h1 = np.histogram(np.log(data2[data2>100]), weights=weights[data2>100])
    bincent1 = [h1[1][i]+((h1[1][i+1]-h1[1][i])/2) for i in range(len(h1[0]))]
    popt1, pcov = curve_fit(logfitfunc, bincent1, np.log(h1[0]), bounds=(1e-10, np.inf))
    offset1 = min(np.log(h1[0]))
    dist1 = h1[1][1]-h1[1][0]

    #Plot
    axes[0][1].bar(bincent1, np.log(h1[0])-offset1, bottom=offset1, width=dist1)
    axes[0][1].plot(bincent1, logfitfunc(bincent1, *popt1), label='$A_0$ = %f, $p_{T,0}$ = %f, \n$n_0$ = %f,$n_1$ = %f ' 
                    %(popt1[0], popt1[1], popt1[2], popt1[3]), color='green') # 
    axes[0][1].set_title('$p_T$ spectrum for vacuum jets')
    axes[0][1].set_xlabel('log($p_T$ (GeV))')
    axes[0][1].set_ylabel('log(Count)')
    axes[0][1].legend()
    
    #Ratio
    h2 = np.histogram(np.log(data[data>100]), weights=weights[data>100], bins=10) #medium
    h3 = np.histogram(np.log(data2[data2>100]), weights=weights[data2>100], bins=10) #vacuum
    ratio = h2[0]/h3[0]
    for i in range(len(ratio)):
        if str(ratio[i])=="nan":
            ratio[i] = 0
    axes[1][0].bar(h2[1][:-1], ratio, width=h2[1][1]-h2[1][0], align='edge')
    axes[1][0].set_title("$R_{AA}$ modification factor")
    axes[1][0].set_xlabel("log($p_T$ (GeV))")
    axes[1][0].set_ylabel("$R_{AA}$")
    
    #Plot both fits
    axes[1][1].plot(np.array(bincent)[bincent>np.log(100)], logfitfunc(np.array(bincent)[bincent>np.log(100)], *popt), label='Medium', color='orange')
    axes[1][1].plot(bincent1, logfitfunc(bincent1, *popt1), label='Vacuum', color='green')
    axes[1][1].set_title('Fit comparison')
    axes[1][1].set_xlabel("log($p_T$ (GeV))")
    axes[1][1].set_ylabel("log(Count)")
    axes[1][1].legend()
    fig.tight_layout()
    plt.show()

def plotratio(pt_list, pt_v_list, pt_list2, pt_v_list2, weights_list1, weights_list2):
    """
    Plotting the ratio of vacuum hadron-level jets with vacuum parton-level jets, 
    and ratio of medium hadron-level jets with medium parton-level jets.
    For either parton-gun or jet spectrum data
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    
    h2 = np.histogram(pt_list/1000, weights=weights_list1, bins=10)    #h_med
    h3 = np.histogram(pt_v_list/1000, weights=weights_list1, bins=10)  #h_vac
    h4 = np.histogram(pt_list2/1000, weights=weights_list2, bins=10)   #p_med
    h5 = np.histogram(pt_v_list2/1000, weights=weights_list2, bins=10) #p_vac
    
    r1 = h4[0]/h2[0]
    for i in range(len(r1)):
        if str(r1[i]) == "nan":
            r1[i] = 0
    wd = h2[1][1]-h2[1][0]
    axes[0].bar(h2[1][:-1], r1, width=wd, align='edge')
    axes[0].set_title("Ratio of medium-modified $p_T$ spectrums of parton and hadron level jets")
    axes[0].set_xlabel("Medium $p_T$ (TeV)")
    axes[0].set_ylabel("$\dfrac{dN_{parton}}{dp_{T}}/ \dfrac{dN_{hadron}}{dp_T}$")

    r2 = h5[0]/h3[0]
    for i in range(len(r2)):
        if str(r2[i]) == "nan":
            r2[i] = 0
    wd2 = h3[1][1]-h3[1][0]
    axes[1].bar(h3[1][:-1], r2, width=wd2, align='edge')
    axes[1].set_title("Ratio of vacuum $p_T$ spectrum of parton and hadron level jets")
    axes[1].set_xlabel("Vacuum $p_T$ (TeV)")
    axes[1].set_ylabel("$\dfrac{dN_{parton}}{dp_{T}}/ \dfrac{dN_{hadron}}{dp_T}$")

    fig.tight_layout()
    plt.show()



with gzip.open('real_hadron_sample_py.json.gz', 'r') as infile:
    data1 = infile.readlines()
with gzip.open('real_parton_sample_py.json.gz', 'r') as infile:
    data2 = infile.readlines()


chi_l = []
pt_l = []
pt_v = []
w = []

for data in [data1]:
    for line in data:
        if line==b'\n':
            continue
        j = json.loads(line.decode('utf-8'))
        chi_l.append(j[0]['chi'])
        pt_l.append(j[0]['pt'])
        pt_v.append(j[0]['pt_v'])
        w.append(j[0]['w'])

chi_l2 = []
pt_l2= []
pt_v2 = []
w2 = []

for data in [data2]:
    for line in data:
        if line==b'\n':
            continue
        j = json.loads(line.decode('utf-8'))
        chi_l2.append(j[0]['chi'])
        pt_l2.append(j[0]['pt'])
        pt_v2.append(j[0]['pt_v'])
        w2.append(j[0]['w'])
        


# Plot chi distributions 
plotchi(np.array(chi_l2), np.array(pt_l2))

# Plot pt distribution for parton-gun
#plotspect(np.array(pt_l2), np.array(pt_v2), np.array(w2))

# Plot pt distribution for jet spectrum
plotptspect(np.array(pt_l2), np.array(pt_v2),np.array(w2))

#Plot ratio plots of parton and hadron for medium and vacuum jets
plotratio(np.array(pt_l2), np.array(pt_v2), np.array(pt_l), np.array(pt_v), np.array(w2), np.array(w))