"""
whatt?
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

def Neff(N):
    beta = 0.9998
    N_eff = (1-pow(beta, N))/(1-beta)
    return N_eff

def getErrorbars(h):
    x_list=[]
    y_list=[]
    stdv_list = []
    for p,x in enumerate(np.array(h[0])):
        if (x>0).any():
            if len(x[x>0])>1:
                w = x[x>0]/np.sum(x[x>0])
            elif sum(x[x>0]) > 1:
                w = x[x>0]/sum(x[x>0])
            else:
                w = x[x>0]/len(x[x>0])

            y_lims = h[2][:-1][x>0]
            su = []
            for k,l in enumerate(y_lims):
                su.append(w[k]*(l+0.012))
            k = sum(su)/sum(w)
            su_sd=[]
            for m, l in enumerate(y_lims):
                su_sd.append(w[m]*(l+0.012-k)**2)
            st = sum(su_sd)/(sum(w))
            #print(st)
            stdv_list.append(np.sqrt(st))
            x_list.append((h[1][p]+h[1][p+1])/2)
            y_list.append(k)
    return x_list, y_list, stdv_list


def findWeights2d(label, pt, En, lim1, lim2):
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
    w_list = []
    for val in label:
        j = np.argmax(np.array(val < lim))
        if En[j-1] == 0:
            w_list.append(0)
            continue
        w_list.append(1/En[j-1])
    return torch.tensor(w_list)


def plotchi(chi_list, pt_list, pt_v_list, weights_list, constituents):
        b=25
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
        h1 = np.histogram2d(pt_list/1000, chi_list, range=([0,1.6], [0.1,1.]), bins=b)#, weights=weights_list)
        #h1 = np.histogram(chi_list, range=[0.1,1.], bins=b)
        w = Neff(h1[0])
        
        #print(w)
        
        w_list = findWeights2d(chi_list, pt_list, w, h1[1], h1[2])
            
        
        #print(w_list)
        axes[0].hist(chi_list, range=[0.1,1.], bins=b)
        axes[0].set_title("$\chi_{jh}$ distribution ")
        axes[0].set_xlabel("True $\chi_{jh}$")
        axes[0].set_ylabel("Count")
        
        axes[1].hist(chi_list, weights=w_list, range=[0.1,1.], bins=b)
        axes[1].set_title("Re-weighted $\chi_{jh}$ distribution")
        axes[1].set_xlabel("True $\chi_{jh}$")
        axes[1].set_ylabel("Reweighet count")
    
        
        h = axes[2].hist2d(pt_list/1000, chi_list, weights=w_list, cmap="Greens", range=([0, 1.6], [0.1, 1.]), bins=b, norm=LogNorm())
        plt.colorbar(h[3], ax=axes[2])
        axes[2].set_title("Medium-modified $p_T$ versus True $\chi_{jh}$, re-weighted")
        axes[2].set_ylabel("True $\chi_{jh}$")
        axes[2].set_xlabel("Medium $p_T$ (TeV)")

        axes[3].hist(constituents)
        
        fig.tight_layout()
        plt.show()
        
        
def plotspect(pt_list, pt_v_list, weights_list):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
    
    
    #mu = np.average(pt_list/1000, weights=weights_list)
    #sigma = np.sqrt(np.average((pt_list/1000 - mu)**2, weights=weights_list))
    #print(mu, sigma)
    
    _, bins, _ = axes[0].hist(pt_list/1000, bins=20, density=True)#, weights=weights_list)#, range=[0, 1.6], density=1)
    mu, sigma = norm.fit(pt_list/1000)
    mu2, sigma2 = norm.fit(pt_v_list/1000)
    
    params = skewnorm.fit(pt_list/1000, 1, loc=mu, scale=sigma)
    #best_fit_line = norm.pdf(bins, mu, sigma)
    best_fit_line = skewnorm.pdf(bins, *params)
    axes[0].plot(bins, best_fit_line, label="$\mu$ = %f \n$\sigma$ = %f \nskewness = %f" %(params[1], params[2], params[0]))
    #print("Gauss parameters, mu=%f, sigma=%f and skewness=%f" %(params[1], params[2], params[0]))
    #y = norm.pdf( pt_list/1000)
    #l = axes[0].plot(pt_list/1000, y, 'r--', linewidth=2, )

    axes[0].set_title("$p_T$ spectrum for medium jets")
    axes[0].set_xlabel("$p_T$ (TeV)")
    axes[0].set_ylabel("Count Normed")
    axes[0].legend(loc='upper left', frameon=False)
    #axes[0].set_yscale('log')
    #axes[0].set_xscale('log')
    
    _, bins2, _ = axes[1].hist(pt_v_list/1000, bins=20, density=True)#, range=[0, 1.6])#, weights=weights_list)#, density=1)
    
    params2 = skewnorm.fit(pt_v_list/1000, 1, loc=mu2, scale=sigma2)
    best_fit_line2 = skewnorm.pdf(bins2, *params2)
    axes[1].plot(bins2, best_fit_line2,label="$\mu$ = %f \n$\sigma$ = %f \nskewness = %f" %(params2[1], params2[2], params2[0]))
    print("Gauss parameters, mu=%f, sigma=%f and skewness=%f" %(params2[1], params2[2], params2[0]))
    
    #axes[1].hist(pt_v_list/1000, weights=weights_list, range=[0, 1.6], bins=25)
    axes[1].set_title("$p_{T}$ spectrum for vacuum jets")
    axes[1].set_xlabel("$p_T$ (TeV)")
    axes[1].set_ylabel("Count Normed")
    axes[1].legend(loc='upper left', frameon=False)
    #axes[1].set_yscale('log')
    #axes[1].set_xscale('log')
    
    
    h2 = np.histogram(pt_list/1000, weights=weights_list, range=[0, 1.6], bins=20)
    h3 = np.histogram(pt_v_list/1000, weights=weights_list, range=[0, 1.6], bins=20)
    ratio = h2[0]/h3[0]
    for i in range(len(ratio)):
        if str(ratio[i])=="nan":
            ratio[i] = 0
    #print(h2[1][1]-h2[1][0])
    axes[2].bar(h2[1][:-1], ratio, width=h2[1][1]-h2[1][0], align='edge')
    axes[2].set_title("Nuclear modification factor")
    axes[2].set_xlabel("$p_T$ (TeV)")
    axes[2].set_ylabel("$R_{AA}$")#$\dfrac{d\sigma}{dp_{T,medium}}/\dfrac{d\sigma}{dp_{T,vacuum}}$")
    
    #axes[2].plot(bins, best_fit_line, label="Medium", color='green')
    #axes[2].plot(bins2, best_fit_line2,label="Vacuum", color='orange')
    #axes[2].set_title("Fit comparison")
    #axes[2].set_xlabel("True $p_T$ (TeV)")
    #axes[2].set_ylabel("Count Normed")
    #axes[2].legend()
    
    
    fig.tight_layout()
    plt.show()
    

def plotratio(pt_list, pt_v_list, pt_list2, pt_v_list2, weights_list1, weights_list2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    
    
    
    h2 = np.histogram(pt_list/1000, weights=weights_list1, bins=10)#, range=[0, 1.7]) #h_med
    h3 = np.histogram(pt_v_list/1000, weights=weights_list1, bins=10)#, range=[0, 1.7]) #h_vac
    h4 = np.histogram(pt_list2/1000, weights=weights_list2, bins=10)#, range=[0, 1.7]) #p_med
    h5 = np.histogram(pt_v_list2/1000, weights=weights_list2, bins=10)#, range=[0, 1.7]) #p_vac
    
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
    """_summary_
    
    r3 = h4[0]/h5[0]
    for i in range(len(r3)):
        if str(r3[i]) == "nan":
            r3[i] = 0

    axes[2].bar(h5[1][:-1], r3, width=0.15, align='edge')
    axes[2].set_title("Ratio of pt spectrum p_med/p_vac")
    axes[2].set_xlabel("$p_T$ (TeV)")
    axes[2].set_ylabel("$\dfrac{p_{T,med}}{p_{T,vac}}$")
    
    r4 = h2[0]/h3[0]
    for i in range(len(r4)):
        if str(r4[i]) == "nan":
            r4[i] = 0

    axes[3].bar(h3[1][:-1], r4, width=0.15, align='edge')
    axes[3].set_title("Ratio of pt spectrum h_med/h_vac ")
    axes[3].set_xlabel("$p_T$ (TeV)")
    axes[3].set_ylabel("$\dfrac{p_{T,med}}{p_{T,vac}}$")

    """
    fig.tight_layout()
    plt.show()

def jointplot(chi_list, pt_list):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
        h = axes[0].hist2d( pt_list/1000,chi_list, density=True, bins=15)
        plt.colorbar(h[3])
        axes[0].set_title("Joint histogram")
        axes[0].set_ylabel("True $\chi_{jh}$")
        axes[0].set_xlabel("True $p_T$ (TeV)")
        #axes[0].set_xlim([0, 1.1])
        axes[1].hist(pt_list/1000, density=True, bins=15)
        axes[1].set_title("Histogram for Jet pt")
        axes[1].set_xlabel("True $p_T$ (TeV)")
        #axes[1].set_yscale('log')
        
        fig.tight_layout()
        plt.show()

def plotConstits(medium_cons, vacuum_cons):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    up = 20
    axes[0].hist(medium_cons, bins=20, range=[0, up])
    axes[0].set_title("Number of constituents for medium jets")
    axes[0].set_xlabel("Number of Jet Constituents")
    axes[0].set_ylabel("Count")
    
    
    axes[1].hist(vacuum_cons, bins=20, range=[0, up])
    axes[1].set_title("Number of constituents for vacuum jets")
    axes[1].set_xlabel("Number of Jet Constituents")
    axes[1].set_ylabel("Count")
    
    fig.tight_layout()
    plt.show()
    

def logfitfunc(x, A0, pt0, n0, n1):#, n2):
    #print(A0, pt0)
    return np.log(A0)+(np.log(pt0)-x)*(n0+n1*(x-np.log(pt0)))#+n2*(np.log(x)-np.log(np.log(pt0))))
    
def funct(data, data2, weights):
    #h = np.histogram(data, bins=np.logspace(np.log10(min(data)),np.log10(max(data)),10))
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
    
    h = np.histogram(np.log(data[data>100]), weights=weights[data>100])
    bincent = [h[1][i]+((h[1][i+1]-h[1][i])/2) for i in range(len(h[0]))]
    print(np.array(bincent)[bincent>np.log(100)])
    popt, pcov = curve_fit(logfitfunc, np.array(bincent)[bincent>np.log(100)], np.log(np.array(h[0])[bincent>np.log(100)]), bounds=(1e-10, np.inf))
    offset = min(np.log(h[0]))
    dist = h[1][1]-h[1][0]

    axes[0][0].bar(bincent, np.log(h[0])-offset, bottom=offset, width=dist)
    axes[0][0].plot(np.array(bincent)[bincent>np.log(100)], logfitfunc(np.array(bincent)[bincent>np.log(100)], *popt), label='$A_0$ = %f, $p_{T,0}$ = %f, \n$n_0$ = %f,$n_1$ = %f ' %(popt[0], popt[1], popt[2], popt[3]), color='orange') # 
    axes[0][0].set_title('$p_T$ spectrum for medium jets')
    axes[0][0].set_xlabel('log($p_T$ (GeV))')
    axes[0][0].set_ylabel('log(Count)')
    #axes[0].scatter(bincent, np.log(h[0]), color='r')
    axes[0][0].legend()
    
    h1 = np.histogram(np.log(data2[data2>100]), weights=weights[data2>100])
    bincent1 = [h1[1][i]+((h1[1][i+1]-h1[1][i])/2) for i in range(len(h1[0]))]
    popt1, pcov = curve_fit(logfitfunc, bincent1, np.log(h1[0]), bounds=(1e-10, np.inf))
    offset1 = min(np.log(h1[0]))
    dist1 = h1[1][1]-h1[1][0]

    axes[0][1].bar(bincent1, np.log(h1[0])-offset1, bottom=offset1, width=dist1)
    axes[0][1].plot(bincent1, logfitfunc(bincent1, *popt1), label='$A_0$ = %f, $p_{T,0}$ = %f, \n$n_0$ = %f,$n_1$ = %f ' %(popt1[0], popt1[1], popt1[2], popt1[3]), color='green') # 
    axes[0][1].set_title('$p_T$ spectrum for vacuum jets')
    axes[0][1].set_xlabel('log($p_T$ (GeV))')
    axes[0][1].set_ylabel('log(Count)')
    #axes[1].scatter(bincent1, np.log(h1[0]), color='r')
    axes[0][1].legend()
    
    
    h2 = np.histogram(np.log(data[data>100]), weights=weights[data>100], bins=10) #medium
    h3 = np.histogram(np.log(data2[data2>100]), weights=weights[data2>100], bins=10) #vacuum
    ratio = h2[0]/h3[0]
    for i in range(len(ratio)):
        if str(ratio[i])=="nan":
            ratio[i] = 0
    #print(h2[1][1]-h2[1][0])
    axes[1][0].bar(h2[1][:-1], ratio, width=h2[1][1]-h2[1][0], align='edge')
    axes[1][0].set_title("$R_{AA}$ modification factor")
    axes[1][0].set_xlabel("log($p_T$ (GeV))")
    axes[1][0].set_ylabel("$R_{AA}$")
    
    
    axes[1][1].plot(np.array(bincent)[bincent>np.log(100)], logfitfunc(np.array(bincent)[bincent>np.log(100)], *popt), label='Medium', color='orange')
    axes[1][1].plot(bincent1, logfitfunc(bincent1, *popt1), label='Vacuum', color='green')
    axes[1][1].set_title('Fit comparison')
    axes[1][1].set_xlabel("log($p_T$ (GeV))")
    axes[1][1].set_ylabel("log(Count)")
    axes[1][1].legend()
    fig.tight_layout()
    plt.show()


def functratio(data, data2, data_v, data2_v, weights, weights2):
    #h = np.histogram(data, bins=np.logspace(np.log10(min(data)),np.log10(max(data)),10))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    h = np.histogram(np.log(data[data>100]), weights=weights[data>100])
    bincent = [h[1][i]+((h[1][i+1]-h[1][i])/2) for i in range(len(h[0]))]
    popt, pcov = curve_fit(logfitfunc, np.array(bincent)[bincent>np.log(100)], np.log(np.array(h[0])[bincent>np.log(100)]), bounds=(1e-10, np.inf))


    h1 = np.histogram(np.log(data2[data2>100]), weights=weights2[data2>100])
    bincent1 = [h1[1][i]+((h1[1][i+1]-h1[1][i])/2) for i in range(len(h1[0]))]
    popt1, pcov = curve_fit(logfitfunc, bincent1, np.log(h1[0]), bounds=(1e-10, np.inf))

    
    h3 = np.histogram(np.log(data_v[data_v>100]), weights=weights[data_v>100])
    bincent3 = [h3[1][i]+((h3[1][i+1]-h[1][i])/2) for i in range(len(h3[0]))]
    popt3, pcov = curve_fit(logfitfunc, np.array(bincent3)[bincent3>np.log(100)], np.log(np.array(h3[0])[bincent3>np.log(100)]), bounds=(1e-10, np.inf))


    h4 = np.histogram(np.log(data2_v[data2_v>100]), weights=weights2[data2_v>100])
    bincent4 = [h4[1][i]+((h4[1][i+1]-h4[1][i])/2) for i in range(len(h4[0]))]
    popt4, pcov = curve_fit(logfitfunc, bincent4, np.log(h4[0]), bounds=(1e-10, np.inf))

    print(popt)
    print(popt1)
    axes[0].plot(bincent1, logfitfunc(bincent1, *popt1), label='Parton level jets', color='green', alpha=0.5)
    axes[0].plot(np.array(bincent)[bincent>np.log(100)], logfitfunc(np.array(bincent)[bincent>np.log(100)], *popt), label='Hadron level jets', color='orange', alpha=0.5) # 
    
    axes[0].set_title('$p_T$ spectrum for medium jets')
    axes[0].set_xlabel('log($p_T$ (GeV))')
    axes[0].set_ylabel('log(Count)')
    axes[0].legend()
    
    axes[1].plot(np.array(bincent3)[bincent3>np.log(100)], logfitfunc(np.array(bincent3)[bincent3>np.log(100)], *popt3), label='Hadron level jets', color='orange') # 
    axes[1].plot(bincent4, logfitfunc(bincent4, *popt4), label='Parton level jets', color='green')
    axes[1].set_title('$p_T$ spectrum for vacuum jets')
    axes[1].set_xlabel('log($p_T$ (GeV))')
    axes[1].set_ylabel('log(Count)')
    axes[1].legend()
    
    
    fig.tight_layout()
    plt.show()
    

with gzip.open('real_parton_sample_py.json.gz', 'r') as infile:
    data1 = infile.readlines()
with gzip.open('real_hadron_sample_py_parton.json.gz', 'r') as infile:
    data2 = infile.readlines()
#with gzip.open('test_sample_hadron.json.gz', 'r') as infile:
#    data3 = infile.readlines()

chi_l = []
pt_l = []
pt_v = []
w = []
data_full = [data1]#, data2, data3]

for data in data_full:
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
constits = []
med_cons = []
vac_cons = []
for data in [data2]:
    for line in data:
        if line==b'\n':
            continue
        j = json.loads(line.decode('utf-8'))
        #constits.append(len(j)-1)
        chi_l2.append(j[0]['chi'])
        pt_l2.append(j[0]['pt'])
        pt_v2.append(j[0]['pt_v'])
        med_cons.append(j[0]['nr'])
        vac_cons.append(j[0]['nr_v'])
        w2.append(j[0]['w'])
        
#if (np.array(w2)!=np.array(w)).any():
#    print("Not the same")
    
print(min(chi_l2))
print(min(pt_l2))
print(max(pt_l2))
print(max(pt_v2))
print(max(med_cons))
print(len(chi_l2))
print(len(chi_l))
#jointplot(np.array(chi_l), np.array(pt_l))
plotchi(np.array(chi_l2), np.array(pt_l2), np.array(pt_v2), np.array(w2), np.array(med_cons))
#plotspect(np.array(pt_l2), np.array(pt_v2), np.array(w2))
#funct(np.array(pt_l2), np.array(pt_v2),np.array(w2))
#functratio(np.array(pt_l2), np.array(pt_l), np.array(pt_v2), np.array(pt_v), np.array(w2), np.array(w))
#plotratio(np.array(pt_l2), np.array(pt_v2), np.array(pt_l), np.array(pt_v), np.array(w2), np.array(w))

#plotConstits(med_cons, vac_cons)


