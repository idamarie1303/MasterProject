"""
whatt?
"""
import numpy as np
import fastjet as fj
import matplotlib.pyplot as plt
import torch
import gzip
import json
import plotly.graph_objects as go
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable

import networkx as nx

jet_def = fj.JetDefinition(fj.cambridge_algorithm, 1000.)

def Neff(N):
    beta = 0.99998
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


def plotchi(chi_list, pt_list, pt_v_list, weights_list, constituents):

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 6))
        h1 = np.histogram2d(pt_list/1000, chi_list, range=([0,2.], [0,1.]))#, weights=weights_list)
        #h1 = np.histogram(chi_list)
        w = Neff(h1[0])

        w_list = []
        for val,pt in zip(chi_list, pt_list):
            i = np.argmax(np.array(val < h1[2]))
            j = np.argmax(np.array(pt/1000) < h1[1])
            if w[j-1][i-1] == 0:
                w_list.append(0)
                continue
            w_list.append(1/w[j-1][i-1])#[i-1])
            
        
        #print(w_list)
        axes[0].hist(chi_list, weights=weights_list)
        axes[0].set_title("Histogram for True $\chi_{jh}$")
        axes[0].set_xlabel("True $\chi_{jh}$")
        axes[0].set_ylabel("Count")
        
        axes[1].hist(chi_list, weights=w_list)
        axes[1].set_title("Histogram for reweighted True $\chi_{jh}$")
        axes[1].set_xlabel("True $\chi_{jh}$")
        axes[1].set_ylabel("Reweighet count")
        
        
        h = axes[2].hist2d(pt_list/1000, chi_list, weights=w_list, cmap="Greens", range=([0, 1.7], [0, 1.]), norm=LogNorm())
        plt.colorbar(h[3])
        axes[2].set_title("Joint histogram")
        axes[2].set_ylabel("True $\chi_{jh}$")
        axes[2].set_xlabel("True $p_T$ (TeV)")

        #x_list, y_list, stdv_list = getErrorbars(h)
        #axes[2].plot(x_list, y_list, color='r')
        #axes[2].errorbar(x_list,y_list, yerr=stdv_list, xerr=0.05,color='r', fmt='o', capsize=4, markersize=2)
        
        axes[3].hist(constituents, range=[1,10], bins=11)
        
        
        fig.tight_layout()
        plt.show()
        
        
def plotspect(pt_list, weights_list, pm, cons, theta, z, kt):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    #print(pt_list)
    axes[0][0].hist(pt_list, range=[0, max(pt_list)], bins=50, weights=weights_list)
    axes[0][0].set_title("$p_T$ spectrum")
    axes[0][0].set_xlabel("$p_T$ (TeV)")
    axes[0][0].set_ylabel("Count")
    #axes[0].set_yscale('log')
    #axes[0].set_xscale('log')
    print(max(kt))
    axes[0][1].hist(kt, weights=weights_list,bins=20)
    axes[0][1].set_title("$k_{T,max}$")
    axes[0][1].set_xlabel("$k_{T,max}$")
    axes[0][1].set_ylabel("Count")
    axes[0][1].set_yscale('log')
    
    axes[1][0].hist(theta,weights=weights_list, bins=25)
    axes[1][0].set_title("$\Theta$ distribution")
    axes[1][0].set_xlabel("$\Theta$")
    axes[1][0].set_ylabel("Count")
    
    axes[1][1].hist(pm,weights=weights_list, bins=25)
    axes[1][1].set_title("Mass of jet")
    axes[1][1].set_xlabel("m")
    axes[1][1].set_ylabel("Count")
    
    
    fig.tight_layout()
    plt.show()
    

def plotkorr(theta, nrCons):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    r = 20
    bins=21
    
    pl1 = nrCons[theta<0.1]
    axes[0][0].hist(pl1, range=[0,r], bins=bins)
    axes[0][0].set_title("$\Theta$ < 0.1")
    axes[0][0].set_xlabel("Nr. of Constituents")
    axes[0][0].set_ylabel("Count")
    #print(theta < 0.3)
    pl2 = nrCons[(0.1 < theta) & (theta < 0.2)]
    axes[0][1].hist(pl2, range=[0,r], bins=bins)
    axes[0][1].set_title("0.1 < $\Theta$ < 0.2")
    axes[0][1].set_xlabel("Nr. of Constituents")
    axes[0][1].set_ylabel("Count")
    
    pl3 = nrCons[(0.2 < theta) & (theta < 0.3)]
    axes[1][0].hist(pl3, range=[0,r], bins=bins)
    axes[1][0].set_title("0.3 < $\Theta$ < 0.2")
    axes[1][0].set_xlabel("Nr. of Constituents")
    axes[1][0].set_ylabel("Count")
    
    pl4 = nrCons[0.3 < theta]
    axes[1][1].hist(pl4, range=[0,r], bins=bins)
    axes[1][1].set_title("$\Theta$ > 0.3")
    axes[1][1].set_xlabel("Nr. of Constituents")
    axes[1][1].set_ylabel("Count")
    
    fig.tight_layout()
    plt.show()


def plotkorr2(theta, kt, chi, cons, weights):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))
    r = max(kt)
    bins=21
    lim1 = 0.90
    lim2 = 0.95
    lim3 = 1.0
    
    pl1 = kt[chi<lim1]
    pl2 = kt[(lim1 < chi) & (chi < lim2)]
    pl3 = kt[(lim2 < chi) & (chi < lim3)]
    pl4 = kt[lim3 == chi]
    axes[0].hist(kt, range=[0,r], bins=bins, weights=weights, label='Full $\chi$ range', color='black', histtype='step')#, density=True)
    axes[0].hist(pl1, range=[0,r], bins=bins, weights=weights[chi<lim1], label="$\chi$ < 0.4", color='blue', histtype='step')#, density=True)
    axes[0].hist(pl2, range=[0,r], bins=bins, weights=weights[(lim1 < chi) & (chi < lim2)], label="0.4 < $\chi$ < 0.6", color='red', histtype='step')#, density=True)
    axes[0].hist(pl3, range=[0,r], bins=bins, weights=weights[(lim2 < chi) & (chi < lim3)], label ="0.6 < $\chi$ < 0.8", color='green', histtype='step')
    axes[0].hist(pl4, range=[0,r], bins=bins, weights=weights[lim3 == chi], label="$\chi$ > 0.8", color='orange', histtype='step')#, density=True)
    axes[0].set_xlabel("m")#$k_{T,max}$")#"Nr. of constituents")#"z")#"$\Theta$")#"$k_{T,0}$")#
    axes[0].set_ylabel("Count")
    axes[0].set_title('Histogram for $m$ w/Weights')
    axes[0].set_yscale('log')
    axes[0].legend()
    
    r=max(theta)
    pl5 = theta[chi<lim1]
    pl6 = theta[(lim1 < chi) & (chi < lim2)]
    pl7 = theta[(lim2 < chi) & (chi < lim3)]
    pl8 = theta[lim3 == chi]
    axes[1].hist(theta, range=[0,r], bins=bins, weights=weights, label='Full $\chi$ range', color='black', histtype='step')#, density=True)
    axes[1].hist(pl5, range=[0,r], bins=bins, weights=weights[chi<lim1], label="$\chi$ < 0.4", color='blue', histtype='step')#, density=True)
    axes[1].hist(pl6, range=[0,r], bins=bins, weights=weights[(lim1 < chi) & (chi < lim2)], label="0.4 < $\chi$ < 0.6", color='red', histtype='step')#, density=True)
    axes[1].hist(pl7, range=[0,r], bins=bins, weights=weights[(lim2 < chi) & (chi < lim3)], label ="0.6 < $\chi$ < 0.8", color='green', histtype='step')#, density=True)
    axes[1].hist(pl8, range=[0,r], bins=bins, weights=weights[lim3 == chi], label="$\chi$ > 0.8", color='orange', histtype='step')#, density=True)
    axes[1].set_xlabel("$z$")#"Nr. of constituents")#"z")#"$\Theta$")#"$k_{T,0}$")#
    axes[1].set_ylabel("Count")
    axes[1].set_yscale('log')
    axes[1].set_title('Histogram for $z$ w/Weights')
    axes[1].legend()
    
    r=max(cons)
    pl5 = cons[chi<lim1]
    pl6 = cons[(lim1 < chi) & (chi < lim2)]
    pl7 = cons[(lim2 < chi) & (chi < lim3)]
    pl8 = cons[lim3 == chi]
    axes[2].hist(cons, range=[0,r], bins=bins, weights=weights, label='Full $\chi$ range', color='black', histtype='step')#, density=True)
    axes[2].hist(pl5, range=[0,r], bins=bins, weights=weights[chi<lim1], label="$\chi$ < 0.4", color='blue', histtype='step')#, density=True)
    axes[2].hist(pl6, range=[0,r], bins=bins, weights=weights[(lim1 < chi) & (chi < lim2)], label="0.4 < $\chi$ < 0.6", color='red', histtype='step')#, density=True)
    axes[2].hist(pl7, range=[0,r], bins=bins, weights=weights[(lim2 < chi) & (chi < lim3)], label ="0.6 < $\chi$ < 0.8", color='green', histtype='step')#, density=True)
    axes[2].hist(pl8, range=[0,r], bins=bins, weights=weights[lim3 == chi], label="$\chi$ > 0.8", color='orange', histtype='step')#, density=True)
    axes[2].set_xlabel("Number of constituents")#"Nr. of constituents")#"z")#"$\Theta$")#"$k_{T,0}$")#
    axes[2].set_ylabel("Count")
    axes[2].set_yscale('log')
    axes[2].set_title('Histogram for number of jet constituents w/Weights')
    axes[2].legend()
    
    fig.tight_layout()
    plt.show()


def plotkt(kt_med, kt_vac, chi, pt):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    #print(pt_list)
    b = 25
    axes[0][0].hist(kt_med, bins=b)
    axes[0][0].set_title("$k_{T,0}$ spectrum medium")
    axes[0][0].set_xlabel("$k_{T,0}$")
    axes[0][0].set_ylabel("Count")


    axes[0][1].hist(kt_vac, bins=b)
    axes[0][1].set_title("$k_{T,0}$ spectrum vacuum")
    axes[0][1].set_xlabel("$k_{T,0}$")
    axes[0][1].set_ylabel("Count")
    
    h1 = np.histogram(chi, bins=b)
    
    
    chi2 = []
    for k1,k2 in zip(kt_med, pt):
        chi2.append(k1/k2)

    
    h2 = axes[1][0].hist(np.array(chi2), bins=b)
    axes[1][0].set_title("$\chi_{k_{T,0}}$ distribution")
    axes[1][0].set_xlabel("$\chi_{k_{T,0}}$")
    axes[1][0].set_ylabel("Count")
    
    r1 = h2[0]/h1[0]
    for i in range(len(r1)):
        if str(r1[i]) == "nan":
            r1[i] = 0
    dist = h2[1][1]- h2[1][0]
    axes[1][1].bar(h2[1][:-1], r1, width=dist, align='edge')
    axes[1][1].set_title("$\chi / \chi_{k_{T,0}}$ ratio")
    axes[1][1].set_xlabel("$\chi$")
    axes[1][1].set_ylabel("$\chi / \chi_{k_{T,0}}$")

    
    fig.tight_layout()
    plt.show()
    

def logfitfunc(x, A0, pt0, n0, n1):#, n2):
    #print(A0, pt0)
    return np.log(A0)+(np.log(pt0)-x)*(n0+n1*(x-np.log(pt0)))#+n2*(np.log(x)-np.log(np.log(pt0))))
    
def funct(data):
    #h = np.histogram(data, bins=np.logspace(np.log10(min(data)),np.log10(max(data)),10))
    
    h = np.histogram(np.log(data[data>100]))
    
    bincent = [h[1][i]+((h[1][i+1]-h[1][i])/2) for i in range(len(h[0]))]
    print(h[0])
    
    popt, pcov = curve_fit(logfitfunc, bincent, np.log(h[0]), bounds=(1e-10, np.inf))
    print(popt)
    
    #z= np.polyfit(np.log(bincent), np.log(h[0]), 3)
    #print(z)
    #p = np.poly1d(z)
    
    
    dist = h[1][1]-h[1][0]
    plt.bar(bincent, np.log(h[0]), width=dist)
    plt.plot(bincent, logfitfunc(bincent, *popt), label='$A_0$ = %f, $p_{T,0}$ = %f, \n$n_0$ = %f,$n_1$ = %f ' %(popt[0], popt[1], popt[2], popt[3]), color='r') # 
    plt.title('Fitted to the function $A_0 (\dfrac{p_{T,0}}{p_T})^{n(p_T)}$')
    plt.xlabel('log($p_T$ (GeV))')
    plt.ylabel('log(Count)')
    plt.scatter(bincent, np.log(h[0]), color='r')
    plt.legend()
    plt.show()

def findmaxkt(jet):
    currentlist = jet.exclusive_subjets(2)
    lastjet = jet
    theta = []
    kt = []

    done = False
    nrdeclust = 0
    while (done==False):
        nrdeclust +=1
        theta.append(currentlist[0].delta_R(currentlist[1]))
        z2 = min(currentlist[0].pt(), currentlist[1].pt())/(currentlist[0].pt() + currentlist[1].pt())
        kt.append(z2*(1-z2)*currentlist[0].delta_R(currentlist[1])*lastjet.pt())
        
        if (currentlist[0].pt() > currentlist[1].pt()):
            if len(currentlist[0].exclusive_subjets_up_to(2))<2:
                done=True
                continue
        
            #continue with this jet
            currentlist = currentlist[0].exclusive_subjets(2)
            lastjet = currentlist[0]
        
        if (currentlist[1].pt() > currentlist[0].pt()):
            if len(currentlist[0].exclusive_subjets_up_to(2))<2:
                done = True
                continue
            
            #continue with this jet
            currentlist = currentlist[1].exclusive_subjets(2)
    
    return max(kt)

def findmaxkt2(jet):
    listl = [jet]
    theta = []
    kt = []
    for i,l in enumerate(listl):
        
        if len(l.exclusive_subjets_up_to(2))==2:
            the = l.exclusive_subjets(2)[0].delta_R(l.exclusive_subjets(2)[1])
            theta.append(the)
            z2 = min(l.exclusive_subjets(2)[0].pt(), l.exclusive_subjets(2)[1].pt())/(l.exclusive_subjets(2)[0].pt() + l.exclusive_subjets(2)[1].pt())
            
            kt.append(z2*(1-z2)*the*l.pt())
            #print(z2*(1-z2)*the*l.pt())
            listl.append(l.exclusive_subjets(2)[0])
            listl.append(l.exclusive_subjets(2)[1])
    #print(len(listl))
    #print(max(kt))
    return kt


def SoftDrop(jet, z_cut, beta, R):
    finaljet = [jet]
    done=False
    nrGroom = 0
    while(done==False):
        listsubjet = finaljet[0].exclusive_subjets(2)
        dR = listsubjet[0].delta_R(listsubjet[1])
        z3 = min(listsubjet[0].pt(), listsubjet[1].pt())/(listsubjet[0].pt() + listsubjet[1].pt())
        sdcond = z_cut*pow(dR/R, beta)
        if (z3>sdcond):
            done=True
        else:
            #Not done
            if (listsubjet[0].pt() > listsubjet[1].pt()):
                if (listsubjet[0].has_exclusive_subjets() == False):
                    #no children
                    done=True
                elif len(listsubjet[0].exclusive_subjets_up_to(2)) < 2:
                    done=True
                    
                else:
                    #continue with this subjet
                    nrGroom +=1
                    finaljet=[listsubjet[0]]
                
            else:
                if (listsubjet[1].has_exclusive_subjets() == False):
                    #no children
                    done=True
                elif len(listsubjet[1].exclusive_subjets_up_to(2)) < 2:
                    done=True
                
                else:
                    #continue with this subjet
                    nrGroom += 1
                    finaljet=[listsubjet[1]]
                
    return finaljet

def plotish(chi, kt, kt_v, theta, theta_v, z, z_v, w, w_v):
    bins=20
    r = 0.8#max(kt_v)
    lim1 = 0.25
    lim2 = 0.5
    lim3 = 0.75
    lim4 = 0.85
    lim5 = 0.95
    lim6 = 1
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    
    divider = make_axes_locatable(axes[0])
    ax2 = divider.append_axes("bottom", size="40%", pad=0)
    axes[0].figure.add_axes(ax2)
    h = axes[0].hist(kt_v, bins=bins, range=[0,r], histtype='step', color='black', label='Vacuum', weights=w_v)
    h1 = axes[0].hist(kt, bins=bins, range=[0,r], histtype='step', color='pink', label='Medium', weights=w)
    h2 = axes[0].hist(kt[(lim1 < chi) & (chi < lim2)], bins=bins, range=[0,r], histtype='step', color='purple', label='0.25 < $\chi$ < 0.5', weights=w[(lim1 < chi) & (chi < lim2)])
    h3 = axes[0].hist(kt[(lim2 < chi) & (chi < lim3)], bins=bins, range=[0,r], histtype='step', color='red', label='0.5 < $\chi$ < 0.75', weights=w[(lim2 < chi) & (chi < lim3)])
    h4 = axes[0].hist(kt[(lim3 < chi) & (chi < lim4)], bins=bins, range=[0,r], histtype='step', color='green', label='0.75 < $\chi$ < 0.85', weights=w[(lim3 < chi) & (chi < lim4)])
    h5 = axes[0].hist(kt[(lim4 < chi) & (chi < lim5)], bins=bins, range=[0,r], histtype='step', color='orange', label='0.85 < $\chi$ < 0.95', weights=w[(lim4 < chi) & (chi < lim5)])
    h6 = axes[0].hist(kt[(lim5 < chi) & (chi < lim6)], bins=bins, range=[0,r], histtype='step', color='blue', label='0.95 < $\chi$ < 0.1', weights=w[(lim5 < chi) & (chi < lim6)])
    
    
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel('$\Theta_{SD}$')#k_{T,SD}$')
    axes[0].set_ylim(0, 10000)
    axes[0].set_title('Histogram for $\Theta_{SD}$ w/weights')
    
    ratio = h1[0]/h[0]
    for i in range(len(ratio)):
        if str(ratio[i])=="nan":
            ratio[i] = 0
    ratio2 = h2[0]/h[0]
    for i in range(len(ratio2)):
        if str(ratio2[i])=="nan":
            ratio2[i] = 0
    ratio3 = h3[0]/h[0]
    for i in range(len(ratio3)):
        if str(ratio3[i])=="nan":
            ratio3[i] = 0
    ratio4 = h4[0]/h[0]
    for i in range(len(ratio4)):
        if str(ratio4[i])=="nan":
            ratio4[i] = 0
    ratio5 = h5[0]/h[0]
    for i in range(len(ratio5)):
        if str(ratio5[i])=="nan":
            ratio5[i] = 0
    ratio6 = h6[0]/h[0]
    for i in range(len(ratio6)):
        if str(ratio6[i])=="nan":
            ratio6[i] = 0

    
            
    ax2.plot(h[1][:-1], ratio, color='pink')
    ax2.plot(h[1][:-1], ratio2, color='purple')
    ax2.plot(h[1][:-1], ratio3, color='red')
    ax2.plot(h[1][:-1], ratio4, color='green')
    ax2.plot(h[1][:-1], ratio5, color='orange')
    ax2.plot(h[1][:-1], ratio6, color='blue')
    ax2.axhline(y=1.0, color='r', linestyle='--')
    ax2.set_xlabel('$\Theta_{SD}$')
    ax2.set_ylabel('Ratio over vacuum')
    
    r = max(theta)
    divider2 = make_axes_locatable(axes[1])
    ax3 = divider2.append_axes("bottom", size="40%", pad=0)
    axes[1].figure.add_axes(ax3)
    h7 = axes[1].hist(theta_v, bins=bins, range=[0,r], histtype='step', color='black', label='Vacuum', weights=w_v)
    h8 = axes[1].hist(theta, bins=bins, range=[0,r], histtype='step', color='pink', label='Medium', weights=w)
    h9 = axes[1].hist(theta[(lim1 < chi) & (chi < lim2)], bins=bins, range=[0,r], histtype='step', color='purple', label='0.25 < $\chi$ < 0.5', weights=w[(lim1 < chi) & (chi < lim2)])
    h10 = axes[1].hist(theta[(lim2 < chi) & (chi < lim3)], bins=bins, range=[0,r], histtype='step', color='red', label='0.5 < $\chi$ < 0.75', weights=w[(lim2 < chi) & (chi < lim3)])
    h11 = axes[1].hist(theta[(lim3 < chi) & (chi < lim4)], bins=bins, range=[0,r], histtype='step', color='green', label='0.75 < $\chi$ < 0.85', weights=w[(lim3 < chi) & (chi < lim4)])
    h12 = axes[1].hist(theta[(lim4 < chi) & (chi < lim5)], bins=bins, range=[0,r], histtype='step', color='orange', label='0.85 < $\chi$ < 0.95', weights=w[(lim4 < chi) & (chi < lim5)])
    h13 = axes[1].hist(theta[(lim5 < chi) & (chi < lim6)], bins=bins, range=[0,r], histtype='step', color='blue', label='0.95 < $\chi$ < 0.1', weights=w[(lim5 < chi) & (chi < lim6)])
    
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Count')
    axes[1].set_xlabel('Number of constituents')
    axes[1].set_title('Histogram for Number of Constituents w/weights')
    
    ratio7 = h8[0]/h7[0]
    #for i in range(len(ratio7)):
    #    if str(ratio7[i])=="nan":
    #        ratio7[i] = 0
    ratio8 = h9[0]/h7[0]
    #for i in range(len(ratio8)):
    #    if str(ratio8[i])=="nan":
    #        ratio8[i] = 0
    ratio9 = h10[0]/h7[0]
    #for i in range(len(ratio9)):
    #    if str(ratio9[i])=="nan":
    #        ratio9[i] = 0
    ratio10 = h11[0]/h7[0]
    #for i in range(len(ratio10)):
    #    if str(ratio10[i])=="nan":
    #        ratio10[i] = 0
    ratio11 = h12[0]/h7[0]
    #for i in range(len(ratio11)):
    #    if str(ratio11[i])=="nan":
    #        ratio11[i] = 0
    ratio12 = h13[0]/h7[0]
    #for i in range(len(ratio12)):
    #    if str(ratio12[i])=="nan":
    #        ratio12[i] = 0

    
            
    ax3.plot(h7[1][:-1], ratio7, color='pink')
    ax3.plot(h7[1][:-1], ratio8, color='purple')
    ax3.plot(h7[1][:-1], ratio9, color='red')
    ax3.plot(h7[1][:-1], ratio10, color='green')
    ax3.plot(h7[1][:-1], ratio11, color='orange')
    ax3.plot(h7[1][:-1], ratio12, color='blue')
    ax3.axhline(y=1.0, color='r', linestyle='--')
    ax3.set_xlabel('Number of constituents')
    ax3.set_xlim(0,r)
    ax3.set_ylabel('Ratio over vacuum')
    
    r = max(z)
    """ 
    divider3 = make_axes_locatable(axes[2])
    ax4 = divider3.append_axes("bottom", size="40%", pad=0)
    axes[2].figure.add_axes(ax4)
    h14 = axes[2].hist(z_v, bins=bins, range=[0,r], histtype='step', color='black', label='Vacuum', weights=w_v)
    h15 = axes[2].hist(z, bins=bins, range=[0,r], histtype='step', color='pink', label='Medium', weights=w)
    h16 = axes[2].hist(z[(lim1 < chi) & (chi < lim2)], bins=bins, range=[0,r], histtype='step', color='purple', label='0.25 < $\chi$ < 0.5', weights=w[(lim1 < chi) & (chi < lim2)])
    h17 = axes[2].hist(z[(lim2 < chi) & (chi < lim3)], bins=bins, range=[0,r], histtype='step', color='red', label='0.5 < $\chi$ < 0.75', weights=w[(lim2 < chi) & (chi < lim3)])
    h18 = axes[2].hist(z[(lim3 < chi) & (chi < lim4)], bins=bins, range=[0,r], histtype='step', color='green', label='0.75 < $\chi$ < 0.85', weights=w[(lim3 < chi) & (chi < lim4)])
    h19 = axes[2].hist(z[(lim4 < chi) & (chi < lim5)], bins=bins, range=[0,r], histtype='step', color='orange', label='0.85 < $\chi$ < 0.95', weights=w[(lim4 < chi) & (chi < lim5)])
    h20 = axes[2].hist(z[(lim5 < chi) & (chi < lim6)], bins=bins, range=[0,r], histtype='step', color='blue', label='0.95 < $\chi$ < 0.1', weights=w[(lim5 < chi) & (chi < lim6)])
    
    axes[2].legend()
    axes[2].set_yscale('log')
    axes[2].set_ylabel('Count')
    axes[2].set_xlabel('$z_{SD}$')
    axes[2].set_title('Histogram for $z_{SD}$ w/weights')
    
    ratio13 = h15[0]/h14[0]
    for i in range(len(ratio13)):
        if str(ratio13[i])=="nan":
            ratio13[i] = 0
    ratio14 = h16[0]/h14[0]
    for i in range(len(ratio14)):
        if str(ratio14[i])=="nan":
            ratio14[i] = 0
    ratio15 = h17[0]/h14[0]
    for i in range(len(ratio15)):
        if str(ratio15[i])=="nan":
            ratio15[i] = 0
    ratio16 = h18[0]/h14[0]
    for i in range(len(ratio16)):
        if str(ratio16[i])=="nan":
            ratio16[i] = 0
    ratio17 = h19[0]/h14[0]
    for i in range(len(ratio17)):
        if str(ratio17[i])=="nan":
            ratio17[i] = 0
    ratio18 = h20[0]/h14[0]
    for i in range(len(ratio18)):
        if str(ratio18[i])=="nan":
            ratio18[i] = 0

    
            
    ax4.plot(h14[1][:-1], ratio13, color='pink')
    ax4.plot(h14[1][:-1], ratio14, color='purple')
    ax4.plot(h14[1][:-1], ratio15, color='red')
    ax4.plot(h14[1][:-1], ratio16, color='green')
    ax4.plot(h14[1][:-1], ratio17, color='orange')
    ax4.plot(h14[1][:-1], ratio18, color='blue')
    ax4.axhline(y=1.0, color='r', linestyle='--')
    ax4.set_xlabel('$z_{SD}$')
    ax4.set_ylabel('Ratio over vacuum')
    
    """
    
    fig.tight_layout()
    plt.show()

with gzip.open('real_hadron_sample_py.json.gz', 'r') as infile:
    data1 = infile.readlines()

with gzip.open('real_hadron_sample_py_vac.json.gz', 'r') as infile:
    data2 = infile.readlines()
    
chi_l = []
pt_l = []
pt_v = []
w = []
p_m = []
p_cons = []
p_theta = []
p_z = []
p_kt = []
c_help = 0
kt_m = []
p_kt3 = []

nr_split = []
nr_split_cut1 = []
nr_split_cut2 = []
nr_split_cut3 = []

sd_pt = []
sd_theta = []
sd_kt = []
sd_m = []
sd_z = []
sd_cons = []

for data in [data1]:
    for line in data:
        if line==b'\n':
            continue
        j = json.loads(line.decode('utf-8'))

        if j[0]['pt_v'] > 100:
            particles = []
            for p in j[1:]:
                particles.append(fj.PseudoJet(p['px'], p['py'], p['pz'], p['E']))
            clusterseq_v = fj.ClusterSequence(particles, jet_def)
            injet_v = clusterseq_v.inclusive_jets()
            
            if len(injet_v[0].constituents()) == 1:
                #print(injet_v[0].pt())
                c_help += 1
                continue
            

            
            subs = injet_v[0].exclusive_subjets(2)
            w.append(j[0]['w'])  
            chi_l.append(j[0]['chi'])
            p_cons.append(j[0]['nr'])
            pt_l.append(j[0]['pt'])
            p_theta.append(subs[0].delta_R(subs[1]))
            z2 =min(subs[0].pt(), subs[1].pt())/(subs[0].pt() + subs[1].pt())
            p_z.append(z2)
            p_kt.append(z2*(1-z2)*subs[0].delta_R(subs[1])*injet_v[0].pt())
            p_m.append(injet_v[0].m())
            p_kt3.append(np.sqrt(injet_v[0].kt2()))
            
            #Max kt
            all_kt = findmaxkt2(injet_v[0])
            kt_m.append(max(all_kt))
            nr_split.append(len(all_kt))
            nr_split_cut1.append(len(np.array(all_kt)[np.array(all_kt) > 0.1]))
            nr_split_cut2.append(len(np.array(all_kt)[np.array(all_kt) > 1]))
            nr_split_cut3.append(len(np.array(all_kt)[np.array(all_kt) > 2]))
            #Softdrop
            sdjet = SoftDrop(injet_v[0], 0.1, 1, 0.4)
            sd_cons.append(len(sdjet[0].constituents()))
            sdsub = sdjet[0].exclusive_subjets(2)
            sd_pt.append(sdjet[0].pt())
            sd_theta.append(sdsub[1].delta_R(sdsub[0]))
            sdz = min(sdsub[0].pt(), sdsub[1].pt())/(sdsub[0].pt() + sdsub[1].pt())
            sd_z.append(sdz)
            sd_kt.append(sdz*(1-sdz)*sdsub[0].delta_R(sdsub[1])*sdjet[0].pt())
            sd_m.append(sdjet[0].m())
            
            
print(p_kt[0])
print(kt_m[0])
print(p_kt3[0])


chi_l2 = []
pt_l2 = []
pt_v2 = []
w2 = []
p_m2 = []
p_cons2 = []
p_theta2 = []
p_z2 = []
p_kt2 = []
c_help2 = 0

sd_pt2 = []
sd_theta2 = []
sd_kt2 = []
sd_m2 = []
sd_z2 = []
sd_cons2 = []

for data in [data2]:
    for line in data:
        if line==b'\n':
            continue
        j = json.loads(line.decode('utf-8'))
        #chi_l.append(j[0]['chi'])
        if j[0]['pt_v'] > 100:
            particles = []
            for p in j[1:]:
                particles.append(fj.PseudoJet(p['px'], p['py'], p['pz'], p['E']))
            clusterseq_v = fj.ClusterSequence(particles, jet_def)
            injet_v = clusterseq_v.inclusive_jets()
            
            if len(injet_v[0].constituents()) == 1:
                #print(injet_v[0].pt())
                c_help2 += 1
                continue
            
            pt_l2.append(j[0]['pt_v'])
            chi_l2.append(j[0]['chi'])
            w2.append(j[0]['w'])
            p_cons2.append(j[0]['nr_v'])
            
            subs = injet_v[0].exclusive_subjets(2)
            z2 =min(subs[0].pt(), subs[1].pt())/(subs[0].pt() + subs[1].pt())
            p_z2.append(z2)
            p_kt2.append(z2*(1-z2)*subs[0].delta_R(subs[1])*injet_v[0].pt())
            p_m2.append(injet_v[0].m())
            p_theta2.append(subs[0].delta_R(subs[1]))
            
            sdjet = SoftDrop(injet_v[0], 0.1, 1, 0.4)
            sd_cons2.append(len(sdjet[0].constituents()))
            sdsub = sdjet[0].exclusive_subjets(2)
            sd_pt2.append(sdjet[0].pt())
            sd_theta2.append(sdsub[1].delta_R(sdsub[0]))
            sdz = min(sdsub[0].pt(), sdsub[1].pt())/(sdsub[0].pt() + sdsub[1].pt())
            sd_z2.append(sdz)
            sd_kt2.append(sdz*(1-sdz)*sdsub[0].delta_R(sdsub[1])*sdjet[0].pt())
            sd_m2.append(sdjet[0].m())
            
            
            

#print(len(pt_l))
#print(len(pt_l2))
#print(len(p_kt))
#print(len(p_kt2))
#print("Number of jets with only one constituent within cut of 500GeV: " + str(c_help))
#jointplot(np.array(chi_l), np.array(pt_l))
#fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
#axes[0].hist(p_kt)
#axes[0].set_yscale('log')
#axes[1].hist(kt_m)
#axes[1].set_yscale('log')
#fig.tight_layout()
#plt.show()
#plotspect(np.array(pt_l), np.array(w), np.array(p_m), np.array(kt_m), np.array(p_theta), np.array(p_z), np.array(p_kt))
#plotkt(p_theta2, p_theta, chi_l,pt_l)
#plotkorr(np.array(p_theta), np.array(p_nr1))
#plotkorr2(np.array(p_m), np.array(p_z), np.array(chi_l),np.array(p_cons), np.array(w))
plotish(np.array(chi_l), np.array(sd_theta), np.array(sd_theta2), np.array(p_cons), np.array(p_cons2), np.array(sd_z), np.array(sd_z2), np.array(w), np.array(w2))
#plotratio(np.array(pt_l), np.array(pt_v), np.array(pt_l2), np.array(pt_v2), np.array(w))

#funct(np.array(pt_l))
"""_summary_
    
plt.hist(nr_split, range=[0, 100], bins=50, histtype='step', label="No cut", density=True)
plt.hist(nr_split_cut1, color="green", range=[0,100], bins=50, histtype='step', label="$k_T > 0.1$", density=True)
plt.hist(nr_split_cut2, color="orange", range=[0,100], bins=50, histtype='step', label="$k_T > 1$", density=True)
plt.hist(nr_split_cut3, color="red", range=[0,100], bins=50, histtype='step', label="$k_T > 2$", density=True)
plt.legend()
plt.title("Number of Splittings/Nodes in graph")
plt.xlabel("Number of Splittings")
plt.ylabel("Count")
plt.show()
"""

#plt.hist(kt_m,bins=50, range=[0,5])
#plt.show()
plt.hist(kt_m, range=[0,3], bins=50)
plt.xlabel("$k_{T,max}$")
plt.ylabel("A.U.")
plt.show()