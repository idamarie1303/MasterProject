"""
Plot distribution of observables and kTmax plot
"""
import numpy as np
import fastjet as fj
import matplotlib.pyplot as plt
import gzip
import json
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable


jet_def = fj.JetDefinition(fj.cambridge_algorithm, 1000.)



#Find observables
def findmaxkt(jet):
    """
    Find a list of the kt for all nodes in the jet, 
    since the jets clustered with C/A dont necessarily have the hardest kt nodes at the top.
    Can find the max kt by taking the max of the list
    """
    listl = [jet]
    theta = []
    kt = []
    for i,l in enumerate(listl):
        
        if len(l.exclusive_subjets_up_to(2))==2:
            the = l.exclusive_subjets(2)[0].delta_R(l.exclusive_subjets(2)[1])
            theta.append(the)
            z2 = min(l.exclusive_subjets(2)[0].pt(), l.exclusive_subjets(2)[1].pt())/(l.exclusive_subjets(2)[0].pt() + l.exclusive_subjets(2)[1].pt())
            kt.append(z2*(1-z2)*the*l.pt())
            
            listl.append(l.exclusive_subjets(2)[0])
            listl.append(l.exclusive_subjets(2)[1])

    return kt

def SoftDrop(jet, z_cut, beta, R):
    """
    Apply SoftDrop grooming in python
    """
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

# Ploting
def plotkt(nr_split, nr_split_cut1, nr_split_cut2, nr_split_cut3):
    """
    Plot the distribution of number of splitting what applying a kt cut
    """
    plt.hist(nr_split, range=[0, 100], bins=50, histtype='step', label="No cut", density=True)
    plt.hist(nr_split_cut1, color="green", range=[0,100], bins=50, histtype='step', label="$k_T > 0.1$", density=True)
    plt.hist(nr_split_cut2, color="orange", range=[0,100], bins=50, histtype='step', label="$k_T > 1$", density=True)
    plt.hist(nr_split_cut3, color="red", range=[0,100], bins=50, histtype='step', label="$k_T > 2$", density=True)
    plt.legend()
    plt.title("Number of Splittings/Nodes in graph")
    plt.xlabel("Number of Splittings")
    plt.ylabel("Count")
    plt.show()

def plotish(chi, kt, kt_v, theta, theta_v, z, z_v, w, w_v):
    """
    Plot distribution of observables for jets with different chi values
    Can change what observables we want 
    """
    bins=20
    
    #Find limits to cut chi
    lim1 = 0.25
    lim2 = 0.5
    lim3 = 0.75
    lim4 = 0.85
    lim5 = 0.95
    lim6 = 1
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    
    ####First plot
    r = max(kt_v) #Range of x-axis
    
    #Divide plot into two
    divider = make_axes_locatable(axes[0])
    ax2 = divider.append_axes("bottom", size="40%", pad=0)
    axes[0].figure.add_axes(ax2)
    
    #Plot
    h = axes[0].hist(kt_v, bins=bins, range=[0,r], histtype='step', color='black', label='Vacuum', weights=w_v)
    h1 = axes[0].hist(kt, bins=bins, range=[0,r], histtype='step', color='pink', label='Medium', weights=w)
    h2 = axes[0].hist(kt[(lim1 < chi) & (chi < lim2)], bins=bins, range=[0,r], 
                      histtype='step', color='purple', label='0.25 < $\chi$ < 0.5', weights=w[(lim1 < chi) & (chi < lim2)])
    h3 = axes[0].hist(kt[(lim2 < chi) & (chi < lim3)], bins=bins, range=[0,r], 
                      histtype='step', color='red', label='0.5 < $\chi$ < 0.75', weights=w[(lim2 < chi) & (chi < lim3)])
    h4 = axes[0].hist(kt[(lim3 < chi) & (chi < lim4)], bins=bins, range=[0,r], 
                      histtype='step', color='green', label='0.75 < $\chi$ < 0.85', weights=w[(lim3 < chi) & (chi < lim4)])
    h5 = axes[0].hist(kt[(lim4 < chi) & (chi < lim5)], bins=bins, range=[0,r], 
                      histtype='step', color='orange', label='0.85 < $\chi$ < 0.95', weights=w[(lim4 < chi) & (chi < lim5)])
    h6 = axes[0].hist(kt[(lim5 < chi) & (chi < lim6)], bins=bins, range=[0,r], 
                      histtype='step', color='blue', label='0.95 < $\chi$ < 0.1', weights=w[(lim5 < chi) & (chi < lim6)])
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel('$k_{T,SD}$')
    axes[0].set_title('Histogram for $k_{T,SD}$ w/weights')
    
    #Calculate ratios
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

    #Plot ratio
    ax2.plot(h[1][:-1], ratio, color='pink')
    ax2.plot(h[1][:-1], ratio2, color='purple')
    ax2.plot(h[1][:-1], ratio3, color='red')
    ax2.plot(h[1][:-1], ratio4, color='green')
    ax2.plot(h[1][:-1], ratio5, color='orange')
    ax2.plot(h[1][:-1], ratio6, color='blue')
    ax2.axhline(y=1.0, color='r', linestyle='--')
    ax2.set_xlabel('$k_{T}}$')
    ax2.set_ylabel('Ratio over vacuum')
    
    
    ####Second plot
    r = max(theta) #Range of x-axis
    
    #Divide plot into two
    divider2 = make_axes_locatable(axes[1])
    ax3 = divider2.append_axes("bottom", size="40%", pad=0)
    axes[1].figure.add_axes(ax3)
    
    #Plot
    h7 = axes[1].hist(theta_v, bins=bins, range=[0,r], histtype='step', color='black', label='Vacuum', weights=w_v)
    h8 = axes[1].hist(theta, bins=bins, range=[0,r], histtype='step', color='pink', label='Medium', weights=w)
    h9 = axes[1].hist(theta[(lim1 < chi) & (chi < lim2)], bins=bins, range=[0,r], 
                      histtype='step', color='purple', label='0.25 < $\chi$ < 0.5', weights=w[(lim1 < chi) & (chi < lim2)])
    h10 = axes[1].hist(theta[(lim2 < chi) & (chi < lim3)], bins=bins, range=[0,r], 
                       histtype='step', color='red', label='0.5 < $\chi$ < 0.75', weights=w[(lim2 < chi) & (chi < lim3)])
    h11 = axes[1].hist(theta[(lim3 < chi) & (chi < lim4)], bins=bins, range=[0,r], 
                       histtype='step', color='green', label='0.75 < $\chi$ < 0.85', weights=w[(lim3 < chi) & (chi < lim4)])
    h12 = axes[1].hist(theta[(lim4 < chi) & (chi < lim5)], bins=bins, range=[0,r], 
                       histtype='step', color='orange', label='0.85 < $\chi$ < 0.95', weights=w[(lim4 < chi) & (chi < lim5)])
    h13 = axes[1].hist(theta[(lim5 < chi) & (chi < lim6)], bins=bins, range=[0,r], 
                       histtype='step', color='blue', label='0.95 < $\chi$ < 0.1', weights=w[(lim5 < chi) & (chi < lim6)])
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Count')
    axes[1].set_xlabel('Number of constituents')
    axes[1].set_title('Histogram for Number of Constituents w/weights')
    
    #Calculate ratio
    ratio7 = h8[0]/h7[0]
    for i in range(len(ratio7)):
        if str(ratio7[i])=="nan":
            ratio7[i] = 0
    ratio8 = h9[0]/h7[0]
    for i in range(len(ratio8)):
        if str(ratio8[i])=="nan":
            ratio8[i] = 0
    ratio9 = h10[0]/h7[0]
    for i in range(len(ratio9)):
        if str(ratio9[i])=="nan":
            ratio9[i] = 0
    ratio10 = h11[0]/h7[0]
    for i in range(len(ratio10)):
        if str(ratio10[i])=="nan":
            ratio10[i] = 0
    ratio11 = h12[0]/h7[0]
    for i in range(len(ratio11)):
        if str(ratio11[i])=="nan":
            ratio11[i] = 0
    ratio12 = h13[0]/h7[0]
    for i in range(len(ratio12)):
        if str(ratio12[i])=="nan":
            ratio12[i] = 0

    #Plot ratio
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
    
    
    
    """ 
    ####Third figure
    
    r = max(z) #Range of x-axis
    
    #Divide plot into two
    divider3 = make_axes_locatable(axes[2])
    ax4 = divider3.append_axes("bottom", size="40%", pad=0)
    axes[2].figure.add_axes(ax4)
    #Plot
    h14 = axes[2].hist(z_v, bins=bins, range=[0,r], histtype='step', color='black', label='Vacuum', weights=w_v)
    h15 = axes[2].hist(z, bins=bins, range=[0,r], histtype='step', color='pink', label='Medium', weights=w)
    h16 = axes[2].hist(z[(lim1 < chi) & (chi < lim2)], bins=bins, range=[0,r], 
    histtype='step', color='purple', label='0.25 < $\chi$ < 0.5', weights=w[(lim1 < chi) & (chi < lim2)])
    h17 = axes[2].hist(z[(lim2 < chi) & (chi < lim3)], bins=bins, range=[0,r], 
    histtype='step', color='red', label='0.5 < $\chi$ < 0.75', weights=w[(lim2 < chi) & (chi < lim3)])
    h18 = axes[2].hist(z[(lim3 < chi) & (chi < lim4)], bins=bins, range=[0,r], 
    histtype='step', color='green', label='0.75 < $\chi$ < 0.85', weights=w[(lim3 < chi) & (chi < lim4)])
    h19 = axes[2].hist(z[(lim4 < chi) & (chi < lim5)], bins=bins, range=[0,r], 
    histtype='step', color='orange', label='0.85 < $\chi$ < 0.95', weights=w[(lim4 < chi) & (chi < lim5)])
    h20 = axes[2].hist(z[(lim5 < chi) & (chi < lim6)], bins=bins, range=[0,r], 
    histtype='step', color='blue', label='0.95 < $\chi$ < 0.1', weights=w[(lim5 < chi) & (chi < lim6)])
    axes[2].legend()
    axes[2].set_yscale('log')
    axes[2].set_ylabel('Count')
    axes[2].set_xlabel('$z_{SD}$')
    axes[2].set_title('Histogram for $z_{SD}$ w/weights')
    
    #Calculate ratio
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

    #Plot Ratio 
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
            all_kt = findmaxkt(injet_v[0])
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
        if j[0]['pt_v'] > 100:
            particles = []
            for p in j[1:]:
                particles.append(fj.PseudoJet(p['px'], p['py'], p['pz'], p['E']))
            clusterseq_v = fj.ClusterSequence(particles, jet_def)
            injet_v = clusterseq_v.inclusive_jets()
            
            if len(injet_v[0].constituents()) == 1:
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
            
            #SoftDrop
            sdjet = SoftDrop(injet_v[0], 0.1, 1, 0.4)
            sd_cons2.append(len(sdjet[0].constituents()))
            sdsub = sdjet[0].exclusive_subjets(2)
            sd_pt2.append(sdjet[0].pt())
            sd_theta2.append(sdsub[1].delta_R(sdsub[0]))
            sdz = min(sdsub[0].pt(), sdsub[1].pt())/(sdsub[0].pt() + sdsub[1].pt())
            sd_z2.append(sdz)
            sd_kt2.append(sdz*(1-sdz)*sdsub[0].delta_R(sdsub[1])*sdjet[0].pt())
            sd_m2.append(sdjet[0].m())
            
            
#Plot the kT_max distribution
plt.hist(kt_m, range=[0,3], bins=50)
plt.xlabel("$k_{T,max}$")
plt.ylabel("A.U.")
plt.show()

#Plot the number of constituents for the different kt cuts
plotkt(nr_split, nr_split_cut1, nr_split_cut2, nr_split_cut3)

#Plot distributions for different chi cuts
#plotish(np.array(chi_l), np.array(sd_theta), np.array(sd_theta2), np.array(p_cons), np.array(p_cons2), np.array(sd_z), np.array(sd_z2), np.array(w), np.array(w2))



