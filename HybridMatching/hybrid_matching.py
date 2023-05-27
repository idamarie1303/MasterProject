"""
This file creates the dataset for hybrid jets. 
The code match medium-modified and vacuum jets, 
and saves the corresponding jets to a json file, 
ready to be sent into the LundNet model
"""

import fastjet as fj
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time


jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)

start = time.time()
print("start")

#Define lists
list_part = []
mini_list = []

chi_l = []
pt_l = []
con_l = []
pt_v_l = []
p_medium_l = []
p_vacuum_l = []

p_medium = []
p_vacuum = []
weights = []

dictionary = []

lastev = "12345678"

with open('/Users/idamarie/Physics/HYBRID_realistic.out', "r") as f:
    lines = f.readlines()
    #print(len(lines))
    for line in lines:
        
        if line[0].isalpha() or line[3].isalpha():
            #end of event
            if line[0]=="#":
                lastev=line
            if line[0]=="w":
                weight_line = line.strip("\n").split(" ")
                weight = eval(weight_line[1])
                
            if line[0]=='e':
                #clustering
                    
                clusterseq_m = fj.ClusterSequence(p_medium, jet_def)
                injet_m = clusterseq_m.inclusive_jets()

                clusterseq_v = fj.ClusterSequence(p_vacuum, jet_def)
                injet_v = clusterseq_v.inclusive_jets()
                
                sortedJetsM = sorted(injet_m, key=lambda x: x.pt() ,reverse=True)
                sortedJetsV = sorted(injet_v, key=lambda x: x.pt() , reverse=True)
                
                if sortedJetsV[0].pt() < 100:
                    p_medium.clear()
                    p_vacuum.clear()
                    continue
                
    

                med_index=0
                if sortedJetsM[0].pt() > sortedJetsV[0].pt():
                    med_index = 1
                    if sortedJetsM[1].pt() > sortedJetsV[0].pt():
                        med_index=2
                        if sortedJetsM[2].pt() > sortedJetsV[0].pt():
                            print("oh noo")

                dRmin = 0.4
                mindR = 5
                nr = 0
                same = 0
                while (mindR>dRmin):
                    if (nr == len(sortedJetsV)-1):
                        #No more particles
                        break
                    dR = sortedJetsM[med_index].delta_R(sortedJetsV[nr])
                    if (dR < mindR and sortedJetsM[med_index].pt()< sortedJetsV[nr].pt() ):
                        #More similar
                        mindR = dR
                        same = nr
                    else:
                        nr +=1
                
                if sortedJetsM[med_index].pt() > sortedJetsV[same].pt():
                    print("Too large")
                    
                
                
                
                #Create dataset
                dictionary_m = [{"chi": sortedJetsM[med_index].pt()/sortedJetsV[same].pt(), "pt": sortedJetsM[med_index].pt(), "pt_v": sortedJetsV[same].pt(), "nr": len(sortedJetsM[med_index].constituents()), "nr_v": len(sortedJetsV[same].constituents()), "w": weight}]
                for cons in sortedJetsM[med_index].constituents(): #Change to sortedJetsV[same].constituents() to get vacuum dataset
                    dictionary_m.append({"E": cons.E(),"px": cons.px(), "py": cons.py(), "pz": cons.pz()})    
                
                
                #For plotting stuff
                if sortedJetsV[same].pt()>100: #Cannot trust jets with less than 100GeV
                    dictionary.append(dictionary_m)
                    
                    chi_l.append(sortedJetsM[med_index].pt()/sortedJetsV[same].pt())
                    pt_l.append(sortedJetsM[med_index].pt())
                    pt_v_l.append(sortedJetsV[same].pt())
                    p_medium_l.append(len(sortedJetsM[med_index].constituents()))
                    p_vacuum_l.append(len(sortedJetsV[same].constituents()))
                    weights.append(weight)
                
                #Finish this event and move on
                p_medium.clear()
                p_vacuum.clear()
                if int(lastev[7:])==499:
                    #Print every 500 jets to keep track of how far along the algorithm is
                    print(len(dictionary))

        else:
            particle = line.strip("\n").split(" ")
            coords = [eval(i) for i in particle[:4]]
            E = np.sqrt(pow(coords[0], 2)+ pow(coords[1],2) + pow(coords[2],2) + pow(coords[3],2))
            pjet = fj.PseudoJet(coords[0], coords[1], coords[2], E)
            if pjet.pt() == 0:
                continue
            #Medium particles for hadrons: 1 and 0
            #Medium particles for partons: 10
            if int(particle[5])==0 or int(particle[5])==1:
                p_medium.append(pjet)
            #Vacuum particles for hadrons: 5
            #Vacuum particles for partons: 50
            elif int(particle[5]) == 5:
                p_vacuum.append(pjet)


print("Number of samples: " + str(len(chi_l)))


with open('real_hadron_sample_py.json', "w") as f:
    for line in dictionary:
        f.write(json.dumps(line).replace("]", "]\n"))

#Print how long it takes
end = time.time()
print("Time it takes: " + str(end - start))

#Plot div. observables
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
sns.histplot(x=np.array(pt_l)/1000,       ax=axes[0][0], bins=50, weights=np.array(weights))
sns.histplot(x=chi_l,      ax=axes[0][1], bins=50, weights=np.array(weights), stat="probability")
sns.histplot(x=p_medium_l, ax=axes[0][2], bins=25, weights=np.array(weights))
sns.histplot(x=np.array(pt_v_l)/1000,       ax=axes[1][0], bins=50, weights=np.array(weights))
#sns.histplot(x=chi_l,      ax=axes[1][1], bins=50, weights=np.array(w_list), stat="probability")
sns.histplot(x=p_vacuum_l, ax=axes[1][2], bins=25, weights=np.array(weights))

axes[0][0].set_title("Histogram for jet $p_T$ in medium")
axes[0][0].set_xlabel("Jet $p_T$ (TeV)")
axes[0][1].set_title("Histogram for true $\chi_{jh}$")
axes[0][1].set_xlabel("True $\chi_{jh}$")
axes[0][1].set_ylabel("Normalized to Unity")
axes[0][2].set_title("Histogram for Number of Constituents in medium jet")
axes[0][2].set_xlabel("Nr. of jet constituents")
axes[0][2].set_ylabel("Normalized to Unity")
axes[1][0].set_title("Histogram for jet $p_T$ in vacuum")
axes[1][0].set_xlabel("Jet $p_T$ (TeV)")


axes[1][1].set_title("Histogram for reweighted true $\chi_{jh}$")
axes[1][1].set_ylabel("Reweighted and Normalized to Unity")
axes[1][1].set_xlabel("True $\chi_{jh}$")
axes[1][2].set_title("Histogram for Number of Constituents in vacuum jet")
axes[1][2].set_xlabel("Nr. of jet constituents")
axes[1][2].set_ylabel("Normalized to Unity")

fig.tight_layout()
plt.show()
