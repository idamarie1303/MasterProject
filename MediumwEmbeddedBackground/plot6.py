import numpy as np
import matplotlib.pyplot as plt
import torch
import gzip
import json
import fastjet as fj


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


  
jet_def = fj.JetDefinition(fj.cambridge_algorithm, 1000.)



with gzip.open('real_hadron_sample_py.json.gz', 'r') as infile:
    data1 = infile.readlines()


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

nr_split = []



for line in data1:
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
        
        #for i in range(len(injet_v[0].constituents())):
        #    if injet_v[0].constituents()[i].pt()< 0.0001:
        #        print(injet_v[0].constituents()[i].pt())
        
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
        
        #Max kt
        all_kt = findmaxkt2(injet_v[0])
        kt_m.append(max(all_kt))
        nr_split.append(len(all_kt))
        
with gzip.open('hadron_100bkg.json.gz', 'r') as infile:
    data1 = infile.readlines()
    
chi_l100 = []
pt_l100 = []
pt_v100 = []
w100 = []
p_m100 = []
p_cons100 = []
p_theta100 = []
p_z100 = []
p_kt100 = []
c_help100 = 0
kt_m100 = []

nr_split100 = []



for line in data1:
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
            c_help100 += 1
            continue
        
        #for i in range(len(injet_v[0].constituents())):
        #    if injet_v[0].constituents()[i].pt()< 0.0001:
        #        print(injet_v[0].constituents()[i].pt())
        
        subs = injet_v[0].exclusive_subjets(2)
        w100.append(j[0]['w'])  
        chi_l100.append(j[0]['chi'])
        p_cons100.append(j[0]['nr'])
        pt_l100.append(j[0]['pt'])
        p_theta100.append(subs[0].delta_R(subs[1]))
        z2 =min(subs[0].pt(), subs[1].pt())/(subs[0].pt() + subs[1].pt())
        p_z100.append(z2)
        p_kt100.append(z2*(1-z2)*subs[0].delta_R(subs[1])*injet_v[0].pt())
        p_m100.append(injet_v[0].m())
        
        #Max kt
        all_kt = findmaxkt2(injet_v[0])
        kt_m100.append(max(all_kt))
        nr_split100.append(len(all_kt))

with gzip.open('hadron_400bkg.json.gz', 'r') as infile:
    data1 = infile.readlines()
    
chi_l400 = []
pt_l400 = []
pt_v400 = []
w400 = []
p_m400 = []
p_cons400 = []
p_theta400 = []
p_z400 = []
p_kt400 = []
c_help400 = 0
kt_m400 = []

nr_split400 = []



for line in data1:
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
            c_help400 += 1
            continue
        
        #for i in range(len(injet_v[0].constituents())):
        #    if injet_v[0].constituents()[i].pt()< 0.0001:
        #        print(injet_v[0].constituents()[i].pt())
        
        subs = injet_v[0].exclusive_subjets(2)
        w400.append(j[0]['w'])  
        chi_l400.append(j[0]['chi'])
        p_cons400.append(j[0]['nr'])
        pt_l400.append(j[0]['pt'])
        p_theta400.append(subs[0].delta_R(subs[1]))
        z2 =min(subs[0].pt(), subs[1].pt())/(subs[0].pt() + subs[1].pt())
        p_z400.append(z2)
        p_kt400.append(z2*(1-z2)*subs[0].delta_R(subs[1])*injet_v[0].pt())
        p_m400.append(injet_v[0].m())
        
        #Max kt
        all_kt = findmaxkt2(injet_v[0])
        kt_m400.append(max(all_kt))
        nr_split400.append(len(all_kt))

with gzip.open('hadron_1200bkg.json.gz', 'r') as infile:
    data1 = infile.readlines()
    
chi_l1200 = []
pt_l1200 = []
pt_v1200 = []
w1200 = []
p_m1200 = []
p_cons1200 = []
p_theta1200 = []
p_z1200 = []
p_kt1200 = []
c_help1200 = 0
kt_m1200 = []

nr_split1200 = []



for line in data1:
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
            c_help1200 += 1
            continue
        
        #for i in range(len(injet_v[0].constituents())):
        #    if injet_v[0].constituents()[i].pt()< 0.0001:
        #        print(injet_v[0].constituents()[i].pt())
        
        subs = injet_v[0].exclusive_subjets(2)
        w1200.append(j[0]['w'])  
        chi_l1200.append(j[0]['chi'])
        p_cons1200.append(j[0]['nr'])
        pt_l1200.append(j[0]['pt'])
        p_theta1200.append(subs[0].delta_R(subs[1]))
        z2 =min(subs[0].pt(), subs[1].pt())/(subs[0].pt() + subs[1].pt())
        p_z1200.append(z2)
        p_kt1200.append(z2*(1-z2)*subs[0].delta_R(subs[1])*injet_v[0].pt())
        p_m1200.append(injet_v[0].m())
        
        #Max kt
        all_kt = findmaxkt2(injet_v[0])
        kt_m1200.append(max(all_kt))
        nr_split1200.append(len(all_kt))

with gzip.open('hadron_2200bkg.json.gz', 'r') as infile:
    data1 = infile.readlines()
    
chi_l2200 = []
pt_l2200 = []
pt_v2200 = []
w2200 = []
p_m2200 = []
p_cons2200 = []
p_theta2200 = []
p_z2200 = []
p_kt2200 = []
c_help2200 = 0
kt_m2200 = []

nr_split2200 = []



for line in data1:
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
            c_help2200 += 1
            continue
        
        #for i in range(len(injet_v[0].constituents())):
        #    if injet_v[0].constituents()[i].pt()< 0.0001:
        #        print(injet_v[0].constituents()[i].pt())
        
        subs = injet_v[0].exclusive_subjets(2)
        w2200.append(j[0]['w'])  
        chi_l2200.append(j[0]['chi'])
        p_cons2200.append(j[0]['nr'])
        pt_l2200.append(j[0]['pt'])
        p_theta2200.append(subs[0].delta_R(subs[1]))
        z2 =min(subs[0].pt(), subs[1].pt())/(subs[0].pt() + subs[1].pt())
        p_z2200.append(z2)
        p_kt2200.append(z2*(1-z2)*subs[0].delta_R(subs[1])*injet_v[0].pt())
        p_m2200.append(injet_v[0].m())
        
        #Max kt
        all_kt = findmaxkt2(injet_v[0])
        kt_m2200.append(max(all_kt))
        nr_split2200.append(len(all_kt))
        

with gzip.open('hadron_5000bkg.json.gz', 'r') as infile:
    data1 = infile.readlines()
    
chi_l5000 = []
pt_l5000 = []
pt_v5000 = []
w5000 = []
p_m5000 = []
p_cons5000 = []
p_theta5000 = []
p_z5000 = []
p_kt5000 = []
c_help5000 = 0
kt_m5000 = []

nr_split5000 = []



for line in data1:
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
            c_help5000 += 1
            continue
        
        #for i in range(len(injet_v[0].constituents())):
        #    if injet_v[0].constituents()[i].pt()< 0.0001:
        #        print(injet_v[0].constituents()[i].pt())
        
        subs = injet_v[0].exclusive_subjets(2)
        w5000.append(j[0]['w'])  
        chi_l5000.append(j[0]['chi'])
        p_cons5000.append(j[0]['nr'])
        pt_l5000.append(j[0]['pt'])
        p_theta5000.append(subs[0].delta_R(subs[1]))
        z2 =min(subs[0].pt(), subs[1].pt())/(subs[0].pt() + subs[1].pt())
        p_z5000.append(z2)
        p_kt5000.append(z2*(1-z2)*subs[0].delta_R(subs[1])*injet_v[0].pt())
        p_m5000.append(injet_v[0].m())
        
        #Max kt
        all_kt = findmaxkt2(injet_v[0])
        kt_m5000.append(max(all_kt))
        nr_split5000.append(len(all_kt))
        
        
with gzip.open('hadron_8000bkg.json.gz', 'r') as infile:
    data1 = infile.readlines()
    
chi_l8000 = []
pt_l8000 = []
pt_v8000 = []
w8000 = []
p_m8000 = []
p_cons8000 = []
p_theta8000 = []
p_z8000 = []
p_kt8000 = []
c_help8000 = 0
kt_m8000 = []

nr_split8000 = []



for line in data1:
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
            c_help8000 += 1
            continue
        
        #for i in range(len(injet_v[0].constituents())):
        #    if injet_v[0].constituents()[i].pt()< 0.0001:
        #        print(injet_v[0].constituents()[i].pt())
        
        subs = injet_v[0].exclusive_subjets(2)
        w8000.append(j[0]['w'])  
        chi_l8000.append(j[0]['chi'])
        p_cons8000.append(j[0]['nr'])
        pt_l8000.append(j[0]['pt'])
        p_theta8000.append(subs[0].delta_R(subs[1]))
        z2 =min(subs[0].pt(), subs[1].pt())/(subs[0].pt() + subs[1].pt())
        p_z8000.append(z2)
        p_kt8000.append(z2*(1-z2)*subs[0].delta_R(subs[1])*injet_v[0].pt())
        p_m8000.append(injet_v[0].m())
        
        #Max kt
        all_kt = findmaxkt2(injet_v[0])
        kt_m8000.append(max(all_kt))
        nr_split8000.append(len(all_kt))
        
with gzip.open('hadron_10000bkg.json.gz', 'r') as infile:
    data1 = infile.readlines()
    
chi_l10000 = []
pt_l10000 = []
pt_v10000 = []
w10000 = []
p_m10000 = []
p_cons10000 = []
p_theta10000 = []
p_z10000 = []
p_kt10000 = []
c_help10000 = 0
kt_m10000 = []

nr_split10000 = []



for line in data1:
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
            c_help10000 += 1
            continue
        
        #for i in range(len(injet_v[0].constituents())):
        #    if injet_v[0].constituents()[i].pt()< 0.0001:
        #        print(injet_v[0].constituents()[i].pt())
        
        subs = injet_v[0].exclusive_subjets(2)
        w10000.append(j[0]['w'])  
        chi_l10000.append(j[0]['chi'])
        p_cons10000.append(j[0]['nr'])
        pt_l10000.append(j[0]['pt'])
        p_theta10000.append(subs[0].delta_R(subs[1]))
        z2 =min(subs[0].pt(), subs[1].pt())/(subs[0].pt() + subs[1].pt())
        p_z10000.append(z2)
        p_kt10000.append(z2*(1-z2)*subs[0].delta_R(subs[1])*injet_v[0].pt())
        p_m10000.append(injet_v[0].m())
        
        #Max kt
        all_kt = findmaxkt2(injet_v[0])
        kt_m10000.append(max(all_kt))
        nr_split10000.append(len(all_kt))
           
        
with gzip.open('hadron_12000bkg.json.gz', 'r') as infile:
    data1 = infile.readlines()
    
chi_l12000 = []
pt_l12000 = []
pt_v12000 = []
w12000 = []
p_m12000 = []
p_cons12000 = []
p_theta12000 = []
p_z12000 = []
p_kt12000 = []
c_help12000 = 0
kt_m12000 = []

nr_split12000 = []



for line in data1:
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
            c_help12000 += 1
            continue
        
        #for i in range(len(injet_v[0].constituents())):
        #    if injet_v[0].constituents()[i].pt()< 0.0001:
        #        print(injet_v[0].constituents()[i].pt())
        
        subs = injet_v[0].exclusive_subjets(2)
        w12000.append(j[0]['w'])  
        chi_l12000.append(j[0]['chi'])
        p_cons12000.append(j[0]['nr'])
        pt_l12000.append(j[0]['pt'])
        p_theta12000.append(subs[0].delta_R(subs[1]))
        z2 =min(subs[0].pt(), subs[1].pt())/(subs[0].pt() + subs[1].pt())
        p_z12000.append(z2)
        p_kt12000.append(z2*(1-z2)*subs[0].delta_R(subs[1])*injet_v[0].pt())
        p_m12000.append(injet_v[0].m())
        
        #Max kt
        all_kt = findmaxkt2(injet_v[0])
        kt_m12000.append(max(all_kt))
        nr_split12000.append(len(all_kt))
        

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
     
axes[0].hist(nr_split, bins=50, histtype='step', label="No bkg", density=True, color='black', range=[0,300])
axes[0].hist(nr_split100, bins=50, histtype='step', label="20 bkg", density=True, color='magenta', range=[0,300])
axes[0].hist(nr_split400, bins=50, histtype='step', label="80 bkg", density=True, color='gray', range=[0,300])
axes[0].hist(nr_split1200, bins=50, histtype='step', label="240 bkg", density=True, color='blue', range=[0,300])
axes[0].hist(nr_split2200, bins=50, histtype='step', label="440 bkg", density=True, color='yellow', range=[0,300])
axes[0].hist(nr_split5000, bins=50, histtype='step', label="1000 bkg", density=True, color='red', range=[0,300])
axes[0].hist(nr_split8000, bins=50, histtype='step', label="1600 bkg", density=True, color='pink', range=[0,300])
axes[0].hist(nr_split10000, bins=50, histtype='step', label="2000 bkg", density=True, color='orange', range=[0,300])
axes[0].hist(nr_split12000, bins=50, histtype='step', label="2400 bkg", density=True, color='green', range=[0,300])
axes[0].set_ylabel("Count")
axes[0].set_xlabel("Number of constituents")
axes[0].legend()


axes[1].hist(kt_m, bins=50, histtype='step', label="No bkg", density=True, color='black', range=[0,5])
axes[1].hist(kt_m100, bins=50, histtype='step', label="20 bkg", density=True, color='magenta', range=[0,5])
axes[1].hist(kt_m400, bins=50, histtype='step', label="80 bkg", density=True, color='gray', range=[0,5])
axes[1].hist(kt_m1200, bins=50, histtype='step', label="240 bkg", density=True, color='blue', range=[0,5])
axes[1].hist(kt_m2200, bins=50, histtype='step', label="440 bkg", density=True, color='yellow', range=[0,5])
axes[1].hist(kt_m5000, bins=50, histtype='step', label="1000 bkg", density=True, color='red', range=[0,5])
axes[1].hist(kt_m8000, bins=50, histtype='step', label="1600 bkg", density=True, color='pink', range=[0,5])
axes[1].hist(kt_m10000, bins=50, histtype='step', label="2000 bkg", density=True, color='orange', range=[0,5])
axes[1].hist(kt_m12000, bins=50, histtype='step', label="2400 bkg", density=True, color='green', range=[0,5])
axes[1].set_xlabel("$k_{T,max}$")
axes[1].set_ylabel("Count")
axes[1].legend()

fig.tight_layout()
plt.show()


fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
     
axes1[0].hist(p_theta, bins=50, histtype='step', label="No bkg", density=True, color='black', range=[0,0.5])
axes1[0].hist(p_theta100, bins=50, histtype='step', label="20 bkg", density=True, color='magenta', range=[0,0.5])
axes1[0].hist(p_theta400, bins=50, histtype='step', label="80 bkg", density=True, color='gray', range=[0,0.5])
axes1[0].hist(p_theta1200, bins=50, histtype='step', label="240 bkg", density=True, color='blue', range=[0,0.5])
axes1[0].hist(p_theta2200, bins=50, histtype='step', label="440 bkg", density=True, color='yellow', range=[0,0.5])
axes1[0].hist(p_theta5000, bins=50, histtype='step', label="1000 bkg", density=True, color='red', range=[0,0.5])
axes1[0].hist(p_theta8000, bins=50, histtype='step', label="1600 bkg", density=True, color='pink', range=[0,0.5])
axes1[0].hist(p_theta10000, bins=50, histtype='step', label="2000 bkg", density=True, color='orange', range=[0,0.5])
axes1[0].hist(p_theta12000, bins=50, histtype='step', label="2400 bkg", density=True, color='green', range=[0,0.5])
axes1[0].set_ylabel("Count")
axes1[0].set_xlabel("$\Theta$")
axes1[0].legend()


axes1[1].hist(p_m, bins=50, histtype='step', label="No bkg", density=True, color='black', range=[0,200])
axes1[1].hist(p_m100, bins=50, histtype='step', label="20 bkg", density=True, color='magenta', range=[0,200])
axes1[1].hist(p_m400, bins=50, histtype='step', label="80 bkg", density=True, color='gray', range=[0,200])
axes1[1].hist(p_m1200, bins=50, histtype='step', label="240 bkg", density=True, color='blue', range=[0,200])
axes1[1].hist(p_m2200, bins=50, histtype='step', label="440 bkg", density=True, color='yellow', range=[0,200])
axes1[1].hist(p_m5000, bins=50, histtype='step', label="1000 bkg", density=True, color='red', range=[0,200])
axes1[1].hist(p_m8000, bins=50, histtype='step', label="1600 bkg", density=True, color='pink', range=[0,200])
axes1[1].hist(p_m10000, bins=50, histtype='step', label="2000 bkg", density=True, color='orange', range=[0,200])
axes1[1].hist(p_m12000, bins=50, histtype='step', label="2400 bkg", density=True, color='green', range=[0,200])
axes1[1].set_xlabel("$m$")
axes1[1].set_ylabel("Count")
axes1[1].legend()

fig1.tight_layout()
plt.show()