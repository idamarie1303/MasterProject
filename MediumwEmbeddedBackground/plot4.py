import numpy as np
import matplotlib.pyplot as plt
import torch
import gzip
import json

with gzip.open('real_hadron_sample_py.json.gz', 'r') as infile:
    data1 = infile.readlines()

chi = []
pt = []
pt_v = []
w = []
const = []

for line in data1:
    if line==b'\n':
        continue
    j = json.loads(line.decode('utf-8'))
    chi.append(j[0]['chi'])
    pt.append(j[0]['pt'])
    pt_v.append(j[0]['pt_v'])
    w.append(j[0]['w'])
    const.append(j[0]['nr'])
    
    
with gzip.open('hadron_800bkg.json.gz', 'r') as infile:
    data2 = infile.readlines()

chi_50 = []
pt_50 = []
pt_v_50 = []
w_50 = []
const_50 = []

for line in data2:
    if line==b'\n':
        continue
    j = json.loads(line.decode('utf-8'))
    chi_50.append(j[0]['chi'])
    pt_50.append(j[0]['pt'])
    pt_v_50.append(j[0]['pt_v'])
    w_50.append(j[0]['w'])
    const_50.append(j[0]['nr'])
    

    
with gzip.open('hadron_800bkgsub.json.gz', 'r') as infile:
    data3 = infile.readlines()

chi_50s = []
pt_50s = []
pt_v_50s = []
w_50s = []
const_50s = []

for line in data3:
    if line==b'\n':
        continue
    j = json.loads(line.decode('utf-8'))
    chi_50s.append(j[0]['chi'])
    pt_50s.append(j[0]['pt'])
    pt_v_50s.append(j[0]['pt_v'])
    w_50s.append(j[0]['w'])
    const_50s.append(j[0]['nr'])

with gzip.open('hadron_2200bkg.json.gz', 'r') as infile:
    data4 = infile.readlines()

chi_100 = []
pt_100 = []
pt_v_100 = []
w_100 = []
const_100 = []

for line in data4:
    if line==b'\n':
        continue
    j = json.loads(line.decode('utf-8'))
    chi_100.append(j[0]['chi'])
    pt_100.append(j[0]['pt'])
    pt_v_100.append(j[0]['pt_v'])
    w_100.append(j[0]['w'])
    const_100.append(j[0]['nr'])
    
with gzip.open('hadron_2200bkgsub.json.gz', 'r') as infile:
    data5 = infile.readlines()

chi_100s = []
pt_100s = []
pt_v_100s = []
w_100s = []
const_100s = []

for line in data5:
    if line==b'\n':
        continue
    j = json.loads(line.decode('utf-8'))
    chi_100s.append(j[0]['chi'])
    pt_100s.append(j[0]['pt'])
    pt_v_100s.append(j[0]['pt_v'])
    w_100s.append(j[0]['w'])
    const_100s.append(j[0]['nr'])

chi = np.array(chi)
chi_50 = np.array(chi_50)
chi_50s = np.array(chi_50s)
chi_100 = np.array(chi_100)
chi_100s = np.array(chi_100s)

print(const[:10])
print(const_100[:10])

with gzip.open('hadron_12000bkg.json.gz', 'r') as infile:
    data5 = infile.readlines()

chi_12000 = []
pt_12000 = []
pt_v_12000 = []
w_12000 = []
const_12000 = []

for line in data5:
    if line==b'\n':
        continue
    j = json.loads(line.decode('utf-8'))
    chi_12000.append(j[0]['chi'])
    pt_12000.append(j[0]['pt'])
    pt_v_12000.append(j[0]['pt_v'])
    w_12000.append(j[0]['w'])
    const_12000.append(j[0]['nr'])
    
with gzip.open('hadron_12000bkgsub.json.gz', 'r') as infile:
    data5 = infile.readlines()

chi_12000s = []
pt_12000s = []
pt_v_12000s = []
w_12000s = []
const_12000s = []

for line in data5:
    if line==b'\n':
        continue
    j = json.loads(line.decode('utf-8'))
    chi_12000s.append(j[0]['chi'])
    pt_12000s.append(j[0]['pt'])
    pt_v_12000s.append(j[0]['pt_v'])
    w_12000s.append(j[0]['w'])
    const_12000s.append(j[0]['nr'])
    
with gzip.open('hadron_5000bkg.json.gz', 'r') as infile:
    data6 = infile.readlines()

chi_5000 = []
pt_5000 = []
pt_v_5000 = []
w_5000 = []
const_5000 = []

for line in data6:
    if line==b'\n':
        continue
    j = json.loads(line.decode('utf-8'))
    chi_5000.append(j[0]['chi'])
    pt_5000.append(j[0]['pt'])
    pt_v_5000.append(j[0]['pt_v'])
    w_5000.append(j[0]['w'])
    const_5000.append(j[0]['nr'])

with gzip.open('hadron_8000bkg.json.gz', 'r') as infile:
    data6 = infile.readlines()

chi_8000 = []
pt_8000 = []
pt_v_8000 = []
w_8000 = []
const_8000 = []

for line in data6:
    if line==b'\n':
        continue
    j = json.loads(line.decode('utf-8'))
    chi_8000.append(j[0]['chi'])
    pt_8000.append(j[0]['pt'])
    pt_v_8000.append(j[0]['pt_v'])
    w_8000.append(j[0]['w'])
    const_8000.append(j[0]['nr'])
    
with gzip.open('hadron_8000bkgsub.json.gz', 'r') as infile:
    data6 = infile.readlines()

chi_8000s = []
pt_8000s = []
pt_v_8000s = []
w_8000s = []
const_8000s = []

for line in data6:
    if line==b'\n':
        continue
    j = json.loads(line.decode('utf-8'))
    chi_8000s.append(j[0]['chi'])
    pt_8000s.append(j[0]['pt'])
    pt_v_8000s.append(j[0]['pt_v'])
    w_8000s.append(j[0]['w'])
    const_8000s.append(j[0]['nr'])

with gzip.open('hadron_10000bkg.json.gz', 'r') as infile:
    data7 = infile.readlines()
    
chi_10000 = []
pt_10000 = []
pt_v_10000 = []
w_10000 = []
const_10000 = []

for line in data7:
    if line==b'\n':
        continue
    j = json.loads(line.decode('utf-8'))
    chi_10000.append(j[0]['chi'])
    pt_10000.append(j[0]['pt'])
    pt_v_10000.append(j[0]['pt_v'])
    w_10000.append(j[0]['w'])
    const_10000.append(j[0]['nr'])

b=50
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

axes[0].hist(chi, bins=b, label="No bkg", color='black', histtype='step', range=[0.0,1.0])
axes[0].hist(chi_50, bins=b, label="50bkg", color="green", histtype='step', range=[0.0,1.0])
axes[0].hist(chi_100, bins=b, label="100bkg", color="orange", histtype='step', range=[0.0,1.0])
axes[0].set_title("$\chi_{jh}$ distribution")
axes[0].set_xlabel("$\chi_{jh}$")
axes[0].set_ylabel("Count")
axes[0].legend()

axes[1].hist(chi, bins=b, label="No bkg", color='black', histtype='step', range=[0.0,1.0])
axes[1].hist(chi_50s, bins=b, label="50bkg", color="green", histtype='step',range=[0.0,1.0])
axes[1].hist(chi_100s, bins=b, label="100bkg", color="orange", histtype='step',range=[0.0,1.0])
axes[1].set_title("$\chi_{jh}$ distribution after constituent subtraction")
axes[1].set_xlabel("$\chi_{jh}$")
axes[1].set_ylabel("Count")
axes[1].legend()

fig.tight_layout()
plt.show()




fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

axes1[0].hist(pt, bins=b, label="No bkg", color='black', histtype='step', range=[0.0,1600])
axes1[0].hist(pt_50, bins=b, label="50bkg", color="green", histtype='step', range=[0.0,1600])
axes1[0].hist(pt_100, bins=b, label="100bkg", color="orange", histtype='step', range=[0.0,1600])
axes1[0].hist(pt_8000, bins=b, label="1600bkg", color="red", histtype='step', range=[0.0,1600])
axes1[0].hist(pt_10000, bins=b, label="2000bkg", color="red", histtype='step', range=[0.0,1600])
axes1[0].hist(pt_12000, bins=b, label="2400bkg", color="red", histtype='step', range=[0.0,1600])
axes1[0].set_title("$p_t$ distribution")
axes1[0].set_xlabel("$p_T$(GeV)")
axes1[0].set_ylabel("Count")
axes1[0].legend()

axes1[1].hist(pt, bins=b, label="No bkg", color='black', histtype='step', range=[0.0,1600])
axes1[1].hist(pt_50s, bins=b, label="50bkg", color="green", histtype='step',range=[0.0,1600])
axes1[1].hist(pt_100s, bins=b, label="100bkg", color="orange", histtype='step',range=[0.0,1600])
axes1[1].hist(pt_12000s, bins=b, label="2400bkg", color="orange", histtype='step',range=[0.0,1600])
axes1[1].set_title("$p_T$ distribution after constituent subtraction")
axes1[1].set_xlabel("$p_T$(GeV)")
axes1[1].set_ylabel("Count")
axes1[1].legend()

fig1.tight_layout()
plt.show()

print(max(const))
print(max(const_50))
print(max(const_50s))
print(max(const_100))
print(max(const_100s))
print(max(const_12000))
fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))


axes2[0].hist(const, bins=b, label="No bkg", color='black', histtype='step', range=[0.0,350])
axes2[0].hist(const_50, bins=b, label="10bkg per unit eta", color="green", histtype='step', range=[0.0,350])
axes2[0].hist(const_100, bins=b, label="440bkg per unit eta", color="orange", histtype='step', range=[0.0,350])
axes2[0].hist(const_5000, bins=b, label="1000bkg per unit eta", color="grey", histtype='step', range=[0.0,350])
axes2[0].hist(const_8000, bins=b, label="1600 bkg per unit eta", color="pink", histtype='step', range=[0.0,350])
axes2[0].hist(const_10000, bins=b, label="2000 bkg per unit eta", color="yellow", histtype='step', range=[0.0,350])
axes2[0].hist(const_12000, bins=b, label="2400 bkg per unit eta", color="red", histtype='step', range=[0.0,350])
axes2[0].set_title("Number of constituents")
axes2[0].set_xlabel("Number of constituents")
axes2[0].set_ylabel("Count")
axes2[0].legend()

axes2[1].hist(const, bins=b, label="No bkg", color='black', histtype='step', range=[0.0,200])
axes2[1].hist(const_50s, bins=b, label="10bkg per unit eta", color="green", histtype='step',range=[0.0,200])
axes2[1].hist(const_100s, bins=b, label="440bkg per unit eta", color="orange", histtype='step',range=[0.0,200])
axes2[1].hist(const_8000s, bins=b, label="1600bkg per unit eta", color="pink", histtype='step',range=[0.0,200])
axes2[1].hist(const_12000s, bins=b, label="2400bkg per unit eta", color="red", histtype='step',range=[0.0,200])
axes2[1].set_title("Number of constituents after constituent subtraction")
axes2[1].set_xlabel("Number of constituents")
axes2[1].set_ylabel("Count")
axes2[1].legend()

fig2.tight_layout()
plt.show()