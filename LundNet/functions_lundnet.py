#Functions I made for lundnet modifications
import tqdm
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.special import kv, iv

#Weight stuff
def Neff(N, beta=0.9998):
        return (1-pow(beta, N))/(1-beta)

def findEn(dataloaders, w2d=False):
        chis=[]
        pts = []
        for dataloader in dataloaders:
            with tqdm.tqdm(dataloader, ascii=True) as tq:
                for batch in tq:
                    label = batch.label
                    pt = batch.pt
                    chis.append(label)
                    pts.append(pt)
        #print(len(torch.concat(pts).numpy()))
        #print(pts[0])
        if w2d==True:
            h = np.histogram2d(torch.concat(pts).numpy()/1000, torch.concat(chis).numpy(), range=([0,1.6],[0,1.]), bins=25)
            return Neff(h[0]), h[1], h[2]
        else:
            h = np.histogram(torch.concat(chis).numpy(), range=[0,1.], bins=25)
            return Neff(h[0]), h[1], 0

def findWeights2d(label, pt, En, lim1, lim2):
    w_list = []
    for val, p in zip(label,pt):
        i = np.argmax(np.array(p.numpy()/1000) < lim1)
        j = np.argmax(np.array(val.numpy() < lim2))
        if En[i-1][j-1] == 0:
            w_list.append(0)
            continue
        w_list.append(1/En[i-1][j-1])
    return torch.tensor(w_list)

def findWeights(label, pt, En, lim, lim2):
    w_list = []
    for val in label:
        j = np.argmax(np.array(val.numpy() < lim))
        if En[j-1] == 0:
            w_list.append(0)
            continue
        w_list.append(1/En[j-1])
    return torch.tensor(w_list)

#Regression stuff
def regression_roc_auc_score(y_true, y_pred, num_rounds = 10000):
    """
    Computes Regression-ROC-AUC-score.
    
    Parameters:
    ----------
    y_true: array-like of shape (n_samples,). Binary or continuous target variable.
    y_pred: array-like of shape (n_samples,). Target scores.
    num_rounds: int or string. If integer, number of random pairs of observations. 
                If string, 'exact', all possible pairs of observations will be evaluated.
    
    Returns:
    -------
    rroc: float. Regression-ROC-AUC-score.
    """
    
    import numpy as np
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_pairs = 0
    num_same_sign = 0
    
    for i, j in _yield_pairs(y_true, num_rounds):
        diff_true = y_true[i] - y_true[j]
        diff_score = y_pred[i] - y_pred[j]
        if diff_true * diff_score > 0:
            num_same_sign += 1
        elif diff_score == 0:
            num_same_sign += .5
        num_pairs += 1
        
    return num_same_sign / num_pairs

def _yield_pairs(y_true, num_rounds):
    """
    Returns pairs of valid indices. Indices must belong to observations having different values.
    
    Parameters:
    ----------
    y_true: array-like of shape (n_samples,). Binary or continuous target variable.
    num_rounds: int or string. If integer, number of random pairs of observations to return. 
                If string, 'exact', all possible pairs of observations will be returned.
    
    Yields:
    -------
    i, j: tuple of int of shape (2,). Indices referred to a pair of samples.
    
    """
    import numpy as np
    
    if num_rounds == 'exact':
        for i in range(len(y_true)):
            for j in np.where((y_true != y_true[i]) & (np.arange(len(y_true)) > i))[0]:
                yield i, j     
    else:
        for r in range(num_rounds):
            i = np.random.choice(range(len(y_true)))
            j = np.random.choice(np.where(y_true != y_true[i])[0])
            yield i, j

#Plotting stuff
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

def plotloss(train, val, num_epoch):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    axes.plot(np.arange(1,num_epoch+1), np.array(train), color='blue', label='Train')
    #axes.set_title("Loss")
    axes.set_xlabel("Number of Epochs")
    axes.set_ylabel("Loss")
    axes.plot(np.arange(1,num_epoch+1), np.array(val), color='orange', label='Validation')

    plt.legend()

    fig.tight_layout()
    plt.show()


def accmeasure(true, pred):
    h = np.histogram2d(true, pred, range=([0.1,1], [0.1,1]), bins=15)
    accline = []
    for i, line in enumerate(h[0]):
        correct = line[i]
        summie = line.sum()
        accline.append(correct/summie)
    accuracy = np.array(accline).sum()/len(accline)
    return accuracy
    
    
def predweights(true, pred):
    h = np.histogram2d(true, pred, range=([0.1,1], [0.1,1]), bins=15)
    
    ws = []
    for lm in h[0]:
        sm = lm.sum()
        sm_bin = []
        for mk in lm:
            if sm==0:
                sm_bin.append(0)
            else: 
                sm_bin.append(1/sm)
        ws.append(sm_bin.copy())
        sm_bin.clear()

    w_list = []
    for val,pt in zip(true, pred):
        i = np.argmax(val < h[1])
        j = np.argmax(pt < h[2])
        if ws[i-1][j-1] == 0:
            w_list.append(0)
            continue
        w_list.append(ws[i-1][j-1])
    return np.array(w_list)

def plotchi(chi_true_list, chi_list, pt_list, weights):
    list_chi_t = torch.concat(chi_true_list)
    list_chi = torch.concat(chi_list)
    
    #print(len(list_chi_t[list_chi_t == 1.0]))
    #print(len(list_chi[list_chi == 1.0]))
    #print(cons[list_chi == 1.0])
    
    w_pred = predweights(list_chi_t.numpy(), list_chi.numpy())

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    h = axes.hist2d(list_chi_t.numpy(), list_chi.numpy(), weights=w_pred, cmap='Greens', range=([0.3,1],[0.3,1]), bins=15)#, weights=list_weights.numpy())
    plt.colorbar(h[3])
    axes.plot(np.arange(0.1, 1.06, 0.06), np.arange(0.1, 1.06, 0.06), color='black')
    
    x_list, y_list, stdv_list = errorba(list_chi_t.numpy(), list_chi.numpy())
    axes.plot(x_list, y_list, color='r')
    xerr = (h[1][1]-h[1][0])/2
    axes.errorbar(x_list,y_list, yerr=stdv_list, xerr=xerr, color='r', fmt='o', capsize=4, markersize=2)
    
    axes.set_title("True $\chi_{jp}$ versus predicted $\chi_{jp}$")
    axes.set_xlabel("True $\chi_{jp}$")
    axes.set_ylabel("Predicted $\chi_{jp}$")
    #axes[1].hist(list_chi.numpy(), density=True, bins=20, weights=list_weights.numpy())
    #axes[1].set_title("True $\chi$")
    #axes[1].set_xlabel("True $\chi$")

    
    #axes[1].hist(list_chi.numpy())#, weights=list_weights)
    #axes[1].set_title("Predicted $\chi$")
    #axes[1].set_xlabel("Predicted $\chi$")
    
    #axes[2].hist(list_chi_t, weights=list_weights)
    
    fig.tight_layout()
    plt.show()

def plotobs(chi_pred, cons, theta, kt):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 6))
    
    axes[0].hist(theta[chi_pred==1.0])
    axes[0].set_title("theta")
    axes[1].hist(cons[chi_pred==1.0])
    axes[1].set_title("nr constituents")
    axes[2].hist(kt[chi_pred==1.0])
    axes[2].set_title("kt")
    
    fig.tight_layout()
    plt.show()
    
def errorba(true, pred):
    h2 = np.histogram2d(true,pred, range=([0.3,1], [0.3,1]), bins=15)

    y_list = []
    x_list = []
    stdev = []

    dist_y = (h2[2][1]- h2[2][0])/2
    for i in range(len(h2[0])):
        if (h2[0][i]>0).any():
            #exist some point in the x column
            if not sum(h2[0][i]>0) > 1:
                #If only in one bin
                am = np.argmax(h2[0][i])
                y1 = h2[2][am]
                y2 = h2[2][am+1]
                mean = (y1 + y2)/2
                y_list.append(mean)
                x_list.append((h2[1][i]+h2[1][i+1])/2)
                stdev.append((y2-y1)/2)
            else:
                #If points in more than one bin
                wei = []
                bin_c = []
                for j, ki in enumerate(h2[0][i]):
                    if not ki==0.0:
                        wei.append(ki)
                        bin_c.append((h2[2][j]+h2[2][j+1])/2)

                mean = np.average(bin_c, weights=wei)
                
                y_list.append(mean)
                x_list.append((h2[1][i]+h2[1][i+1])/2)
                stdev.append(np.sqrt(np.average((np.array(bin_c) - mean)**2, weights=wei)))
                

    return x_list, y_list, stdev
    
def plotgraph(batch): 
    g = batch.batch_graph
    labs = batch.label
    labs = labs
    print(labs)
    print(batch.features)
    print(batch.pt)
    
    g = g
    print(g.number_of_nodes())
    edgs = g.edges()
    edgx = edgs[0]
    edgy = edgs[1]
    
    print(edgx)
    print(edgy)
    
    x_coord = []
    y_coord = []
    
    x_co = []
    y_co = []
    
    #plt.scatter(0,0)
    
    last = None
    last_pt = None
    
    node_x = [0]*(g.number_of_nodes())
    node_y = [0]*(g.number_of_nodes())
    
    list_x = []
    node_kt = [0]*(g.number_of_nodes())
    node_theta = [0]*(g.number_of_nodes())
    node_text = [0]*g.number_of_nodes()
    list_tf = []
    #print(edgx.size())
    list_nod = []
    for i in range(edgx.size()[0]):
        x = edgy[i] #Second node
        p = edgx[i] #First node
        print(x, p)
        #print(theta_a)
        
        if x in list_x:
            #Not do same edge twice
            continue
        list_x.append(p)
        
        #Get coordinates for first node (might have been updated previously)
        x0 = node_x[p]
        y0 = node_y[p]

        #Find relevant features
        z_b = batch.features[p][0] #z from the first node = z_b
        kt_x = batch.features[x][2] #kt from second node 
        theta_x = batch.features[x][1] # and theta from second node to make tf
        
        #Find the two opening angles
        theta_0 = math.exp(batch.features[p][1])
        theta_a = batch.features[p][3]
        theta_b = (theta_0-z_b*theta_a)/(1-z_b)

        #Calculate tf
        tf = 2/(math.exp((kt_x+theta_x)))#kt_x*theta_x)
        list_tf.append(tf)
        
        #Decide which opening angle to use
        theta_use = theta_a #for the hardest node
        if (last==p):
            #For the softest node (b)
            theta_use = theta_b
        
            
        #Find how much change in x and y direction
        new_x = np.log(tf)*np.sin(theta_use)
        new_y = np.log(tf)*np.cos(theta_use)
        
        #Decide sign 
        y1 = -new_y
        x1 = new_x
        if (last==p):
            x1 = -new_x
            
        #If the first node is THE first node  
        if i == 0:
            #Save info for graph
            node_kt[0] = math.exp(batch.features[p][2])
            node_theta[0] = math.exp(batch.features[p][1])
            node_text[0]= 'kT of node number %d: %f ' %(0, math.exp(batch.features[p][2]))
            list_nod.append(0)
            #If it split to a final state particle and a branch
            if np.unique(edgx,return_counts=True)[1][0]==1:
                x_co.append(x0)
                x_co.append(x0-0.05)
                x_co.append(None)
                y_co.append(y0)
                y_co.append(y0+y1)
                y_co.append(None)
        
        #If the first node has only 2 connections, then it also should have a final state particle
        if np.unique(edgx,return_counts=True)[1][p]==2 and p!=0:
            #Use the second angle
            theta_use = theta_b
            #Scale the new change so its not too large
            new_x1 = 0.5*np.log(tf)*np.sin(theta_use)
            new_y1 = 0.5*np.log(tf)*np.cos(theta_use)
            #Save final state 
            x_co.append(x0)
            x_co.append(x0+new_x1)
            x_co.append(None)
            y_co.append(y0)
            y_co.append(y0-new_y1)
            y_co.append(None)
            #Update so that the next node is to the left
            x1 = -new_x
        
        #If the last node has only one connection, should have 2 final state particles
        if np.unique(edgx,return_counts=True)[1][x]==1:
            #Have even angle between final state particles
            theta_b = theta_x/2
            new_x1 = 0.05*np.sin(theta_b) #Uneven scaling for beauty purposes
            new_y1 = 8*np.cos(theta_b)
            #If the new y coord is bigger than the last nodes updated y coord
            if (y0+y1+new_y1 > y0+y1):
                new_y1 = -new_y1
            #Save the two final state particles
            x_co.append(x0+x1)
            x_co.append(x0+x1-new_x1)
            x_co.append(None)
            y_co.append(y0+y1)
            y_co.append(y0+y1+new_y1)
            y_co.append(None)
            x_co.append(x0+x1)
            x_co.append(x0+x1+new_x1)
            x_co.append(None)
            y_co.append(y0+y1)
            y_co.append(y0+y1+new_y1)
            y_co.append(None)
        
        #Save node info
        node_x[x] = x0+x1
        node_y[x] = y0+y1
        node_kt[x] = math.exp(kt_x)
        node_theta[x] = math.exp(theta_x)
        node_text[x]= 'kT of node number %d: %f ' %(x, math.exp(kt_x))
        
        #Save nodes in graph
        x_coord.append(x0)
        y_coord.append(y0)
        x_coord.append(x0+x1)
        y_coord.append(y0+y1)
        x_coord.append(None)
        y_coord.append(None)
        
        list_nod.append(x)
        
        last=p


    edge_trace = go.Scatter(
        x=x_coord, y=y_coord,
        line=dict(width=1.0, color='black'),
        hoverinfo='none',
        mode='lines')
    
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node kT',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    print(x_coord[1])
    
    
    
    #x_co = [0, 1,None]#[node_x[7], node_x[1], None]
    #y_co = [0, 0, None]#[node_y[7], node_y[1], None] 
    extra_line = go.Scatter(
                x=x_co,
                y=y_co,
                line=dict(width=1.0, color='Black', dash='dash'), mode='lines',
                hoverinfo='none')
    
    
    node_trace.marker.color = node_kt
    node_trace.text = node_text
    
    fig = go.Figure(data=[edge_trace, node_trace, extra_line],
    layout=go.Layout(
    title='<br>Network graph made with Python',
    titlefont_size=16,
    showlegend=False,
    hovermode='closest',    
    margin=dict(b=20,l=5,r=5,t=40),
    annotations=[ dict(
        text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.005, y=-0.002 ) ],
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    fig.show()

    
    #so = np.array(sorted(zip(node_kt, list_nod), key=lambda x: x[1]))
    #plt.plot(np.arange(g.number_of_nodes(), step=1), node_kt)
    #plt.hist(np.log(list_tf))
    #plt.title("$k_T$ at splittings")
    #plt.xlabel("tf")
    #plt.xlim(0, g.number_of_nodes()+1)
    #plt.ylim(0,max(node_kt)+0.1)
    #plt.show()
    #print(np.log(1/np.array(node_theta[0])),np.log(np.array(node_kt[0])))
    
    plt.plot(np.log(1/np.array(node_theta)),np.log(np.array(node_kt)), 'bo-' )
    plt.plot(np.log(1/np.array(node_theta[0])),np.log(np.array(node_kt[0])), 'b*', markersize=20, label="First Node")
    plt.xlabel("ln(1/$\Theta$)")
    plt.ylabel("ln($k_T$)")
    plt.title("Lund plane for graph")
    plt.legend()
    #plt.show()

def plotchi2(chi_true, chi_pred, weights):
    list_chi_t = torch.concat(chi_true)
    list_chi = torch.concat(chi_pred)
    list_weights = torch.concat(weights)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    
    axes[0].hist(list_chi_t, weights=list_weights, bins=25)
    axes[0].set_title("True $\chi_{jh}$")
    axes[0].set_xlabel("True $\chi_{jh}$")
    
    axes[1].hist(list_chi, weights=list_weights, bins=25)
    axes[1].set_title("Predicted $\chi_{jh}$")
    axes[1].set_xlabel("Predicted $\chi_{jh}$")
    
    fig.tight_layout()
    plt.show()
    
#Background particle stuff
def blastWave(p_T, beta_T, T_F, A_i):
    rho = np.arctanh(beta_T)
    inK = p_T*np.cosh(rho)/T_F
    inI = p_T*np.sinh(rho)/T_F
    i0 = iv(0, inI)
    k1 = kv(1, inK)
    return A_i*p_T*k1*i0

def getBackground(N, beta_T, T_F, max_f):
    A_i = 1
    maxeta = 2.5
    rng = np.random.default_rng(12345)
    listjet = []
    
    phi_l = []
    eta_l = []
    pt_l = []
    for i in range(N):
        phi = rng.uniform(0.0, 2*np.pi)
        eta = rng.uniform(-maxeta, maxeta)
        pt = 0.0
        done = False
        while not done:
            rand3 = rng.uniform(0.0, 1000.0)
            rand4 = rng.uniform(0.0, 1.0)
            if rand4 > blastWave(rand3, beta_T, T_F, A_i)/max_f:
                continue
            else:
                pt = rand3
                done=True
        
        #Create 4-vector
        Eni = pt*np.cosh(eta)*pow(10,-3)
        px = pt*np.cos(phi)*pow(10,-3)
        py = pt*np.sin(phi)*pow(10,-3)
        pz = pt*np.sinh(eta)*pow(10,-3)
        
        listjet.append(fj.PseudoJet(px, py, pz, Eni))
        
        phi_l.append(phi)
        eta_l.append(eta)
        pt_l.append(pt)
        
    #plotspec(phi_l, eta_l, pt_l)
    return listjet
        
def plotspec(phi, eta, pt):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))
    
    axes[0].hist(phi, density=True)
    axes[0].set_title("$\phi$ distribution")
    axes[0].set_xlabel("$\phi$")
    axes[0].set_ylabel("A.U.")
    axes[1].hist(eta, density=True)
    axes[1].set_title("$\eta$ distribution")
    axes[1].set_xlabel("$\eta$")
    axes[1].set_ylabel("A.U.")
    axes[2].hist(np.array(pt)*pow(10,-3), density=True)
    axes[2].set_title("$p_T$ distribution")
    axes[2].set_xlabel("$p_T$ (GeV)")
    axes[2].set_ylabel("A.U.")
    
    fig.tight_layout()
    plt.show()
    
    
class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)