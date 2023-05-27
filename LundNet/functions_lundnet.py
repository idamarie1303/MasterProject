#Functions I made for lundnet modifications
import tqdm
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.special import kv, iv

# For re-weighting 
def Neff(N, beta=0.9998):
    """
    Calculates the efficient number of samples in each bin
    
    N (vactor): a one or two dimensional vector containing the number of samples in each bin 
    beta (float): the probability that a new sample in that bin is independent of the previous one
    """
    
    return (1-pow(beta, N))/(1-beta)

def findEn(dataloaders, w2d=False):
    """
    Find the vector of effective number of samples in a jet, and the limits of the bins
    
    dataloaders (DGLDataloader): Input dataset in batches
    wqd (bool): If True then 2D reweighting, if False then 1D reweighting
    """
    chis=[]
    pts = []
    for dataloader in dataloaders:
        with tqdm.tqdm(dataloader, ascii=True) as tq:
            for batch in tq:
                label = batch.label
                pt = batch.pt
                chis.append(label)
                pts.append(pt)

    if w2d==True:
        h = np.histogram2d(torch.concat(pts).numpy()/1000, torch.concat(chis).numpy(), range=([0,1.6],[0,1.]), bins=25)
        return Neff(h[0]), h[1], h[2]
    else:
        h = np.histogram(torch.concat(chis).numpy(), range=[0,1.], bins=25)
        return Neff(h[0]), h[1], 0

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
        i = np.argmax(np.array(p.numpy()/1000) < lim1)
        j = np.argmax(np.array(val.numpy() < lim2))
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
        j = np.argmax(np.array(val.numpy() < lim))
        if En[j-1] == 0:
            w_list.append(0)
            continue
        w_list.append(1/En[j-1])
    return torch.tensor(w_list)

#Plotting performance
def GetErrorbars(true, pred):
    """
    Calculate the error bars for the performance measure plot
    
    true (vector): true distribution of chi
    pred (vector): predicted distribution of chi
    
    returns: 
        x_list (vector): List of mean values in x-direction
        y_list (vector): List of mean values in y-direction
        stdev (vector): List of standard deviation (size of errorbar) in y-direction
        (Note: size of errorbar in x-diraction is binsize/2)
    """
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
    
def plotloss(train, val, num_epoch):
    """
    Plot the training and validation loss for each number of epochs
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    axes.plot(np.arange(1,num_epoch+1), np.array(train), color='blue', label='Train')
    axes.set_xlabel("Number of Epochs")
    axes.set_ylabel("Loss")
    axes.plot(np.arange(1,num_epoch+1), np.array(val), color='orange', label='Validation')

    plt.legend()

    fig.tight_layout()
    plt.show()
    
def predweights(true, pred):
    """
    Calcualate the probability in each predicted bin, sum of 1 in each column for true values
    
    true (vector): true chi distribution
    pred (vector): predicted chi distribution
    
    returns:
        w_list (vector): list of weights to assign each sample so that the color of the bin correspond to the probability
    """
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
    """
    Plot the predicted performance 
    """
    list_chi_t = torch.concat(chi_true_list)
    list_chi = torch.concat(chi_list)
    
    # find probabilities
    w_pred = predweights(list_chi_t.numpy(), list_chi.numpy())

    #Plot 2D histogram
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    h = axes.hist2d(list_chi_t.numpy(), list_chi.numpy(), weights=w_pred, cmap='Greens', range=([0.3,1],[0.3,1]), bins=15)#, weights=list_weights.numpy())
    plt.colorbar(h[3])
    axes.plot(np.arange(0.1, 1.06, 0.06), np.arange(0.1, 1.06, 0.06), color='black')
    
    # Find and plot errorbars
    x_list, y_list, stdv_list = GetErrorbars(list_chi_t.numpy(), list_chi.numpy())
    axes.plot(x_list, y_list, color='r')
    xerr = (h[1][1]-h[1][0])/2
    axes.errorbar(x_list,y_list, yerr=stdv_list, xerr=xerr, color='r', fmt='o', capsize=4, markersize=2)
    
    axes.set_title("True $\chi$ versus predicted $\chi$")
    axes.set_xlabel("True $\chi$")
    axes.set_ylabel("Predicted $\chi$")
    plt.show()
 
def plotgraph(batch): 
    """
    Vizualize graph. Requires batchsize=1.
    """
    g = batch.batch_graph
    print("Number of nodes: " + g.number_of_nodes())
    
    # Find edges
    edgs = g.edges()
    edgx = edgs[0]
    edgy = edgs[1]
    
    # Define vectors
    x_coord = []
    y_coord = []
    x_co = []
    y_co = []
    
    node_x = [0]*(g.number_of_nodes())
    node_y = [0]*(g.number_of_nodes())
    node_kt = [0]*(g.number_of_nodes())
    node_theta = [0]*(g.number_of_nodes())
    node_text = [0]*(g.number_of_nodes())
    
    list_x = []
    list_tf = []
    list_nod = []
    
    last = None
    #Iterate over all edges
    for i in range(edgx.size()[0]):
        n1 = edgx[i] #First node
        n2 = edgy[i] #Second node
        
        if n2 in list_x:
            #Not do same edge twice
            continue
        list_x.append(n1)
        
        #Get coordinates for first node (might have been updated previously)
        x0 = node_x[n1]
        y0 = node_y[n1]

        #Find relevant features
        z_b = batch.features[n1][0] #z from the first node = z_b
        kt_x = batch.features[n2][2] #kt from second node 
        theta_x = batch.features[n2][1] # and theta from second node to make tf
        
        #Find the two opening angles
        theta_0 = math.exp(batch.features[n1][1])
        theta_a = batch.features[n1][3]
        theta_b = (theta_0-z_b*theta_a)/(1-z_b)

        #Calculate tf
        tf = 2/(math.exp((kt_x+theta_x)))
        list_tf.append(tf)
        
        #Decide which opening angle to use
        theta_use = theta_a #for the hardest node
        if (last==n1):
            #For the softest node (b)
            theta_use = theta_b
        
        #Find how much change in x and y direction
        new_x = np.log(tf)*np.sin(theta_use)
        new_y = np.log(tf)*np.cos(theta_use)
        
        #Decide sign 
        y1 = -new_y
        x1 = new_x
        if (last==n1):
            x1 = -new_x
            
        #If the first node is THE first node  
        if i == 0:
            #Save info for graph
            node_kt[0] = math.exp(batch.features[n1][2])
            node_theta[0] = math.exp(batch.features[n1][1])
            node_text[0]= 'kT of node number %d: %f ' %(0, math.exp(batch.features[n1][2]))
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
        if np.unique(edgx,return_counts=True)[1][n1]==2 and p!=0:
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
        if np.unique(edgx,return_counts=True)[1][n2]==1:
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
        node_x[n2] = x0+x1
        node_y[n2] = y0+y1
        node_kt[n2] = math.exp(kt_x)
        node_theta[n2] = math.exp(theta_x)
        node_text[n2]= 'kT of node number %d: %f ' %(n2, math.exp(kt_x))
        
        #Save nodes in graph
        x_coord.append(x0)
        y_coord.append(y0)
        x_coord.append(x0+x1)
        y_coord.append(y0+y1)
        x_coord.append(None)
        y_coord.append(None)
        
        list_nod.append(n2)
        
        last=n1

    # Create edges
    edge_trace = go.Scatter(
        x=x_coord, y=y_coord,
        line=dict(width=1.0, color='black'),
        hoverinfo='none',
        mode='lines')
    
    #Create nodes
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
    
    
    
    #Create final state particles as dashed lines
    extra_line = go.Scatter(
                x=x_co,
                y=y_co,
                line=dict(width=1.0, color='Black', dash='dash'), mode='lines',
                hoverinfo='none')
    
    #Set node colors and text with kt
    node_trace.marker.color = node_kt
    node_trace.text = node_text
    
    #Put everything togheter
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

    #Plot lundplane for graph    
    plt.plot(np.log(1/np.array(node_theta)),np.log(np.array(node_kt)), 'bo-' )
    plt.plot(np.log(1/np.array(node_theta[0])),np.log(np.array(node_kt[0])), 'b*', markersize=20, label="First Node")
    plt.xlabel("ln(1/$\Theta$)")
    plt.ylabel("ln($k_T$)")
    plt.title("Lund plane for graph")
    plt.legend()
    plt.show()

def plotchi2(chi_true, chi_pred, weights):
    """
    Plot the distribution of the true and predicted chi next to eachother
    """
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
   

#Additional loss functions
class LogCoshLoss(torch.nn.Module):
    """
    From https://github.com/tuantle/regression-losses-pytorch
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)