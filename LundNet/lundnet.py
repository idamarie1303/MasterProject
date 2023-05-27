# This file is part of LundNet by F. Dreyer and H. Qu

"""
    lundnet.py: the entry point for LundNet.
"""

from __future__ import print_function
import tqdm
import torch
import numpy as np
import networkx as nx
from functools import partial
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torch.utils.data import DataLoader
import os, time, datetime, argparse, pickle
from torchinfo import summary

from lundnets.dgl_dataset import DGLGraphDatasetParticle, DGLGraphDatasetLund, collate_wrapper, collate_wrapper_tree
#Homemade functions
from lundnets.functions_lundnet import findEn, findWeights, findWeights2d, plotloss, plotchi, LogCoshLoss, plotgraph, plotchi2



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--train-sig', type=str, default='')
    parser.add_argument('--train-bkg', type=str, default='')
    parser.add_argument('--val-sig', type=str, default='')
    parser.add_argument('--val-bkg', type=str, default='')
    parser.add_argument('--test-sig', type=str, default='')
    parser.add_argument('--test-bkg', type=str, default='')
    parser.add_argument('--model', type=str, default='lundnet5', choices=['lundnet5', 'lundnet2',
                                                                          'lundnet3', 'lundnet4',
                                                                          'particlenet', 'particlenet-lite'])
    parser.add_argument('--ln-kt-min', type=float, default=None)#np.log(0.1))
    parser.add_argument('--ln-delta-min', type=float, default=None)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--name', type=str, default='model')
    parser.add_argument('--test-output', type=str, default='')
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--nev', type=int, default=-1)
    parser.add_argument('--nev-val', type=int, default=-1)
    parser.add_argument('--nev-test', type=int, default=-1)
    parser.add_argument('--start-lr', type=float, default=0.01)
    parser.add_argument('--lr-steps', type=str, default='5, 10, 15, 20, 25, 30, 35, 40, 45, 50')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    start = time.time()

    if 'lund' in args.model:
        from lundnets.JetTree import JetTree, LundCoordinates
        if args.model == 'lundnet4':
            LundCoordinates.change_dimension(4, ['lnz', 'lnDelta', 'lnKt', 'lnm'])
        elif args.model == 'lundnet3':
            LundCoordinates.change_dimension(3, ['lnz', 'lnDelta', 'lnKt'])#, 'theta1'])
        elif args.model == 'lundnet2':
            LundCoordinates.change_dimension(2, ['lnDelta', 'lnKt'])
        kt_min = np.exp(args.ln_kt_min) if (args.ln_kt_min is not None and args.ln_kt_min > -99) else 0
        delta_min = np.exp(args.ln_delta_min) if args.ln_delta_min is not None else 0
        JetTree.change_cuts(kt_min, delta_min)
        print('Using %s, kt_min=%f and delta_min=%f' % (args.model, JetTree.ktmin, JetTree.deltamin))

    if args.demo: #The reader only reads bkg
        args.train_bkg = '../../../real_hadron_sample_py.json.gz'


    # training/testing mode
    if args.train_bkg:
        training_mode = True
    else:
        assert(args.load)
        training_mode = False

    # data format
    DGLGraphDataset = DGLGraphDatasetLund if 'lund' in args.model else DGLGraphDatasetParticle

    # model parameter
    if args.model == 'particlenet':
        from lundnets.ParticleNet import ParticleNet
        Net = ParticleNet
        conv_params = [
            (16, (64, 64, 64)),
            (16, (128, 128, 128)),
            (16, (256, 256, 256)),
        ]
        fc_params = [(256, 0.1)]
        use_fusion = False
        if args.batch_size <= 0:
            args.batch_size = 256
        collate_fn = partial(collate_wrapper, k=conv_params[0][0])
    elif args.model == 'particlenet-lite':
        from lundnets.ParticleNet import ParticleNet
        Net = ParticleNet
        conv_params = [
            (7, (32, 32, 32)),
            (7, (64, 64, 64))
        ]
        fc_params = [(128, 0.1)]
        use_fusion = False
        if args.batch_size <= 0:
            args.batch_size = 1024
        collate_fn = partial(collate_wrapper, k=conv_params[0][0])
    else:
        from lundnets.LundNet import LundNet
        Net = LundNet
        conv_params = [[32, 32], [32, 32], [64, 64], [64, 64], [128, 128], [128, 128]]
        fc_params = [(256, 0.2)]
        use_fusion = True
        if args.batch_size <= 0:
            args.batch_size = 512#1024
        collate_fn = collate_wrapper_tree ##Create the batch_graph when applied in DataLoader

    # device
    dev = torch.device(args.device)

    # load data
    all_data = DGLGraphDataset(args.train_bkg, nev=args.nev)
    input_dims = all_data.num_features
    #Split in train-val/test 
    train_size = int(0.8 * len(all_data)) #80% training
    val_test_size = len(all_data) - train_size #20% validation/test
    train_data, val_test_data = torch.utils.data.random_split(all_data, [train_size, val_test_size])
    
    if training_mode:
        train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.batch_size,
                                  collate_fn=collate_fn, shuffle=True, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_test_data, num_workers=args.num_workers, batch_size=args.batch_size,
                                collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)
        test_loader = DataLoader(val_test_data, num_workers=args.num_workers, batch_size=args.batch_size,
                                 collate_fn=collate_fn, shuffle=False, drop_last=False, pin_memory=True)
    else:
        test_loader = DataLoader(val_test_data, num_workers=args.num_workers, batch_size=args.batch_size,
                             collate_fn=collate_fn, shuffle=False, drop_last=False, pin_memory=True)


    # model
    model = Net(input_dims=input_dims, num_classes=1,
                conv_params=conv_params,
                fc_params=fc_params,
                use_fusion=use_fusion)
    model = model.to(dev)

    #print(summary(model))
    
    def train(model, opt, scheduler, train_loader, dev, En, lim1, lim2, weight_function):
        """
            Train the model
        """
        model.train()

        total_loss = 0
        num_batches = 0
        count = 0
        
        tic = time.time()
        with tqdm.tqdm(train_loader, ascii=True) as tq:
            for i,batch in enumerate(tq):
                #If one wish to vizualize the graphs uncomment this section and change the batchsize to 1
                #if i < 0:
                #    continue
                #plotgraph(batch)
                #break
                
                label = batch.label
                num_examples = label.shape[0]
                label = label.to(dev).squeeze()
                weights = weight_function(label, batch.pt.squeeze(), En, lim1, lim2)
                
                opt.zero_grad()
                
                output = model(batch.batch_graph.to(dev), batch.features.to(dev))
                loss = loss_func(torch.reshape(output, (-1,)), label)
                #Weighted loss function
                loss = (loss*weights/weights.sum())   
                
                #Backpropagation  
                loss.mean().backward()
                opt.step()
                
                #Register loss etc. 
                num_batches += 1
                count += num_examples
                
                loss = loss.mean().item()
                total_loss += loss

                tq.set_postfix({
                    'Loss': '%.10f' % loss,
                    'AvgLoss': '%.10f' % (total_loss / num_batches)})

        scheduler.step()

        ts = time.time() - tic
        print('Trained over {count} samples in {ts} secs (avg. speed {speed} samples/s.)'.format(
            count=count, ts=ts, speed=count / ts))
        return total_loss/num_batches

        
   
    def evaluate(model, test_loader, dev, En, lim1, lim2, weight_function, return_scores=False, return_time=False):
        """
            Evaluates model
        """
        model.eval()

        count = 0
        total_loss = 0
        num_batches = 0
        scores = []
        tic = time.time()

        with torch.no_grad():
            with tqdm.tqdm(test_loader, ascii=True) as tq:
                true_chi = []
                pred_chi = []
                true_pt = []
                list_weights = []
                nr_cons = []
                
                for batch in tq:
                    
                    label = batch.label.to(dev).squeeze()
                    true_chi.append(label)
                    nr_cons.append(batch.nr.squeeze())

                    output = model(batch.batch_graph.to(dev), batch.features.to(dev))
                    pred_chi.append(torch.reshape(output, (-1,)))
                    true_pt.append(batch.pt.squeeze())
                    
                    if return_scores:
                        scores.append(torch.reshape(output, (-1,)).cpu().detach().numpy())

                    weights = weight_function(label, batch.pt.squeeze(), En, lim1, lim2)
                    list_weights.append(weights)
                    
                    loss = loss_func(torch.reshape(output, (-1,)), label)
                    loss = loss.mean().item()
                    
                    num_examples = label.shape[0]
                    count += num_examples
                    total_loss += loss
                    num_batches +=1

                    tq.set_postfix({
                        'Loss': '%.10f' % loss,
                        'AvgLoss': '%.10f' % (total_loss / num_batches)})

                
        ts = time.time() - tic
        print('Tested over {count} samples in {ts} secs (avg. speed {speed} samples/s.)'.format(
            count=count, ts=ts, speed=count / ts
        ))
        if return_time:
            return ts

        if return_scores:
            end = time.time()
            print(end - start)
            
            plotloss(list_train_loss, list_val_loss, args.num_epochs)
            plotchi(true_chi, pred_chi, true_pt, list_weights)
            plotchi2(true_chi, pred_chi, list_weights)
  

            print("Total loss: {:.10f}".format(total_loss))
            print("Average loss: {:.10f}".format(total_loss/num_batches))

            
            
            return np.concatenate(scores)
        else:
            return total_loss / num_batches
        
    
    def init_weights(m):
        """ 
            Initialize weights
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
    if training_mode:
        # loss function
        loss_func = torch.nn.MSELoss(reduction="none")#LogCoshLoss()
        
        # initializer
        model.apply(init_weights)
        
        # optimizer
        opt = torch.optim.Adamax(model.parameters(), lr=args.start_lr, weight_decay=1e-8)#torch.optim.Adam(model.parameters(), lr=args.start_lr)

        # learning rate
        lr_steps = [int(x) for x in args.lr_steps.split(',')]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=lr_steps, gamma=0.01)
        
        # find weights
        En, lim1, lim2 = findEn([train_loader, val_loader], w2d=True)
        w_function = findWeights2d
    
        # training loop
 
        list_train_loss = []
        list_val_loss   = []
        for epoch in range(args.num_epochs):
            print("Epoch {:d} ".format(epoch))
            t_loss = train(model, opt, scheduler, train_loader, dev, En, lim1, lim2, w_function)
            list_train_loss.append(t_loss)
            v_loss = evaluate(model, val_loader, dev, En, lim1, lim2, w_function)
            list_val_loss.append(v_loss)

    if training_mode:
        del train_data, train_loader, val_loader

    test_preds = evaluate(model, test_loader, dev, En, lim1, lim2, w_function, return_scores=True)
    
