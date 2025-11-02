import torch
from torch import nn
import sys
from src.models_fidelity_dgcnn import FidelityAwareMultimodalDGCNN
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
from src.eval_metrics import eval_iemocap

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = FidelityAwareMultimodalDGCNN(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    
    settings = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'scheduler': scheduler
    }
    
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    
    # Store training history
    train_losses = []
    valid_losses = []
    epochs = []
    best_valid_loss = float('inf')
    
    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)
            
            model.zero_grad()
            
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision = text.cuda(), audio.cuda(), vision.cuda()
                    eval_attr = eval_attr.cuda().long()
            
            batch_size = text.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            preds, _ = net(text, audio, vision)
            
            # Reshape for loss calculation
            preds = preds.view(-1, 2)
            eval_attr = eval_attr.view(-1)
            
            loss = criterion(preds, eval_attr)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += loss.item() * batch_size
            
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []
        
        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(-1)
                
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision = text.cuda(), audio.cuda(), vision.cuda()
                        eval_attr = eval_attr.cuda().long()
                        
                batch_size = text.size(0)
                net = nn.DataParallel(model) if batch_size > 10 else model
                preds, _ = net(text, audio, vision)
                
                preds = preds.view(-1, 2)
                eval_attr = eval_attr.view(-1)
                total_loss += criterion(preds, eval_attr).item() * batch_size
                
                results.append(preds)
                truths.append(eval_attr)
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_loss = train(model, optimizer, criterion)
        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, results, truths = evaluate(model, criterion, test=True)
        
        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        epochs.append(epoch)
        
        scheduler.step(val_loss)
        
        end = time.time()
        duration = end-start
        
        print("-" * 50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-" * 50)
        
        if val_loss < best_valid_loss:
            print(f"Saved model at pre_trained_models/BEST_{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=f"BEST_{hyp_params.name}")
            best_valid_loss = val_loss

    # Plot and save training curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'plots/{hyp_params.name}_training_curves.png')
    plt.close()

    # Load best model and evaluate
    model = load_model(hyp_params, name=f"BEST_{hyp_params.name}")
    _, results, truths = evaluate(model, criterion, test=True)
    
    print("\nTest performance:")
    eval_iemocap(results, truths)

    return test_loss
