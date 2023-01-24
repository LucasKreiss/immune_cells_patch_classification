# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:30:30 2022

@author: Kreiss
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

def plot_learningrate(lr,training_losses_mean,training_losses_std,title,save_folder):
    Nepochs = len(training_losses_mean)
    epochs = np.linspace(0, Nepochs,Nepochs)
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs,lr,label='train', color='#CC4F1B')
    plt.xlabel('epochs')
    plt.ylabel(title)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    #plt.plot(epochs,lr,label='train', color='#CC4F1B')
    plt.errorbar(lr,training_losses_mean,training_losses_std,alpha=0.5,color='#CC4F1B')
    plt.ylabel('Train loss')
    plt.xlabel(title)
    plt.legend()
    plt.gcf().set_dpi(300)
    filename = 'LearningRate.png'
    plt.gcf().savefig(save_folder/filename,  dpi=1000)
    plt.show()

def plot_loss_curves_std(training_losses_mean,training_losses_std, validation_losses_mean,validation_losses_std,loss_metric,save_folder):
    Nepochs = len(training_losses_mean)
    epochs = np.linspace(0, Nepochs,Nepochs)
    
    plt.plot(epochs,training_losses_mean,label='train', color='#CC4F1B')
    plt.plot(epochs,validation_losses_mean,label='val', color='#1B2ACC')
    
    plt.xlabel('epochs')
    plt.ylabel(loss_metric+' loss')
    plt.legend()
    plt.errorbar(epochs,training_losses_mean,training_losses_std,alpha=0.5,color='#CC4F1B')
    plt.errorbar(epochs,validation_losses_mean,validation_losses_std,alpha=0.5, color='#1B2ACC')
    plt.gcf().set_dpi(300)
    filename = 'Training_curves_'+loss_metric+'.png'
    plt.gcf().savefig(save_folder/filename,  dpi=1000)
    plt.show()
    
def plot_loss_curves_min_max(training_losses_mean,training_losses_min,training_losses_max, validation_losses_mean,validation_losses_min,validation_losses_max,loss_metric,save_folder):
    Nepochs = len(training_losses_mean)
    epochs = np.linspace(0, Nepochs,Nepochs)
    
    plt.plot(epochs,training_losses_mean,label='train', color='#CC4F1B')
    plt.plot(epochs,validation_losses_mean,label='val', color='#1B2ACC')
    
    plt.xlabel('epochs')
    plt.ylabel(loss_metric+' loss')
    plt.legend()
    plt.fill_between(epochs,training_losses_mean-training_losses_min,training_losses_mean+training_losses_max,alpha=0.5,edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.fill_between(epochs,validation_losses_mean-validation_losses_min,validation_losses_mean+validation_losses_max,alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
    plt.gcf().set_dpi(300)
    filename = 'Training_curves_min_max_area_'+loss_metric+'.png'
    plt.gcf().savefig(save_folder/filename,  dpi=1000)
    plt.show()


def plot_training_curves(Nepochs, loss, acc, labels,save_folder):
    
    epochs = np.linspace(0, Nepochs,Nepochs-1)
    
    plt.subplot(1, 2, 1)
    for i_label in range(len(labels)):
        plt.plot(epochs,loss[i_label],label=labels[i_label])

    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i_label in range(len(labels)):
        plt.plot(epochs,acc[i_label],label=labels[i_label])

    plt.xlabel('epochs')
    plt.ylabel('Acc')
    plt.legend()
    
    plt.show()
    
def plot_example_image(x,y,label, save_folder,index_example_imag,range_hist):
    
    print(f'Input: shape = {x.shape}; type = {x.dtype}')
    for idx_ch in range(x.shape[0]):
        print(f'Input Ch{idx_ch}: min = {x[idx_ch,:,:].min()}; max = {x[idx_ch,:,:].max()}')

    print(f'Target: shape = {y.shape}; type =patches {y.dtype}')
    print(f'Target: min = {y.min()}; max = {y.max()}')
    
    cmap = colors.ListedColormap(['white', 'red'])
    fig = plt.figure()
    
    fig.suptitle('class = '+label)
    for idx_ch in range(x.shape[0]):
        plt.subplot(x.shape[0]+1, 2, 2*idx_ch+1)
        img = plt.imshow(x[idx_ch,:,:],vmin=range_hist[0],vmax=range_hist[1])
        plt.title('Ch'+str(idx_ch+1)+' Input')   
        
        # plot the pixel values
        plt.subplot(x.shape[0]+1, 2, 2*idx_ch+2)
        histogram, bin_edges = np.histogram(x[idx_ch,:,:],range=range_hist)#, bins=256, range=(0, 1))
        plt.plot(bin_edges[0:-1], histogram)
        #plt.hist(x[0,:,:].ravel(), density=True)
        plt.xlabel("pixel values")
        plt.ylabel("count")



    plt.subplot(x.shape[0]+1, 2, 2*x.shape[0]+1,)
    plt.imshow(y,vmin=range_hist[0],vmax=range_hist[1])   #, cmap="gray"
    plt.title('Target')
    
    
    # plot the pixel values
    plt.subplot(x.shape[0]+1, 2, 2*x.shape[0]+2)
    histogram, bin_edges = np.histogram(y,range=range_hist)#, bins=256, range=(0, 1))
    plt.plot(bin_edges[0:-1], histogram)
    #plt.hist(y[0,:,:].ravel(), density=True)
    plt.xlabel("pixel values")
    plt.ylabel("count")
    plt.gcf().set_dpi(300)
    filename = 'Test_image_in_Trainer_img'+str(index_example_imag)+'.png'
    plt.gcf().savefig(save_folder/filename) #  dpi=400
    plt.show()
    
def plot_example_prediction(x,y,y_pred,label,metric,save_folder,index_example_imag,range_hist,tag):
    
    #print(f'x: shape = {x.shape}; type = {x.dtype}')
    #print(f'x1: min = {x[0,:,:].min()}; max = {x[0,:,:].max()}')
    #print(f'x2: min = {x[1,:,:].min()}; max = {x[1,:,:].max()}')

    #print(f'y: shape = {y.shape}; type =patches {y.dtype}')
    #print(f'y: min = {y.min()}; max = {y.max()}')
    
    cmap = colors.ListedColormap(['white', 'red'])
    
    n_examples = 5    

    fig = plt.figure()
    #print('GT class = '+str(y)+'; prediction ='+str(y_pred))
    #fig.suptitle('row1 = Ch1 input, ow2 = Ch2 input, row 3 = target GT, row4 = pred output, text = class and metric')
    cmap = colors.ListedColormap(['white', 'red'])
    


    # legend:
    plt.subplot(x.shape[1]+4,n_examples+1,(n_examples+1)*x.shape[1]+1)
    plt.gca().text(0.05, 0.5, 'Target (GT)', fontsize=8, verticalalignment='top', color='black')
    plt.axis('off')
    
    plt.subplot(x.shape[1]+4,n_examples+1,(n_examples+1)*(x.shape[1]+1)+1)
    plt.gca().text(0.05, 0.5, 'Prediction', fontsize=8, verticalalignment='top', color='black')
    plt.axis('off')
    
    plt.subplot(x.shape[1]+4,n_examples+1,(n_examples+1)*(x.shape[1]+2)+1)
    plt.gca().text(0.05, 0.5, 'Class', fontsize=8, verticalalignment='top', color='black')
    plt.axis('off')
    
    plt.subplot(x.shape[1]+4,n_examples+1,(n_examples+1)*(x.shape[1]+3)+1)
    plt.gca().text(0.05, 0.5, 'SSIM, '+tag, fontsize=8, verticalalignment='top', color='black')
    plt.axis('off')
    
    for idx_plt in range(n_examples):
        for idx_ch in range(x.shape[1]):
            
            # legend:
            plt.subplot(x.shape[1]+4,n_examples+1,(n_examples+1)*idx_ch+1)
            plt.gca().text(0.05, 0.5, 'CH'+str(idx_ch+1)+' Input', fontsize=8, verticalalignment='top', color='black')
            plt.axis('off')
            
            #plot channel
            plt.subplot(x.shape[1]+4,n_examples+1,(n_examples+1)*idx_ch+1+idx_plt+1)
            img = plt.imshow(x[idx_plt,idx_ch,:,:],vmin=range_hist[0],vmax=range_hist[1])
            #plt.title('Ch1')   
            plt.axis('off')
        
        
        #target GT
        plt.subplot(x.shape[1]+4,n_examples+1,(n_examples+1)*(x.shape[1])+2+idx_plt)
        img = plt.imshow(y[idx_plt,0,:,:],vmin=range_hist[0],vmax=range_hist[1])
        #plt.title('Ch2')
        plt.axis('off')
        
        # prediction
        plt.subplot(x.shape[1]+4,n_examples+1,(n_examples+1)*(x.shape[1]+1)+2+idx_plt)
        img = plt.imshow(y_pred[idx_plt,0,:,:],vmin=range_hist[0],vmax=range_hist[1])
        #plt.title('Ch2')
        plt.axis('off')
        
        # class
        plt.subplot(x.shape[1]+4,n_examples+1,(n_examples+1)*(x.shape[1]+2)+2+idx_plt)
        plt.gca().text(0.05, 0.5, label[idx_plt], fontsize=10, verticalalignment='top', color='black')
        plt.axis('off')
        plt.subplot(x.shape[1]+4,n_examples+1,(n_examples+1)*(x.shape[1]+3)+2+idx_plt)
        plt.gca().text(0.05, 0.5, str(np.round(metric[idx_plt],3)), fontsize=10, verticalalignment='top', color='black')
        plt.axis('off')

    plt.gcf().set_dpi(300)
    filename = 'Example_image_with_classification_pred'+str(index_example_imag)+'_'+tag+'.png'
    plt.gcf().savefig(save_folder/filename,  dpi=1000)
    plt.show()
    
def plot_example_classification(x,y,y_pred,save_folder,index_example_imag,range_hist,tag):

    n_batches = x.shape[0]    

    fig = plt.figure()
    #print('GT class = '+str(y)+'; prediction ='+str(y_pred))
    #fig.suptitle('GT class = '+str(y)+'; prediction ='+str(y_pred))
    cmap = colors.ListedColormap(['white', 'red'])
    
    for idx_plt in range(n_batches):
        # input ch 1
        plt.subplot(4,n_batches,idx_plt+1)
        img = plt.imshow(x[idx_plt,0,:,:],vmin=range_hist[0],vmax=range_hist[1])
        #plt.title('Ch1')   
        plt.axis('off')
        
        # input ch 2
        plt.subplot(4,n_batches,(idx_plt+9))
        img = plt.imshow(x[idx_plt,1,:,:],vmin=range_hist[0],vmax=range_hist[1])
        #plt.title('Ch2')
        plt.axis('off')
        
        # input ch 3
        if x.shape[0]==3:
            plt.subplot(4,n_batches,(idx_plt+17))
            img = plt.imshow(x[idx_plt,2,:,:],vmin=range_hist[0],vmax=range_hist[1])
            #plt.title('Ch2')
            plt.axis('off')
            
            # GT and pred as text:
            plt.subplot(4,n_batches,(idx_plt+25))
            textstr = '\n'.join((( r'$y=%.1f$' % (y[idx_plt], )), (r'$y_{pr}=%.2f$' % (y_pred[idx_plt], ))))
            plt.gca().text(0.05, 0.9, textstr, fontsize=8, verticalalignment='top', color='black')
            plt.axis('off')
        else:
            plt.subplot(4,n_batches,(idx_plt+17))
            textstr = '\n'.join((( r'$y=%.1f$' % (y[idx_plt], )), (r'$y_{pr}=%.2f$' % (y_pred[idx_plt], ))))
            plt.gca().text(0.05, 0.9, textstr, fontsize=8, verticalalignment='top', color='black')
            plt.axis('off')            

    plt.gcf().set_dpi(300)
    filename = 'Example_image_with_classification_pred'+str(index_example_imag)+'_'+tag+'.png'
    plt.gcf().savefig(save_folder/filename,  dpi=1000)
    plt.show()