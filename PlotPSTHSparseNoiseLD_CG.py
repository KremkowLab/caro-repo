#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 22:12:49 2018

@author: jenskremkow
"""
# %%
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

import scipy.ndimage as nd   # very nice way of smoothing n-d data

# % matplotlib magic in Ipython console to show figures in separate windows
# %% 
stimulus = np.load('/Users/carolingehr/Dropbox/Lab/VisualStimuli/sparse noisefine grid/sparse_noise_fine_grid_24_16_180816.npy')
# stimulus.shape

#stimulus = io.loadmat('/Users/carolingehr/Documents/KremkowLab/Coding/ANALYSIS/VisStimPsyPy/stimuli/sl_12_8_161010b.mat')
# stimulus['stimulus_frames'].shape
# %% SL

experiment_dir_local = "/Volumes/Elements/data/OnlineAnalysis/20180907_LGN/" #"/Users/jenskremkow/Desktop/analysis/20180907_LGN/"
python_data_dir_local = experiment_dir_local+"onlinedata/"
filename = "b11eslf"


filename_save = filename  # ??

figure_dir = experiment_dir_local+"figures/"

# Load data: contains spiketimes and events
data = np.load(python_data_dir_local+filename+".npy").item() #, encoding='latin1').item()  # needs encoding key input for python 3.x;  load with .item as it is a .npy file (dict)

#data.keys()

# this part combines all spiketimes from different units
frametimes = data['events']['2']['0']        # [2][]0] gives franetimes

spiketimes_all_chs_all_units = data['spiketimes']  # len = 32 channels

chs = spiketimes_all_chs_all_units.keys()   #starts with 24, total = 32

# now get the spiketimesfor the corresponding frametime for channel
spiketimes = {}


for ch in chs:   # go through channels, not ordered
    
    units = spiketimes_all_chs_all_units[ch].keys()
    
    st_tmp = np.array([])
    
    for un in units:
        
        st = spiketimes_all_chs_all_units[ch][un]
        
        st_tmp = np.append(st_tmp,st)    # st (array with times) is stored in big st_tmp array
    
    st_tmp.sort()
    spiketimes[ch] = st_tmp       # spiketimes['2'] 
    

# %% Create PSTHs
psth_range = np.arange(-0.1,0.5,0.001)
n_psth = psth_range.shape[0]-1
n_stimX = stimulus.shape[0]
n_stimY = stimulus.shape[1]
n_ch = 32
n_pol = 1 # this is not used here. But if you have 

psths = np.zeros((n_stimX,n_stimY,n_psth,n_pol,n_ch))

polarities = [1.] # this only used in a sparse noise with dark (-1) and light (1) spots



for ch in range(n_ch):
    if (str(ch+1)) in spiketimes:  # Python 2: use has_key() -_> if spiketimes.has_key(str(ch+1)):
        st = spiketimes[str(ch+1)]
        print(ch)
        for stimX in range(stimulus.shape[0]):
            for stimY in range(stimulus.shape[1]):
                for poln, target in enumerate(polarities):
                    use_index, = np.where(stimulus[stimX,stimY,:]==target)
                    
                    psth_tmp = np.zeros((n_psth))
                    
                    for trigger in use_index:
                        ft_tmp = frametimes[trigger]
                        
                        hs = np.histogram(st-ft_tmp,psth_range)
                        psth_tmp += hs[0]
                
                    psth_tmp = psth_tmp / float(use_index.shape[0])      # .shape = 599
                    psths[stimX,stimY,:,poln,ch] = psth_tmp   # combine with stimulus, stsh.shape = (16, 24, 599, 1, 32)
                


# %% Save psths so no need to load it again if restarted kernel. 
#you could save the psth and then reload it later. Calculating the psth takes time
#data = {}       # create empty dictionary for psths
#data['psths'] = psths
#data['psth_range'] = psth_range

#store_dir = "/Users/carolingehr/Documents/KremkowLab/Coding/ANALYSIS/figures/20180907_LGN/"
#np.save(store_dir+filename_save+'_psths',data)    # saves the npy file
                    
        

# %% plots the rf and psth at peak pixel --> unravel
chs = np.array([12,13,14,20,21,22,26,27,28,29,30,31])-1 # which channels, -1 because python index starts at 0, ch 1 = 0
n_chs = len(chs)

psth_range_analysis = psth_range[0:-1]
integration_window = [0.04,0.18] # this is the integration window for the RF. Should be centered around the onset response

nf = 2
fig1 = plt.figure(nf)
plt.clf()
mngr = plt.get_current_fig_manager()

#mngr.window.setGeometry(80, 44, 2373, 470) #

fontsize = 8.
# %
fig, axes = plt.subplots(2,n_chs,num=nf)

psths_smooth = psths.copy() # we need to copy as otherwise the data in psths is also changed

# smoothing the data. I am not 100% sure what the truncate parameter is. Just changed the sigmas until it looked nice
s = 1.
sigma_space = 0.5
sigma_time = 25.
sigma_pol = 0. # does not make sense to smooth across polarities
sigma_chs = 0.1
sigmas = [sigma_space,sigma_space,sigma_time,sigma_pol,sigma_chs] #
w = 3
t = (((w - 1)/2)-0.5)/s


psths_smooth = nd.filters.gaussian_filter(psths_smooth, sigma=sigmas, truncate=t) # amazing, snooting in 



pol = 0
for ich, ch in enumerate(chs):
    psth = psths_smooth[:,:,:,pol,ch].squeeze()

    use_index = (psth_range_analysis >= integration_window[0]) & (psth_range_analysis < integration_window[1])
    
    rf = psth[:,:,use_index].mean(2)
    rf = rf-rf.mean()
    rf = rf / np.abs(rf).max()
    rf = rf * target
    
    
    ind = np.argmax(np.abs(rf))
    x,y=np.unravel_index(ind,rf.shape)
    
    row = ich
    column = 0
    plt.sca(axes[column,row])
    plt.pcolor(rf)
    plt.clim([-1,1])
    plt.xticks([])
    plt.yticks([])
    plt.title(str(ch+1),fontsize=fontsize)
    
    
    # psth
    column = 1
    plt.sca(axes[column,row])
    plt.plot(psth_range_analysis,psth[x,y,:])
    plt.xticks([0.,0.1,0.2,0.3,0.4,0.5],fontsize=fontsize)
    plt.yticks([])

    plt.xlim([-0.1,0.4])
    plt.xlabel('Time (s)',fontsize=fontsize)
plt.draw_all()
plt.tight_layout()
plt.subplots_adjust(wspace=0.1,hspace=0.1)


#plt.show()
#plt.pause(1.)
#plt.draw()

# store figure --> first net to plt.gcf() to get current figure to save the right one and not the new figure
store_dir = "/Users/carolingehr/Documents/KremkowLab/Coding/ANALYSIS/figures/20180907_LGN/"
#save_name = figure_dir+filename+"receptivefield_psth.png"
save_name = store_dir+filename+"receptivefield_psth.png"
plt.savefig(save_name, dpi=200)  # plt.savefig must be used before plt.show()


# 
# %% 
plt.show()
plt.pause(1.)
