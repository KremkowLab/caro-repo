# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 10:28:54 2018

@author: AGKremkow
"""

# %% Create stimulus matrix for checkerboard

import numpy as np
import matplotlib.pyplot as plt

# %%
date = "20181024"
#experiment_name = "checkerboard_45_45"

# %%
# shared folder for configuration file
#shared_folder = u'C:\\Users\\AGKremkow\\Desktop\\visual stimuli clean\\stimuli\\'

#save_name = shared_folder+date+"_"+experiment_name+".npy"

# Create Stimulus .npy file
# %% Generate Stimulus Matrix
# For Checkerboard n_trials = number of frames
# random integers from 0 to 1, Pixels (rows, cols), frame times
y = 25 #45 #6#16           # this has to be adjusted --> calculate grid size in HowtoCalculateScaleForStimuli.py
x = 45 #45 #11#24
#n_trials = 100

target_size = 1 #2 #3 #1 = one position in the grid, 2 = 2x2 positions in the grid # change 1 (5x5 deg) or 2 (10x10 deg) or 3 (15x15 deg)
n_frames = 5000 #(x*y)*2  #int((x*y)/3.) #(5000) 

# random integers from 0 to 1, Pixel grid, (rows, cols), frame times
stimulus = np.random.randint(-1.,2.,[y,x,n_frames])

print('Number of frames', n_frames)
print('Stimulus duration in sec when presented for 8ms', n_frames*(1/120.))          # duration of stimulus in sec = n_frames * 0.0083s when stim_frames = 1
print('Stimulus duration in minutes when presented for 8ms', n_frames*(1/120.)/60.)        # duration of stimulus in minutes
print('Stimulus duration in minutes when presented for 80ms',n_frames*(1/120.)*10/60.)  # duration of stimulus with frame times 10 = 80ms = n_frames * 0.0083s when stim_frames = 1
print('Stimulus duration in minutes when presented for 160ms',n_frames*(1/120.)*20/60.) 
# %%
np.random.seed(12432325)


# %% Save the .npy file
np.save('Checkerboard_'+str(x)+'_'+str(y)+'_target_size_'+str(target_size)+'_n_frames_'+str(n_frames)+'_'+date, stimulus)# %%

# %% Plot

plt.figure(1)
plt.clf()
plt.ion()

# show first 5 frames
for n in range(5):
    plt.pcolor(stimulus[:,:,n]) 
    plt.draw()
    
    plt.pause(0.05)
    
plt.show()
    