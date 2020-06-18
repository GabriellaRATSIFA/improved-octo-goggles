#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from cellpose import models, io
import glob
import os
import pickle, pprint
import pandas


# In[7]:


# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(gpu=True, model_type='cyto')

# list of files
# PUT PATH TO YOUR FILES HERE!
dir = 'D:\Mirindra\subsampling_cfos_ventral'
os.chdir(dir)
filelist = glob.glob('*.tif')

imgs = [skimage.io.imread(f) for f in filelist]
nimg = len(imgs)

# define CHANNELS to run segmentation on
# channels = [cytoplasm, nucleus]

# if NUCLEUS channel does not exist, set the second channel to 0
channels = [[0,0]]*nimg

# channels = [0,0] # IF YOU HAVE GRAYSCALE
filelist


# In[13]:


#"run segmentation" 
masks, flows, styles, diams = model.eval(imgs, diameter= 12.4,flow_threshold=0.2, cellprob_threshold=5.0, do_3D=True, channels=channels)


# In[14]:


tempsave = dict()
#io.masks_flows_to_seg(imgs, masks, flows, diams, channels, filelist)
#io.save_to_png(imgs, masks, flows,filelist)
for i,file in enumerate(filelist):
    filename=str(os.path.splitext(file)[0]+'_tempsave.pkl')   
    
    tempsave[file]={"imgs":imgs[i],"masks": masks[i],"flows": flows[i],"diams": diams[i],"channels": channels[i]}

    with open(filename, 'wb') as handle:
        pickle.dump(tempsave, handle, protocol=pickle.HIGHEST_PROTOCOL)  
        


# In[15]:


#to open the pickled file
#pkl_file = open('cfos-GroupA-N1-ventr-s2-prox_tempsave.pkl', 'rb')

#11 pour error = 0.2 et cell prob=5.0
for file in filelist:
    pkl_file = open(os.path.splitext(file)[0]+'_tempsave.pkl', 'rb')
    data1 = pickle.load(pkl_file)
    diams = data1[file]['diams']
    masks = data1[file]['masks']
    flows = data1[file]['flows']
    print(file)
    print(diams)
    #print(len(masks[2][500]))
    #print(np.size(masks[2][500]))
    print(np.amax(masks))
    unique, counts = np.unique(masks, return_counts=True)
    #print(dict(zip(unique, counts)))    #to verify if the nb of masks computed is exact or not
    #pprint.pprint(data1)
    pkl_file.close()


# The position for each cells detected has to be collected for a later analysis.
# 

# For each image, the masks are in a 7-sized-ndarray. 

# The 'masks' is an 7 x 512 x 512 ndarray. It contains the labels of all cells' masks : 0 if there is no masks and 1,2,3,....200,201....are the masks labels.
# The information about each image needs to be gathered inside a dictionary, which keys are the filenames and the subdicitonaries contain subkeys such as : diameter estimated, nb. masks computed, flow, position of each cell.
