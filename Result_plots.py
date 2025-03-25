# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:27:10 2023

@author: Ahsan
"""

#%%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
from scipy import ndimage
import pandas as pd
import math
# from models import VxmDense_1, VxmDense_2, VxmDense_huge
print('\n\n Libraries are loaded')




#%%
#atlas

#test image
path = 'C:/Drive/Workspace/data_sets/OASIS_dataset/images_5/p_0438_0439.pkl'
file = open(path, 'rb')
test_img = pickle.load(file)
test_img = np.array(test_img)
moving = test_img[0,:,:,:]
atlas = test_img[1,:,:,:]

#model outputs
# case4
images_case4 = np.load('images_flows/case4/images.npy')
flow_case4 = np.load('images_flows/case4/flow.npy')
x_seg_case4 = np.load('images_flows/case4/x_seg.npy')
y_seg_case4 = np.load('images_flows/case4/y_seg.npy')

#transmrph
images_tm = np.load('images_flows/transmorph/images.npy')
flow_tm = np.load('images_flows/transmorph/flow.npy')
x_seg_tm = np.load('images_flows/transmorph/x_seg.npy')
y_seg_tm = np.load('images_flows/transmorph/y_seg.npy')

#vm1
images_vm1 = np.load('images_flows/vm1/images.npy')
flow_vm1 = np.load('images_flows/vm1/flow.npy')
x_seg_vm1 = np.load('images_flows/vm1/x_seg.npy')
y_seg_vm1 = np.load('images_flows/vm1/y_seg.npy')

#vm2
images_vm2 = np.load('images_flows/vm2/images.npy')
flow_vm2 = np.load('images_flows/vm2/flow.npy')
x_seg_vm2 = np.load('images_flows/vm2/x_seg.npy')
y_seg_vm2 = np.load('images_flows/vm2/y_seg.npy')

#vit
images_vit = np.load('images_flows/vit/images.npy')
flow_vit = np.load('images_flows/vit/flow.npy')
x_seg_vit = np.load('images_flows/vit/x_seg.npy')
y_seg_vit = np.load('images_flows/vit/y_seg.npy')

#nnformer
images_nnf = np.load('images_flows/nnformer/images.npy')
flow_nnf = np.load('images_flows/nnformer/flow.npy')
x_seg_nnf = np.load('images_flows/nnformer/x_seg.npy')
y_seg_nnf = np.load('images_flows/nnformer/y_seg.npy')

#cotr
images_cotr = np.load('images_flows/cotr/images.npy')
flow_cotr = np.load('images_flows/cotr/flow.npy')
x_seg_cotr = np.load('images_flows/cotr/x_seg.npy')
y_seg_cotr = np.load('images_flows/cotr/y_seg.npy')


# flowall = np.load('images_flows/case4/allcase4_flow.npy')

#%%
slice_no = 95
fsize = 20
plt.rcParams["font.family"] = "Times New Roman"
fig, axs = plt.subplots(3, 3, figsize = (8, 10))

axs[0,0].imshow(ndimage.rotate(moving[:,:,slice_no], -90), cmap = 'gray')
axs[0,1].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'gray')
axs[0,2].imshow(ndimage.rotate(images_vm1[0,0,:,:,slice_no], -90), cmap = 'gray')
axs[1,0].imshow(ndimage.rotate(images_vm2[0,0,:,:,slice_no], -90), cmap = 'gray')
axs[1,1].imshow(ndimage.rotate(images_vit[0,0,:,:,slice_no], -90), cmap = 'gray')
axs[1,2].imshow(ndimage.rotate(images_nnf[0,0,:,:,slice_no], -90), cmap = 'gray')
axs[2,0].imshow(ndimage.rotate(images_cotr[0,0,:,:,slice_no], -90), cmap = 'gray')
axs[2,1].imshow(ndimage.rotate(images_tm[0,0,:,:,slice_no], -90), cmap = 'gray')
axs[2,2].imshow(ndimage.rotate(images_case4[0,0,:,:,slice_no], -90), cmap = 'gray')

axs[0,0].set_title("$Moving$", fontsize= fsize, pad=10)
axs[0,1].set_title('$Fixed$', fontsize= fsize, pad=10)
axs[0,2].set_title('$VM \ \ 1$', fontsize= fsize, pad=10)
axs[1,0].set_title('$VM \ \ 2$', fontsize= fsize, pad=10)
axs[1,1].set_title('$Vit-V-Net$', fontsize= fsize, pad=10)
axs[1,2].set_title('$nnFormer$', fontsize= fsize, pad=10)
axs[2,0].set_title('$CoTr$', fontsize= fsize, pad=10)
axs[2,1].set_title('$TransMorph$', fontsize= fsize, pad=10)
axs[2,2].set_title('$Ours$', fontsize= fsize, pad=10)
# axs[1,9].set_title('$Ours-b-spline$', fontsize= fsize, pad=10)
axs[0,0].axis('off')
axs[0,1].axis('off')
axs[0,2].axis('off')
axs[1,0].axis('off')
axs[1,1].axis('off')
axs[1,2].axis('off')
axs[2,0].axis('off')
axs[2,1].axis('off')
axs[2,2].axis('off')



plt.savefig('plots/Result_plot_OASIS2', dpi=800)

#%% sample wise comparision graph
data = pd.read_csv("sample_wise_results_for_graph.csv")
plt.figure(figsize=(8, 6), dpi=600)
sns.barplot(data=data, x="Samples", y="Dice", hue="Method")
plt.yticks(np.arange(0, 0.8, 0.05))
plt.legend(loc = 'lower right')
plt.savefig('sample_wise_comparision', dpi=600)
plt.show()

#%% residual plots
fsize = 15
padd = 15
# plt.rcParams["font.family"] = "Times New Roman"
fig, axs = plt.subplots(2, 4, figsize = (12, 8))
slice_no = 95

axs[0,0].imshow(ndimage.rotate(atlas[:,:,slice_no], -90) - ndimage.rotate(moving[:,:,slice_no], -90), cmap = 'RdBu', vmin = -np.std(ndimage.rotate(atlas[:,:,slice_no], -90)), vmax = np.std(ndimage.rotate(moving[:,:,slice_no], -90)))
axs[0,1].imshow(ndimage.rotate(atlas[:,:,slice_no], -90)-ndimage.rotate(images_vm1[0,0,:,:,slice_no], -90), cmap = 'RdBu', vmin = -np.std(ndimage.rotate(atlas[:,:,slice_no], -90)), vmax = np.std(ndimage.rotate(images_vm1[0,0,:,:,slice_no], -90)))
axs[0,2].imshow(ndimage.rotate(atlas[:,:,slice_no], -90)-ndimage.rotate(images_vm2[0,0,:,:,slice_no], -90), cmap = 'RdBu', vmin = -np.std(ndimage.rotate(atlas[:,:,slice_no], -90)), vmax = np.std(ndimage.rotate(images_vm2[0,0,:,:,slice_no], -90)))
axs[0,3].imshow(ndimage.rotate(atlas[:,:,slice_no], -90)-ndimage.rotate(images_vit[0,0,:,:,slice_no], -90), cmap = 'RdBu', vmin = -np.std(ndimage.rotate(atlas[:,:,slice_no], -90)), vmax = np.std(ndimage.rotate(images_vit[0,0,:,:,slice_no], -90)))
axs[1,0].imshow(ndimage.rotate(atlas[:,:,slice_no], -90)-ndimage.rotate(images_nnf[0,0,:,:,slice_no], -90), cmap = 'RdBu', vmin = -np.std(ndimage.rotate(atlas[:,:,slice_no], -90)), vmax = np.std(ndimage.rotate(images_nnf[0,0,:,:,slice_no], -90)))

axs[1,1].imshow(ndimage.rotate(atlas[:,:,slice_no], -90)-ndimage.rotate(images_cotr[0,0,:,:,slice_no], -90), cmap = 'RdBu', vmin = -np.std(ndimage.rotate(atlas[:,:,slice_no], -90)), vmax = np.std(ndimage.rotate(images_cotr[0,0,:,:,slice_no], -90)))
axs[1,2].imshow(ndimage.rotate(atlas[:,:,slice_no], -90)-ndimage.rotate(images_tm[0,0,:,:,slice_no], -90), cmap = 'RdBu', vmin = -np.std(ndimage.rotate(atlas[:,:,slice_no], -90)), vmax = np.std(ndimage.rotate(images_tm[0,0,:,:,slice_no], -90)))
axs[1,3].imshow(ndimage.rotate(atlas[:,:,slice_no], -90)-ndimage.rotate(images_case4[0,0,:,:,slice_no], -90), cmap = 'RdBu', vmin = -np.std(ndimage.rotate(atlas[:,:,slice_no], -90)), vmax = np.std(ndimage.rotate(images_case4[0,0,:,:,slice_no], -90)))


plt.rcParams["font.family"] = "Times New Roman"
axs[0,0].set_title("$Moving-Reference$", fontsize= fsize, pad=padd)
axs[0,1].set_title('$VM1-Reference$', fontsize= fsize, pad=padd)
axs[0,2].set_title('$VM2-Reference$', fontsize= fsize, pad=padd)
axs[0,3].set_title('$Vit-V-Net-Reference$', fontsize= fsize, pad=padd)
axs[1,0].set_title('$nnFormer-Reference$', fontsize= fsize, pad=padd)
axs[1,1].set_title("$CoTr-Reference$", fontsize= fsize, pad=padd)
axs[1,2].set_title('$TransMorph-Reference$', fontsize= fsize, pad=padd)
axs[1,3].set_title('$Ours-Reference$', fontsize= fsize, pad=padd)


axs[0,0].axis('off')
axs[0,1].axis('off')
axs[0,2].axis('off')
axs[0,3].axis('off')
axs[1,0].axis('off')
axs[1,1].axis('off')
axs[1,2].axis('off')
axs[1,3].axis('off')


plt.savefig('plots/Result_residual_plot_OASIS2', dpi=800)

#%% displacement distribution

n,bins,patchs = plt.hist(np.abs(flow[:,:,:,:,:]).flatten(), bins=30,  histtype=  'stepfilled', range = (0, 16))
plt.savefig('plots/alldisplacement_plot',dpi=600)

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()

flow_model = flow[0,:,:,:,:]
vol_size=(160,224,192)
xx = np.arange(vol_size[1])
yy = np.arange(vol_size[0])
zz = np.arange(vol_size[2])
grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
flow_model = np.rollaxis(flow_model, 0,4)
flow_model = np.rollaxis(flow_model, 2,1)
sample = flow_model+grid


fig, ax = plt.subplots(figsize=(25,25))
plot_grid(sample[:,95,:,2],sample[:,95,:,1], ax=ax,  color="C0")
plt.savefig('plots/flow.png', dpi=600)

#%%
img = images_case4[0,0,:,:,:]

fig, axs = plt.subplots(1, 3, figsize = (20, 10))

axs[0].imshow(ndimage.rotate(moving[0,:,:,slice_no], -90), cmap = 'gray')
axs[1].imshow(ndimage.rotate(atlas[0,:,:,slice_no], -90), cmap = 'gray')
axs[2].imshow(ndimage.rotate(img[:,:,slice_no], -90), cmap = 'gray')
       

axs[0].set_title("$Moving$", fontsize= fsize, pad=10)
axs[1].set_title('$Fixed$', fontsize= fsize, pad=10)
axs[2].set_title('$VM \ \ 1$', fontsize= fsize, pad=10)
#%% creating overlay

slice_no = 110
fsize = 20


plt.imshow(ndimage.rotate(atlas[0,:,:,slice_no], -90), cmap = 'binary')
# plt.imshow(ndimage.rotate(moving[0,:,:,slice_no], -90), cmap = 'summer',  alpha=0.8)

plt.imshow(ndimage.rotate(images_case4[:,:,slice_no], -90), cmap = 'jet',  alpha=0.2)

plt.savefig('plots/IXI_Atlas_ours2', dpi=600)


#%% two options for overlay
slice_no = 110
fsize = 20

fig, axs = plt.subplots(2, 2, figsize = (8, 10))

axs[0,0].imshow(ndimage.rotate(atlas[0,:,:,slice_no], -90), cmap = 'binary')
axs[0,0].imshow(ndimage.rotate(moving[0,:,:,slice_no], -90), cmap = 'jet',  alpha=0.2)

axs[0,1].imshow(ndimage.rotate(atlas[0,:,:,slice_no], -90), cmap = 'binary')
axs[0,1].imshow(ndimage.rotate(images_case4[:,:,slice_no], -90), cmap = 'jet',  alpha=0.2)

axs[1,0].imshow(ndimage.rotate(atlas[0,:,:,slice_no], -90), cmap = 'autumn')
axs[1,0].imshow(ndimage.rotate(moving[0,:,:,slice_no], -90), cmap = 'summer',  alpha=0.8)

axs[1,1].imshow(ndimage.rotate(atlas[0,:,:,slice_no], -90), cmap = 'autumn')
axs[1,1].imshow(ndimage.rotate(images_case4[:,:,slice_no], -90), cmap = 'summer',  alpha=0.8)

plt.savefig('plots/overrelay_options', dpi=600)

#%%
slice_no = 110
fsize = 20

fig, axs = plt.subplots(2, 4, figsize = (20, 10))

axs[0,0].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'binary')
axs[0,0].imshow(ndimage.rotate(moving[:,:,slice_no], -90), cmap = 'jet',  alpha=0.5)

axs[0,1].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'binary')
axs[0,1].imshow(ndimage.rotate(images_vm1[0,0,:,:,slice_no], -90), cmap = 'jet',  alpha=0.5)

axs[0,2].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'binary')
axs[0,2].imshow(ndimage.rotate(images_vm2[0,0,:,:,slice_no], -90), cmap = 'jet',  alpha=0.5)

axs[0,3].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'binary')
axs[0,3].imshow(ndimage.rotate(images_vit[0,0,:,:,slice_no], -90), cmap = 'jet',  alpha=0.5)



axs[1,0].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'binary')
axs[1,0].imshow(ndimage.rotate(images_nnf[0,0,:,:,slice_no], -90), cmap = 'jet',  alpha=0.5)

axs[1,1].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'binary')
axs[1,1].imshow(ndimage.rotate(images_cotr[0,0,:,:,slice_no], -90), cmap = 'jet',  alpha=0.5)

axs[1,2].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'binary')
axs[1,2].imshow(ndimage.rotate(images_tm[0,0,:,:,slice_no], -90), cmap = 'jet',  alpha=0.5)

axs[1,3].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'binary')
axs[1,3].imshow(ndimage.rotate(images_case4[0,0,:,:,slice_no], -90), cmap = 'jet',  alpha=0.5)




axs[0,0].set_title("$Moving$", fontsize= fsize, pad=10)
axs[0,1].set_title('$VM \ \ 1$', fontsize= fsize, pad=10)
axs[0,2].set_title('$VM \ \ 2$', fontsize= fsize, pad=10)
axs[0,3].set_title('$Vit-V-Net$', fontsize= fsize, pad=10)
axs[1,0].set_title('$nnformer$', fontsize= fsize, pad=10)
axs[1,1].set_title('$CoTr$', fontsize= fsize, pad=10)
axs[1,2].set_title('$Transmorph$', fontsize= fsize, pad=10)
axs[1,3].set_title('$Ours$', fontsize= fsize, pad=10)

# axs[1,9].set_title('$Ours-b-spline$', fontsize= fsize, pad=10)
axs[0,0].axis('off')
axs[0,1].axis('off')
axs[0,2].axis('off')
axs[0,3].axis('off')
axs[1,0].axis('off')
axs[1,1].axis('off')
axs[1,2].axis('off')
axs[1,3].axis('off')



plt.savefig('plots/Result_plot_OASIS_test2', dpi=600)

#%%
slice_no = 110
fsize = 20



fig, axs = plt.subplots(2, 4, figsize = (20, 10))

axs[0,0].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'autumn')
axs[0,0].imshow(ndimage.rotate(moving[:,:,slice_no], -90), cmap = 'summer',  alpha=0.8)

axs[0,1].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'autumn')
axs[0,1].imshow(ndimage.rotate(images_vm1[0,0,:,:,slice_no], -90), cmap = 'summer',  alpha=0.8)

axs[0,2].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'autumn')
axs[0,2].imshow(ndimage.rotate(images_vm2[0,0,:,:,slice_no], -90), cmap = 'summer',  alpha=0.8)

axs[0,3].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'autumn')
axs[0,3].imshow(ndimage.rotate(images_vit[0,0,:,:,slice_no], -90), cmap = 'summer',  alpha=0.8)



axs[1,0].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'autumn')
axs[1,0].imshow(ndimage.rotate(images_nnf[0,0,:,:,slice_no], -90), cmap = 'summer',  alpha=0.8)

axs[1,1].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'autumn')
axs[1,1].imshow(ndimage.rotate(images_cotr[0,0,:,:,slice_no], -90), cmap = 'summer',  alpha=0.8)

axs[1,2].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'autumn')
axs[1,2].imshow(ndimage.rotate(images_tm[0,0,:,:,slice_no], -90), cmap = 'summer',  alpha=0.8)

axs[1,3].imshow(ndimage.rotate(atlas[:,:,slice_no], -90), cmap = 'autumn')
axs[1,3].imshow(ndimage.rotate(images_case4[0,0,:,:,slice_no], -90), cmap = 'summer',  alpha=0.8)




axs[0,0].set_title("$Moving$", fontsize= fsize, pad=10)
axs[0,1].set_title('$VM \ \ 1$', fontsize= fsize, pad=10)
axs[0,2].set_title('$VM \ \ 2$', fontsize= fsize, pad=10)
axs[0,3].set_title('$Vit-V-Net$', fontsize= fsize, pad=10)
axs[1,0].set_title('$nnformer$', fontsize= fsize, pad=10)
axs[1,1].set_title('$CoTr$', fontsize= fsize, pad=10)
axs[1,2].set_title('$Transmorph$', fontsize= fsize, pad=10)
axs[1,3].set_title('$Ours$', fontsize= fsize, pad=10)

# axs[1,9].set_title('$Ours-b-spline$', fontsize= fsize, pad=10)
axs[0,0].axis('off')
axs[0,1].axis('off')
axs[0,2].axis('off')
axs[0,3].axis('off')
axs[1,0].axis('off')
axs[1,1].axis('off')
axs[1,2].axis('off')
axs[1,3].axis('off')



plt.savefig('plots/Result_plot_IXItestsummer', dpi=600)