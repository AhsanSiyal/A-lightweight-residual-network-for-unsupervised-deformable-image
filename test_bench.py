# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 09:12:39 2022

@author: Ahsan
"""

'''
writing script for test bench of 5 methods and their comparision. 
following matrices will be compaired:
Dice, MI, number of Positive jacobians, histogram of displacment, trainable paramters, loss function landscape.
'''

#%%
'''
loading libraries
'''

import torch
# from models import VxmDense_1
import os, losses, utils, utils1
from torch.utils.data import DataLoader
from data import datasets, trans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import transforms
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
# from arch.case4 import ResUnet as ResNet
import numpy as np
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph

# from models1 import VxmDense_1, VxmDense_2, VxmDense_huge
print('\n\n Libraries are loaded')

#%%
'''
GPU configuration
'''
GPU_iden = 0
GPU_num = torch.cuda.device_count()
print('Number of GPU: ' + str(GPU_num))
for GPU_idx in range(GPU_num):
    GPU_name = torch.cuda.get_device_name(GPU_idx)
    print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
torch.cuda.set_device(GPU_iden)
GPU_avai = torch.cuda.is_available()
print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
print('If the GPU is available? ' + str(GPU_avai))

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

print('\n\n GPU configured')

#%%
'''
loading the model
'''

'''
MODEL 1: 
    Discription: 
'''
img_size = (160, 192, 224)
# try:
#     # model_1 =  VxmDense_1(img_size)
#     model_1 =  ResNet(img_size,2)
#     best_model1 = torch.load('result_models/case_4/case4_2/dsc0.7997.pth.tar')['state_dict']
#     model_1.load_state_dict(best_model1)
#     model_1.cuda()
#     print('\n\n Model 1 has loaded')
# except Exception as err:
#     print(f"\n\n Oops! Model 1 could not be loaded: {err}, {type(err)}")

# '''
# MODEL 2: 
#     discription: 
# '''
config = CONFIGS_TM['TransMorph-Large']
model_1 = TransMorph.TransMorph(config)
best_model1 = torch.load('reg_paper_model/reg_AD3_1/dsc0.7588.pth.tar')['state_dict']
model_1.load_state_dict(best_model1)
model_1.cuda()



print('\n\n Model 1 has loaded')    
  

'''
    Initialize spatial transformation function
'''
reg_model = utils.register_model(img_size, 'nearest')
reg_model.cuda()
reg_model_bilin = utils.register_model(img_size, 'bilinear')
reg_model_bilin.cuda()



#%%
'''
loading the test dataset
'''

'''
Model 1
'''
val_dir = 'C:/Drive/Workspace/data_sets/OASIS_dataset/test/'
atlas_dir = 'C:/Drive/Workspace/data_sets/IXI_data/atlas.pkl'


val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
val_set = datasets.OASISBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)

simf = losses.SSIM3D()
lmi_f = losses.localMutualInformation()
eval_dsc = utils1.AverageMeter()
dice = []
jac_n = []
flow_arr = []
sim = []
lmi = []
x_seg_arr = []
x_seg_def_arr = []
y_seg_arr = []
x_arr = []
x_def_arr = []
y_arr = []
with torch.no_grad():
    for data in tqdm(val_loader):
        model_1.eval()
        data = [t.cuda() for t in data]
        x = data[0]
        y = data[1]
        x_seg = data[2]
        y_seg = data[3]
        x_in = torch.cat((x, y), dim=1)
        grid_img = mk_grid_img(8, 1, img_size)
        output = model_1(x_in)
        def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
        def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
        dsc = utils1.dice_val_VOI(def_out.long(), y_seg.long())
        x_arr.append(x.detach().cpu().numpy())
        y_arr.append(y.detach().cpu().numpy())
        x_seg_def_arr.append(def_out.detach().cpu().numpy()[0, 0, :, :, :])
        y_seg_arr.append(y_seg.detach().cpu().numpy()[0, 0, :, :, :])
        x_seg_arr.append(x_seg.detach().cpu().numpy()[0, 0, :, :, :])
        x_def_arr.append(output[0].detach().cpu().numpy())        
        eval_dsc.update(dsc.item(), x.size(0))
        jac_det = utils1.jacobian_determinant_vxm(output[1].detach().cpu().numpy()[0, :, :, :, :])
        jac_n.append(np.sum(jac_det <= 0) / np.prod((160,192,224)))
        flow_arr.append(output[1].detach().cpu().numpy()[0, :, :, :, :])
        dice.append(dsc)
        #print(eval_dsc.avg)

x_seg_arr = np.array(x_seg_arr)
y_seg_arr = np.array(y_seg_arr)
x_seg_def_arr = np.array(x_seg_def_arr)
x_arr = np.array(x_arr)
x_def_arr = np.array(x_def_arr)
y_arr = np.array(y_arr)
dice_array = np.array(dice)
jac_n = np.array(jac_n)
flow_arr = np.array(flow_arr)

mean_dice = np.mean(dice_array)
mean_jac = np.mean(jac_n)
# np.savetxt('Result_variables/R1_OASIS_case4_30_dice.csv', dice_array)
# np.savetxt('Result_variables/R1_OASIS_case4_30_jac_n.csv', jac_n)
# np.savetxt('Result_variables/R1_OASIS_case4_30_ssim_n.csv', ssim)
# np.savetxt('Result_variables/R1_OASIS_case4_30_lmi_n.csv', lmi_arr)
md = 'AD3_1'
np.save("Regularization_paper_results/" + md + "/" + md+ "_TM_x_seg_def.npy", x_seg_def_arr)
np.save("Regularization_paper_results/" + md + "/" + md+ "_TM_y_seg.npy", y_seg_arr)
np.save("Regularization_paper_results/" + md + "/" + md+ "_TM_x_seg.npy", x_seg_arr)
np.save("Regularization_paper_results/" + md + "/" + md+ "_TM_x_arr.npy", x_arr)
np.save("Regularization_paper_results/" + md + "/" + md+ "_TM_x_def_arr.npy", x_def_arr)
np.save("Regularization_paper_results/" + md + "/" + md+ "_TM_y_arr.npy", y_arr)
np.save("Regularization_paper_results/" + md + "/" + md+ "_TM_flow_arr.npy", flow_arr)
# x1 = x.detach().cpu().numpy()[0,0]
# y1 = y.detach().cpu().numpy()[0,0]
#%%
def calculate_jacobian_determinant(displacement_field):
    # Assuming displacement_field is of shape (3, H, W, D) where 3 corresponds to the x, y, z displacements
    u, v, w = displacement_field
    
    # Compute partial derivatives for Jacobian matrix
    dudx = np.gradient(u, axis=0)
    dudy = np.gradient(u, axis=1)
    dudz = np.gradient(u, axis=2)
    
    dvdx = np.gradient(v, axis=0)
    dvdy = np.gradient(v, axis=1)
    dvdz = np.gradient(v, axis=2)
    
    dwdx = np.gradient(w, axis=0)
    dwdy = np.gradient(w, axis=1)
    dwdz = np.gradient(w, axis=2)
    
    # Construct Jacobian matrix and compute determinant
    det_jacobian = (
        dudx * (dvdy * dwdz - dvdz * dwdy) -
        dudy * (dvdx * dwdz - dvdz * dwdx) +
        dudz * (dvdx * dwdy - dvdy * dwdx)
    )
    
    return det_jacobian

fig, axes = plt.subplots(10, 4, figsize=(20, 25))
axes = axes.ravel()

# Calculate and plot Jacobian determinant heatmaps
for i, displacement_field in enumerate(flow_arr):
    jacobian_determinant = calculate_jacobian_determinant(displacement_field)
    
    # Select a slice for visualization (middle slice along z-axis)
    heatmap_slice = jacobian_determinant[:, :, jacobian_determinant.shape[2] // 2]
    
    # Plot the heatmap
    ax = axes[i]
    im = ax.imshow(heatmap_slice, cmap='coolwarm', vmin=-1, vmax=1)  # Adjust vmin/vmax for better contrast
    ax.set_title(f'Field {i+1}')
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
plt.save

#%%
pt = output[0].detach().cpu().numpy()[0, :, :, :, :]
plt.imshow(x1[80])
plt.imshow(y1[80])
plt.imshow(pt[0,80])
