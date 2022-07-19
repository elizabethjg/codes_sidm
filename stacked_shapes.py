import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import matplotlib.pyplot as plt
from stacked import stack_halos
import pandas as pd
from member_distribution import compute_axis
zs = ['z0','z51','z96']


z = zs[0]

path_sidm = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/SCDM/'
main_sidm = '/home/elizabeth/SIDM/halo_props/halo_props_fof_sidm1_'+z+'_main.csv.bz2'
path_cdm = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/CDM/'
main_cdm = '/home/elizabeth/SIDM/halo_props/halo_props_fof_cdm_'+z+'_main.csv.bz2'


main = pd.read_csv(main_cdm)
main1 = pd.read_csv(main_sidm)

rc  = np.array(np.sqrt((main.xc - main.xc_rc)**2 + (main.yc - main.yc_rc)**2 + (main.zc - main.zc_rc)**2))
offset  = rc/main.r_max

rc1  = np.array(np.sqrt((main1.xc - main1.xc_rc)**2 + (main1.yc - main1.yc_rc)**2 + (main1.zc - main1.zc_rc)**2))
offset1  = rc1/main1.r_max

mhalos = (offset < 0.1)*(main.lgM < 14.2)
mhalos1 = (offset1 < 0.1)*(main1.lgM < 14.2)

haloids = np.array(main.column_halo_id)[mhalos]
haloids1 = np.array(main1.column_halo_id)[mhalos1]
    
x,y,z,x2d,y2d = stack_halos(main_cdm,path_cdm,haloids,True) 
x1,y1,z1,x2d1,y2d1 = stack_halos(main_sidm,path_sidm,haloids1,True) 


m3d = (abs(x) < 4.)*(abs(y) < 4.)*(abs(z) < 4.)
m2d = (abs(x2d) < 4.)*(abs(y2d) < 4.)

r3d2 = x**2+ y**2 + z**2
r2d2 = x2d**2+ y2d**2 

v3d,w3d,v2d,w2d = compute_axis(x[m3d],y[m3d],z[m3d],x2d[m2d],y2d[m2d])
v3dr,w3dr,v2dr,w2dr = compute_axis(x[m3d],y[m3d],z[m3d],x2d[m2d],y2d[m2d],(1./r3d2)[m3d],(1./r2d2)[m2d])

m3d1 = (abs(x1) < 4.)*(abs(y1) < 4.)*(abs(z1) < 4.)
m2d1 = (abs(x2d1) < 4.)*(abs(y2d1) < 4.)

r3d21 = x1**2+ y1**2 + z1**2
r2d21 = x2d1**2+ y2d1**2 

v3d1,w3d1,v2d1,w2d1 = compute_axis(x1[m3d1],y1[m3d1],z1[m3d1],x2d1[m2d1],y2d1[m2d1])
v3dr1,w3dr1,v2dr1,w2dr1 = compute_axis(x1[m3d1],y1[m3d1],z1[m3d1],x2d1[m2d1],y2d1[m2d1],(1./r3d21)[m3d1],(1./r2d21)[m2d1])
