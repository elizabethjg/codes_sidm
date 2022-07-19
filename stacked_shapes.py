import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import matplotlib.pyplot as plt
from stacked import stack_halos
import pandas as pd
from member_distribution import compute_axis
zs = ['z0','z51','z96']


z = zs[0]

# path = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/SCDM/'
# main_file = '/home/elizabeth/SIDM/halo_props/halo_props_fof_sidm1_'+z+'_main.csv.bz2'
path = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/CDM/'
main_file = '/home/elizabeth/SIDM/halo_props/halo_props_fof_cdm_'+z+'_main.csv.bz2'


main = pd.read_csv(main_file)

rc  = np.array(np.sqrt((main.xc - main.xc_rc)**2 + (main.yc - main.yc_rc)**2 + (main.zc - main.zc_rc)**2))
offset  = rc/main.r_max


nhalos = len(main)

haloids = np.array(main.column_halo_id)[offset < 0.1]
    
x,y,z,x2d,y2d = stack_halos(main_file,path,haloids) 

m3d = (abs(x) < 4.)*(abs(y) < 4.)*(abs(z) < 4.)
m2d = (abs(x2d) < 4.)*(abs(y2d) < 4.)

r3d2 = x**2+ y**2 + z**2
r2d2 = x2d**2+ y2d**2 

v3d,w3d,v2d,w2d = compute_axis(x[m3d],y[m3d],z[m3d],x2d[m2d],y2d[m2d])
