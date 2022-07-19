import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import matplotlib.pyplot as plt
from stacked import stack_halos

zs = ['z0','z51','z96']


z = zs[0]

path = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/SCDM/'
mainfile = '/home/elizabeth/SIDM/halo_props/halo_props_fof_sidm1_'+z+'_main.csv.bz2'


nhalos = len(main)

haloids = range(nhalos)
    
x,y,z,x2d,y2d = stack_halos(mainfile,path,haloids)   
    

f, ax = plt.subplots(1,3, figsize=(12.7,8))
ax[0].plot(x,y,'C7,')
ax[1].plot(x,z,'C7,')
ax[2].plot(y,z,'C7,')
ax[0].plot(x,y,'C2.',alpha=0.005)
ax[1].plot(x,z,'C2.',alpha=0.005)
ax[2].plot(y,z,'C2.',alpha=0.005)

ax[0].set_xlim([-4.01,4.01])
ax[0].set_ylim([-4.01,4.01])
ax[1].set_xlim([-4.01,4.01])
ax[1].set_ylim([-4.01,4.01])
ax[2].set_xlim([-4.01,4.01])
ax[2].set_ylim([-4.01,4.01])


ax[0].set_xlabel('x [Mpc/h]')
ax[0].set_ylabel('y [Mpc/h]')

ax[1].set_xlabel('x [Mpc/h]')
ax[1].set_ylabel('z [Mpc/h]')

ax[2].set_xlabel('y [Mpc/h]')
ax[2].set_ylabel('z [Mpc/h]')

f.savefig('/home/elizabeth/SIDM/coords_sidm1.png',bbox_inches='tight')
