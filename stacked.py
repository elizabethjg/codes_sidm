import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
# halos = fits.open('/home/elizabeth/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/HS-lensing/HALO_Props_MICE.fits')[1].data        

zs = ['z0','z51','z96']


z = zs[0]

# path = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/CDM/'
# main = pd.read_csv('/home/elizabeth/SIDM/halo_props/halo_props_fof_cdm_'+z+'_main.csv.bz2')
path = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/SCDM/'
main = pd.read_csv('/home/elizabeth/SIDM/halo_props/halo_props_fof_sidm1_'+z+'_main.csv.bz2')

x = np.array([])
y = np.array([])
z = np.array([])

nhalos = 2

for j in range(nhalos):
    
    halo = h5py.File(path+'halo_'+str(j)+'.hdf5','r')       
    
    X = np.array(halo['X']) - main.xc_rc[j]/1.e3
    Y = np.array(halo['Y']) - main.yc_rc[j]/1.e3
    Z = np.array(halo['Z']) - main.zc_rc[j]/1.e3
    
    xrot = (main.a3Dx[j]*X)+(main.a3Dy[j]*Y)+(main.a3Dz[j]*Z);
    yrot = (main.b3Dx[j]*X)+(main.b3Dy[j]*Y)+(main.b3Dz[j]*Z);
    zrot = (main.c3Dx[j]*X)+(main.c3Dy[j]*Y)+(main.c3Dz[j]*Z);
    
    x = np.append(x,xrot)
    y = np.append(y,yrot)
    z = np.append(z,zrot)

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
