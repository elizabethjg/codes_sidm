import h5py
import numpy as np


nhalos = 207
# nhalos = 143
nhalos = 75
nhalos = 258
nhalos = 249

# path = '/mnt/simulations/SIDM_simus/Lentes/V2/SIDM1/'
# path = '/mnt/simulations/SIDM_simus/Lentes/V2/SIDM1/Snap43/'
# path = '/mnt/simulations/SIDM_simus/Lentes/V2/SIDM1/Snap39/'
# path = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/rockstar/SIDM1/'
path = '/mnt/projects/lensing/SIDM_project/Lentes/Subhalos/CDM/'

f = h5py.File(path+'halo_0.hdf5','r')
cols = list(f.attrs.keys())

head = ''
for col in cols:
    head += col+','
head += 'N_sh'


props = np.zeros((nhalos,len(cols)+1))

for j in range(nhalos):
    f = h5py.File(path+'halo_'+str(j)+'.hdf5','r')
    for i in range(len(cols)):
        props[j,i] = f.attrs[cols[i]]
    props[j,i+1] = len(np.array(f['X']))
    
# out_file = '/home/elizabeth/SIDM/halo_props/halo_props_sidm1_z0_rock.csv.bz2'
# out_file = '/home/elizabeth/SIDM/halo_props/halo_props_sidm1_z51_rock.csv.bz2'
out_file = '/home/elizabeth/SIDM/halo_props/halo_props_sh_cdm_z0_rock2.csv.bz2'

np.savetxt(out_file,props,fmt='%12.6f',header=head,comments='',delimiter=',')
