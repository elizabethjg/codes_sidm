import h5py
import numpy as np


nhalos = 207
# nhalos = 143
nhalos = 75
nhalos = 258

# path = '/mnt/simulations/SIDM_simus/Lentes/V2/SIDM1/'
# path = '/mnt/simulations/SIDM_simus/Lentes/V2/SIDM1/Snap43/'
# path = '/mnt/simulations/SIDM_simus/Lentes/V2/SIDM1/Snap39/'
path = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/rockstar/SIDM1/'

f = h5py.File(path+'halo_0.hdf5','r')
cols = list(f.attrs.keys())

head = ''
for col in cols[:-1]:
    head += col+','
head += cols[-1]


props = np.zeros((nhalos,len(cols)))

for j in range(nhalos):
    f = h5py.File(path+'halo_'+str(j)+'.hdf5','r')
    for i in range(len(cols)):
        props[j,i] = f.attrs[cols[i]]
    
# out_file = '/home/elizabeth/SIDM/halo_props/halo_props_sidm1_z0_rock.csv.bz2'
# out_file = '/home/elizabeth/SIDM/halo_props/halo_props_sidm1_z51_rock.csv.bz2'
out_file = '/home/elizabeth/SIDM/halo_props/halo_props_sidm1_z0_rock2.csv.bz2'

np.savetxt(out_file,props,fmt='%12.6f',header=head,comments='',delimiter=',')
