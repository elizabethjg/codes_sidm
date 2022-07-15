import h5py
import numpy as np


# path = '/mnt/simulations/SIDM_simus/Lentes/Eli_Agus/snapshot_050/CDM/'
path = '/home/elizabeth/CDM/'

atri = h5py.File(path+'all_halos.hdf5','r')

cols = list(atri.keys())
nhalos = len(atri[cols[0]])

head = ''
for col in cols[:-1]:
    head += col+','
head += cols[-1]

props = np.zeros((nhalos,len(cols)))

for j in range(nhalos):
    j = 0
    
    f = h5py.File(path+'halo_'+str(j)+'.hdf5','w')   
    
    for i in range(len(cols)):
        f.attrs.create(cols[i],atri[cols[i]][j])
        props[j,i] = f.attrs[cols[i]]
    f.close()
        
    
out_file = '/home/elizabeth/SIDM/halo_props/halo_props_cdm_z0_fof.csv.bz2'

np.savetxt(out_file,props,fmt='%12.6f',header=head,comments='',delimiter=',')
