import h5py
import numpy as np


nhalos = 207

f = h5py.File('halo_0.hdf5','r')
cols = list(f.attrs.keys())

head = ''
for col in cols[:-1]:
    head += col+','
head += cols[-1]


props = np.zeros((nhalos,len(cols)))

for j in range(nhalos):
    f = h5py.File('halo_'+str(j)+'.hdf5','r')
    for i in range(len(cols)):
        props[j,i] = f.attrs[cols[i]]
    
out_file = '/home/elizabeth/SIDM/halo_props/halo_props_rock_sidm1.csv.bz2'

np.savetxt(out_file,props,fmt='%12.6f',header=head,comments='',delimiter=',')
