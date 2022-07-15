import h5py
import numpy as np


# path = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/SCDM/'
path = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/CDM/'
# path = '/home/elizabeth/CDM/'

atri = h5py.File(path+'all_halos.hdf5','r')

cols = list(atri.keys())
nhalos = len(atri[cols[0]])

head = ''
for col in cols[:-1]:
    head += col+','
head += cols[-1]

props = np.zeros((nhalos,len(cols)))

for j in range(nhalos):
    
    data = h5py.File(path+'halo_'+str(j)+'.hdf5','a')       
    
    for i in range(len(cols)):
        if cols[i] == 'Mvir':
            data.attrs.create(cols[i],10**atri[cols[i]][j])
            props[j,i] = data.attrs[cols[i]]
        else:
            data.attrs.create(cols[i],atri[cols[i]][j])
            props[j,i] = data.attrs[cols[i]]
    data.close()
        
    
# out_file = '/home/elizabeth/SIDM/halo_props/halo_props_sidm1_z0_fof.csv.bz2'
out_file = '/home/elizabeth/SIDM/halo_props/halo_props_cdm_z0_fof.csv.bz2'

np.savetxt(out_file,props,fmt='%12.6f',header=head,comments='',delimiter=',')
