import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import matplotlib.pyplot as plt
from stacked import stack_halos
import pandas as pd
from member_distribution import compute_axis
zs = ['z0','z51','z96']


z = zs[0]

# path_sidm = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/SCDM/'
# main_sidm = '/home/elizabeth/SIDM/halo_props/halo_props_fof_sidm1_'+z+'_main.csv.bz2'
# path_cdm = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/CDM/'
# main_cdm = '/home/elizabeth/SIDM/halo_props/halo_props_fof_cdm_'+z+'_main.csv.bz2'

path_sidm = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/rockstar/SIDM1/'
main_sidm = '/home/elizabeth/SIDM/halo_props/halo_props_rock2_sidm1_z0_main.csv.bz2'
path_cdm = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/rockstar/CDM/'
main_cdm = '/home/elizabeth/SIDM/halo_props/halo_props_rock2_cdm_z0_main.csv.bz2'


main = pd.read_csv(main_cdm)
main1 = pd.read_csv(main_sidm)

S = main.c3D/main.a3D
Q = main.c3D/main.a3D
q = main.b2D/main.a2D
S1 = main1.c3D/main1.a3D
Q1 = main1.c3D/main1.a3D
q1 = main1.b2D/main1.a2D

rc  = np.array(np.sqrt((main.xc - main.xc_rc)**2 + (main.yc - main.yc_rc)**2 + (main.zc - main.zc_rc)**2))
offset  = rc/main.r_max

rc1  = np.array(np.sqrt((main1.xc - main1.xc_rc)**2 + (main1.yc - main1.yc_rc)**2 + (main1.zc - main1.zc_rc)**2))
offset1  = rc1/main1.r_max

mhalos = (offset < 0.1)*(main.lgM > 14)#*(S>0.7)
mhalos1 = (offset1 < 0.1)*(main1.lgM > 14)#*(S1>0.7)

haloids = np.array(main.column_halo_id)[mhalos]
haloids1 = np.array(main1.column_halo_id)[mhalos1]
    
x,y,z,x2d,y2d = stack_halos(main_cdm,path_cdm,haloids,True) 
x1,y1,z1,x2d1,y2d1 = stack_halos(main_sidm,path_sidm,haloids1,True) 


m3d = (abs(x) < 2.)*(abs(y) < 2.)*(abs(z) < 2.)
m2d = (abs(x2d) < 2.)*(abs(y2d) < 2.)

r3d2 = x**2+ y**2 + z**2
r2d2 = x2d**2+ y2d**2 

v3d,w3d,v2d,w2d = compute_axis(x[m3d],y[m3d],z[m3d],x2d[m2d],y2d[m2d])
v3dr,w3dr,v2dr,w2dr = compute_axis(x[m3d],y[m3d],z[m3d],x2d[m2d],y2d[m2d],(1./r3d2)[m3d],(1./r2d2)[m2d])

S_dm = np.sqrt(w3d[2])/np.sqrt(w3d[0])
q_dm = np.sqrt(w2d[1])/np.sqrt(w2d[0])

m3d1 = (abs(x1) < 2.)*(abs(y1) < 2.)*(abs(z1) < 2.)
m2d1 = (abs(x2d1) < 2.)*(abs(y2d1) < 2.)

r3d21 = x1**2+ y1**2 + z1**2
r2d21 = x2d1**2+ y2d1**2 

v3d1,w3d1,v2d1,w2d1 = compute_axis(x1[m3d1],y1[m3d1],z1[m3d1],x2d1[m2d1],y2d1[m2d1])
v3dr1,w3dr1,v2dr1,w2dr1 = compute_axis(x1[m3d1],y1[m3d1],z1[m3d1],x2d1[m2d1],y2d1[m2d1],(1./r3d21)[m3d1],(1./r2d21)[m2d1])

S_1 = np.sqrt(w3d1[2])/np.sqrt(w3d1[0])
q_1 = np.sqrt(w2d1[1])/np.sqrt(w2d1[0])

print('S - CDM')
print(S_dm)
print('q - CDM')
print(q_dm)

print('S - SIDM1')
print(S_1)
print('q - SIDM')
print(q_1)

print('Sratio',S_dm/S_1)
print('qratio',q_dm/q_1)

plt.hist(S[mhalos],np.linspace(0.,1.,25),histtype='step',label='CDM',color='C2')
plt.hist(S1[mhalos1],np.linspace(0.,1.,25),histtype='step',label='SIDM',color='C3')
plt.axvline(S_dm,color='C2')
plt.axvline(S_1,color='C3')
plt.legend()
plt.savefig('../Sdist_rock2_lgMM14_off1.png')

