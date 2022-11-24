import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import matplotlib.pyplot as plt
from stacked import stack_halos
from stacked import stack_profile
from stacked import fit_profiles
import pandas as pd
from fit_models_colossus import *
from models_profiles import *
from colossus.halo import concentration
from colossus.cosmology import cosmology  
from colossus.halo import mass_defs
params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
cosmology.addCosmology('MICE', params)
cosmo = cosmology.setCosmology('MICE')

z = 0.

# READ catalogues from halo parameters
rock       = pd.read_csv('/mnt/projects/lensing/SIDM_project/halo_props/halo_props_match_cdm_z0_rock2.csv.bz2')
rock1      = pd.read_csv('/mnt/projects/lensing/SIDM_project/halo_props/halo_props_match_sidm1_z0_rock2.csv.bz2')

# FOLDERS WHERE PARTICLES ARE SAVED
path = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/rockstar/matcheados/CDM/'
path1 = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/rockstar/matcheados/SIDM1/'

# READ halos computed profperties
main_file1 = '/mnt/projects/lensing/SIDM_project/halo_props/halo_propsv2_rock2_match_sidm1_z0_main.csv.bz2'
main_file = '/mnt/projects/lensing/SIDM_project/halo_props/halo_propsv2_rock2_match_cdm_z0_main.csv.bz2'
main  = pd.read_csv(main_file)
main1 = pd.read_csv(main_file1)


# COMPUTE SHAPES AND ENERGY RATIO
S_it  = main.c3D_it/main.a3D_it
S1_it = main1.c3D_it/main1.a3D_it

S_itr  = main.c3Dr_it/main.a3Dr_it
S1_itr = main1.c3Dr_it/main1.a3Dr_it

S  = main.c3D_it/main.a3D_it
S1 = main1.c3D_it/main1.a3D_it

S_r  = main.c3Dr/main.a3Dr
S1_r = main1.c3Dr/main1.a3Dr

q2d = main.b2D/main.a2D
q2d1 = main1.b2D/main1.a2D

q2dr = main.b2Dr/main.a2Dr
q2dr1 = main1.b2Dr/main1.a2Dr

q2d_it  = main.b2D_it/main.a2D_it
q2d1_it  = main1.b2D_it/main1.a2D_it

q2dr_it = main.b2Dr_it/main.a2Dr_it
q2dr1_it = main1.b2Dr_it/main1.a2Dr_it


Eratio  = (2.*main.EKin/abs(main.EPot))
Eratio1 = (2.*main1.EKin/abs(main1.EPot))


# SELECT AN HALO SAMPLE
sname = 'subset'
m = (S_itr-S1_itr)/S_itr < -0.1

# sname = 'total'
# m = S_itr < 100.

haloids  = np.array(main.column_halo_id)[m]
haloids1 = np.array(main1.column_halo_id)[m]
nhalos = len(haloids)

# ROTATE, STACK AND PROJECT PARTICLES    
x,y,z,x2d,y2d      = stack_halos(main_file,path,haloids,True)   
x1,y1,z1,x2d1,y2d1 = stack_halos(main_file1,path1,haloids1,True)   

# SELECT ONLY PARTICLES WITHIN 4Mpc
m3d = (abs(x) < 4.)*(abs(y) < 4.)*(abs(z) < 4.)
m2d = (abs(x2d) < 4.)*(abs(y2d) < 4.)

m3d1 = (abs(x1) < 4.)*(abs(y1) < 4.)*(abs(z1) < 4.)
m2d1 = (abs(x2d1) < 4.)*(abs(y2d1) < 4.)

X,Y,Z = x[m3d]*1.e3,y[m3d]*1.e3,z[m3d]*1.e3 # 3D coordinates in kpc
Xp,Yp = x2d[m2d]*1.e3,y2d[m2d]*1.e3 # 2D coordinates in kpc

X1,Y1,Z1 = x1[m3d1]*1.e3,y1[m3d1]*1.e3,z1[m3d1]*1.e3
Xp1,Yp1  = x2d1[m2d1]*1.e3,y2d1[m2d1]*1.e3

# COMPUTE PROFILES USING PARTICLES
p_DM    = stack_profile(X,Y,Z,Xp,Yp,nhalos)
p_SIDM  = stack_profile(X1,Y1,Z1,Xp1,Yp1,nhalos)

# COMPUTE AND FIT PROFILES USING MAPS
pm_DM    = fit_profiles(Xp,Yp,nhalos)
pm_SIDM  = fit_profiles(Xp1,Yp1,nhalos)


# M200c = 10**13.6
# c200c = concentration.concentration(M200c, '200c', z, model = 'diemer19')


plt.figure()
plt.plot(pm_SIDM.r,pm_SIDM.GX,'C3',label='SIDM')
plt.plot(pm_SIDM.r,pm_SIDM.GX_fit,'C3',alpha=0.5)
plt.plot(pm_SIDM.r,pm_SIDM.GX_fit2,'C3--',alpha=0.5)
plt.plot(pm_DM.r,pm_DM.GX,'k',label='CDM')
plt.plot(pm_DM.r,pm_DM.GX_fit,'k',alpha=0.5,label='fit separately')
plt.plot(pm_DM.r,pm_DM.GX_fit2,'k--',alpha=0.5,label='fit simultaneously')
plt.xscale('log')
plt.xlabel('$R [Mpc]$')
plt.ylabel(r'$\epsilon \times \Gamma_X [M_\odot/pc^2]$')
plt.legend()
plt.savefig('../profile_GX_'+sname+'.png')

plt.figure()
plt.plot(pm_SIDM.r,pm_SIDM.GT,'C3',label='SIDM')
plt.plot(pm_SIDM.r,pm_SIDM.GT_fit,'C3',alpha=0.5)
plt.plot(pm_SIDM.r,pm_SIDM.GT_fit2,'C3--',alpha=0.5)
plt.plot(pm_DM.r,pm_DM.GT,'k',label='CDM')
plt.plot(pm_DM.r,pm_DM.GT_fit,'k',alpha=0.5,label='fit separately')
plt.plot(pm_DM.r,pm_DM.GT_fit2,'k--',alpha=0.5,label='fit simultaneously')
plt.loglog()
plt.xlabel('$R [Mpc]$')
plt.ylabel(r'$\epsilon \times \Gamma_T [M_\odot/pc^2]$')
plt.legend()
plt.savefig('../profile_GT_'+sname+'.png')

plt.figure()
plt.plot(pm_SIDM.r,pm_SIDM.S2,'C3',label='SIDM')
plt.plot(pm_SIDM.r,pm_SIDM.S2_fit,'C3',alpha=0.5)
plt.plot(pm_DM.r,pm_DM.S2,'k',label='CDM')
plt.plot(pm_DM.r,pm_DM.S2_fit,'k',alpha=0.5)
plt.loglog()
plt.xlabel('$R [Mpc]$')
plt.ylabel(r'$\epsilon \times \Sigma_2 [M_\odot/pc^2]$')
plt.legend()
plt.savefig('../profile_S2_'+sname+'.png')


# q2d distributions for DM
plt.figure()
plt.title('DM')
plt.hist(q2d_it[m],np.linspace(0.35,1,20),histtype='step',label='std - it',lw=3)
plt.hist(q2dr_it[m],np.linspace(0.35,1,20),histtype='step',label='red - it',lw=3)
plt.axvline(np.mean(q2d_it[m]),color='C0',lw=3,ls='--')
plt.axvline(np.mean(q2dr_it[m]),color='C1',lw=3,ls='--')
plt.axvline(pm_DM.q_s,label='fit S',color='C2',lw=3)
plt.axvline(pm_DM.q_2g,label='fit G',color='C3',lw=3)
plt.axvline(pm_DM.q_gt,label='fit GT',color='C4',lw=3)
plt.axvline(pm_DM.q_gx,label='fit GX',color='C5',lw=3)
plt.legend(loc=2,frameon=False)
plt.xlabel('q2d')
plt.savefig('../profile_q2d_cdm_it_'+sname+'.png')

plt.figure()
plt.title('DM')
plt.hist(q2d[m],np.linspace(0.35,1,20),histtype='step',label='std',lw=3)
plt.hist(q2dr[m],np.linspace(0.35,1,20),histtype='step',label='red',lw=3)
plt.axvline(np.mean(q2d[m]),color='C0',lw=3,ls='--')
plt.axvline(np.mean(q2dr[m]),color='C1',lw=3,ls='--')
plt.axvline(pm_DM.q_s,label='fit S',color='C2',lw=3)
plt.axvline(pm_DM.q_2g,label='fit G',color='C3',lw=3)
plt.axvline(pm_DM.q_gt,label='fit GT',color='C4',lw=3)
plt.axvline(pm_DM.q_gx,label='fit GX',color='C5',lw=3)
plt.legend(loc=2,frameon=False)
plt.xlabel('q2d')
plt.savefig('../profile_q2d_cdm_'+sname+'.png')


# q2d distributions for SIDM
plt.figure()
plt.title('SIDM')
plt.hist(q2d1_it[m],np.linspace(0.35,1,20),histtype='step',label='std - it',lw=3)
plt.hist(q2dr1_it[m],np.linspace(0.35,1,20),histtype='step',label='red - it',lw=3)
plt.axvline(np.mean(q2d1_it[m]),color='C0',lw=3,ls='--')
plt.axvline(np.mean(q2dr1_it[m]),color='C1',lw=3,ls='--')
plt.axvline(pm_SIDM.q_s,label='fit S',color='C2',lw=3)
plt.axvline(pm_SIDM.q_2g,label='fit G',color='C3',lw=3)
plt.axvline(pm_SIDM.q_gt,label='fit GT',color='C4',lw=3)
plt.axvline(pm_SIDM.q_gx,label='fit GX',color='C5',lw=3)
plt.legend(loc=2,frameon=False)
plt.xlabel('q2d')
plt.savefig('../profile_q2d_sidm_it_'+sname+'.png')

plt.figure()
plt.title('SIDM')
plt.hist(q2d1[m],np.linspace(0.35,1,20),histtype='step',label='std',lw=3)
plt.hist(q2dr1[m],np.linspace(0.35,1,20),histtype='step',label='red',lw=3)
plt.axvline(np.mean(q2d1[m]),color='C0',lw=3,ls='--')
plt.axvline(np.mean(q2dr1[m]),color='C1',lw=3,ls='--')
plt.axvline(pm_SIDM.q_s,label='fit S',color='C2',lw=3)
plt.axvline(pm_SIDM.q_2g,label='fit G',color='C3',lw=3)
plt.axvline(pm_SIDM.q_gt,label='fit GT',color='C4',lw=3)
plt.axvline(pm_SIDM.q_gx,label='fit GX',color='C5',lw=3)
plt.legend(loc=2,frameon=False)
plt.xlabel('q2d')
plt.savefig('../profile_q2d_cdm_'+sname+'.png')

