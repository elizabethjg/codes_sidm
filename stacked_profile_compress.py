import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import matplotlib.pyplot as plt
from stacked import stack_halos
from stacked import stack_profile
from stacked import stack_halos_parallel
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


def stack_halos(samp,haloids,ncores):

    print('Sample name: ',sname)
    z = 0.
    ROUT = 5000.
    
    # READ catalogues from halo parameters
    rock       = pd.read_csv('/mnt/projects/lensing/SIDM_project/halo_props/halo_props_match_cdm_z0_rock2.csv.bz2')
    rock1      = pd.read_csv('/mnt/projects/lensing/SIDM_project/halo_props/halo_props_match_sidm1_z0_rock2.csv.bz2')

    # FOLDERS WHERE PARTICLES ARE SAVED
    path  = '/mnt/projects/lensing/SIDM_project/cuadrados/CDM/'
    path1 = '/mnt/projects/lensing/SIDM_project/cuadrados/SIDM/'

    # READ halos computed profperties
    main_file = '/mnt/projects/lensing/SIDM_project/halo_props/projections/v1_extend_halo_propsv2_rock2_match_cdm_z0_main.csv.bz2'
    main_file1 = '/mnt/projects/lensing/SIDM_project/halo_props/projections/v1_extend_halo_propsv2_rock2_match_sidm1_z0_main.csv.bz2'
    main  = pd.read_csv(main_file)
    main1 = pd.read_csv(main_file1)


    # LOAD 2D SHAPES
    q2d  = np.concatenate((main.b2D_xy/main.a2D_xy,main.b2D_zx/main.a2D_zx,main.b2D_yz/main.a2D_yz))
    q2d1 = np.concatenate((main1.b2D_xy/main1.a2D_xy,main1.b2D_zx/main1.a2D_zx,main1.b2D_yz/main1.a2D_yz))
    
    q2dr  = np.concatenate((main.b2Dr_xy/main.a2Dr_xy,main.b2Dr_zx/main.a2Dr_zx,main.b2Dr_yz/main.a2Dr_yz))
    q2dr1 = np.concatenate((main1.b2Dr_xy/main1.a2Dr_xy,main1.b2Dr_zx/main1.a2Dr_zx,main1.b2Dr_yz/main1.a2Dr_yz))
    
    q2d_it  = np.concatenate((main.b2D_it_xy/main.a2D_it_xy,main.b2D_it_zx/main.a2D_it_zx,main.b2D_it_yz/main.a2D_it_yz))
    q2d1_it = np.concatenate((main1.b2D_it_xy/main1.a2D_it_xy,main1.b2D_it_zx/main1.a2D_it_zx,main1.b2D_it_yz/main1.a2D_it_yz))
    
    q2dr_it  = np.concatenate((main.b2Dr_it_xy/main.a2Dr_it_xy,main.b2Dr_it_zx/main.a2Dr_it_zx,main.b2Dr_it_yz/main.a2Dr_it_yz))
    q2dr1_it = np.concatenate((main1.b2Dr_it_xy/main1.a2Dr_it_xy,main1.b2Dr_it_zx/main1.a2Dr_it_zx,main1.b2Dr_it_yz/main1.a2Dr_it_yz))

    nhalos = len(haloids)

    # FIRST STANDARD ORIENTATION
    print('Rotating according to standard orientation and stacking...')
    # ROTATE, STACK AND PROJECT PARTICLES    
    x,y,z,x2d,y2d      = stack_halos_parallel(main_file,path,haloids,reduced=False,iterative=False,ncores=ncores)   
    x1,y1,z1,x2d1,y2d1 = stack_halos_parallel(main_file1,path1,haloids1,reduced=False,iterative=False,ncores=ncores)

    # SELECT ONLY PARTICLES WITHIN 6Mpc
    m3d = (abs(x) < 10.)*(abs(y) < 10.)*(abs(z) < 10.)
    m2d = (abs(x2d) < 10.)*(abs(y2d) < 10.)

    m3d1 = (abs(x1) < 10.)*(abs(y1) < 10.)*(abs(z1) < 10.)
    m2d1 = (abs(x2d1) < 10.)*(abs(y2d1) < 10.)

    Xp,Yp = x2d[m2d]*1.e3,y2d[m2d]*1.e3 # 2D coordinates in kpc
    Xp1,Yp1  = x2d1[m2d1]*1.e3,y2d1[m2d1]*1.e3

    del(x,y,z,x2d,y2d)
    del(x1,y1,z1,x2d1,y2d1)
    # COMPUTE PROFILES USING PARTICLES

    print('Computing and fitting profiles...')
    # COMPUTE AND FIT PROFILES USING MAPS
    
    pm_DM2h_st   = fit_profiles(Xp,Yp,nhalos*3,twohalo=True,ROUT=ROUT)
    pm_SIDM2h_st = fit_profiles(Xp1,Yp1,nhalos*3,twohalo=True,ROUT=ROUT)

    del(Xp,Yp)
    del(Xp1,Yp1)
    
    #############################
    # NOW REDUCED ORIENTATION
    #############################
    
    print('Rotating according to reduced orientation and stacking...')
    # ROTATE, STACK AND PROJECT PARTICLES    
    x,y,z,x2d,y2d      = stack_halos_parallel(main_file,path,haloids,reduced=False,iterative=False,ncores=ncores)   
    x1,y1,z1,x2d1,y2d1 = stack_halos_parallel(main_file1,path1,haloids1,reduced=False,iterative=False,ncores=ncores)

    # SELECT ONLY PARTICLES WITHIN 6Mpc
    m3d = (abs(x) < 10.)*(abs(y) < 10.)*(abs(z) < 10.)
    m2d = (abs(x2d) < 10.)*(abs(y2d) < 10.)

    m3d1 = (abs(x1) < 10.)*(abs(y1) < 10.)*(abs(z1) < 10.)
    m2d1 = (abs(x2d1) < 10.)*(abs(y2d1) < 10.)

    Xp,Yp = x2d[m2d]*1.e3,y2d[m2d]*1.e3 # 2D coordinates in kpc
    Xp1,Yp1  = x2d1[m2d1]*1.e3,y2d1[m2d1]*1.e3

    del(x,y,z,x2d,y2d)
    del(x1,y1,z1,x2d1,y2d1)
    # COMPUTE PROFILES USING PARTICLES

    print('Computing and fitting profiles...')
    # COMPUTE AND FIT PROFILES USING MAPS
    ROUT = 5000.
    pm_DM2h_rd   = fit_profiles(Xp,Yp,nhalos*3,twohalo=True,ROUT=ROUT)
    pm_SIDM2h_rd = fit_profiles(Xp1,Yp1,nhalos*3,twohalo=True,ROUT=ROUT)

    del(Xp,Yp)
    del(Xp1,Yp1)
    
    # SAVING RESULTS
    print('Saving results....'

    CDM_res = [samp,pm_DM2h_st.lM200_ds,pm_DM2h_st.c200_ds,
              np.mean(q2d),np.mean(q2d_it),
              np.mean(q2dr),np.mean(q2dr_it),
              pm_DM2h_st.q1h_2g,pm_DM2h_st.q2h_2g,
              pm_DM2h_rd.q1h_2g,pm_DM2h_rd.q2h_2g,
              pm_DM2h_st.q1h_gt,pm_DM2h_st.q2h_gt,
              pm_DM2h_rd.q1h_gt,pm_DM2h_rd.q2h_gt,
              pm_DM2h_st.q1h_gx,pm_DM2h_st.q2h_gx,
              pm_DM2h_rd.q1h_gx,pm_DM2h_rd.q2h_gx]
              

    SIDM_res = [samp,pm_SIDM2h_st.lM200_ds,pm_SIDM2h_st.c200_ds,
              np.mean(q2d1),np.mean(q2d1_it),
              np.mean(q2dr1),np.mean(q2dr1_it),
              pm_SIDM2h_st.q1h_2g,pm_SIDM2h_st.q2h_2g,
              pm_SIDM2h_rd.q1h_2g,pm_SIDM2h_rd.q2h_2g,
              pm_SIDM2h_st.q1h_gt,pm_SIDM2h_st.q2h_gt,
              pm_SIDM2h_rd.q1h_gt,pm_SIDM2h_rd.q2h_gt,
              pm_SIDM2h_st.q1h_gx,pm_SIDM2h_st.q2h_gx,
              pm_SIDM2h_rd.q1h_gx,pm_SIDM2h_rd.q2h_gx]
              

    CDM_pro = [pm_DM2h_st.r,pm_DM2h_st.DS_T,
               pm_DM2h_st.GT,pm_DM2h_st.GX,
               pm_DM2h_rd.GT,pm_DM2h_rd.GX,
               pm_DM2h_st.DS_fit,
               pm_DM2h_st.GT1h,pm_DM2h_st.GX1h,
               pm_DM2h_st.GT2h,pm_DM2h_st.GX2h,
               pm_DM2h_rd.GT1h,pm_DM2h_rd.GX1h,
               pm_DM2h_rd.GT2h,pm_DM2h_rd.GX2h]

    SIDM_pro = [pm_SIDM2h_st.r,pm_SIDM2h_st.DS_T,
               pm_SIDM2h_st.GT,pm_SIDM2h_st.GX,
               pm_SIDM2h_rd.GT,pm_SIDM2h_rd.GX,
               pm_SIDM2h_st.DS_fit,
               pm_SIDM2h_st.GT1h,pm_SIDM2h_st.GX1h,
               pm_SIDM2h_st.GT2h,pm_SIDM2h_st.GX2h,
               pm_SIDM2h_rd.GT1h,pm_SIDM2h_rd.GX1h,
               pm_SIDM2h_rd.GT2h,pm_SIDM2h_rd.GX2h]

    return CDM_res,CDM_pro,SIDM_res,SIDM_pro
