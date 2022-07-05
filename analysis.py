import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import pandas as pd
from scipy import stats
from astropy.io import fits
from binned_plots import make_plot2

try:
    halos = fits.open('/home/elizabeth/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/HS-lensing/HALO_Props_MICE.fits')[1].data        
except:
    halos = fits.open('/home/elizabeth/Documentos/proyectos/HALO-SHAPE/MICE/HS-lensing/HALO_Props_MICE.fits')[1].data        

zs = ['z0','z51','z96']
# z = 'z96'
# z = 'z51'
mask = (halos.z < 0.1)

f1, ax1 = plt.subplots(1,3, figsize=(14,4),sharex=True,sharey=True)
f1.subplots_adjust(wspace=0)

f2, ax2 = plt.subplots(1,3, figsize=(14,4),sharex=True,sharey=True)
f2.subplots_adjust(wspace=0)

f3, ax3 = plt.subplots(1,3, figsize=(14,4),sharex=True,sharey=True)
f3.subplots_adjust(wspace=0)

f4, ax4 = plt.subplots(1,3, figsize=(14,4),sharex=True,sharey=True)
f4.subplots_adjust(wspace=0)

f5, ax5 = plt.subplots(1,3, figsize=(14,4),sharex=True,sharey=True)
f5.subplots_adjust(wspace=0)

f6, ax6 = plt.subplots(1,3, figsize=(14,4),sharex=True,sharey=True)
f6.subplots_adjust(wspace=0)

f7, ax7 = plt.subplots(1,3, figsize=(14,4),sharex=True,sharey=True)
f7.subplots_adjust(wspace=0)

f8, ax8 = plt.subplots(1,3, figsize=(14,4),sharex=True,sharey=True)
f8.subplots_adjust(wspace=0)

f9, ax9 = plt.subplots(1,3, figsize=(14,4),sharex=True,sharey=True)
f9.subplots_adjust(wspace=0)

f10, ax10 = plt.subplots(1,3, figsize=(14,4),sharex=True,sharey=True)
f10.subplots_adjust(wspace=0)


for j in range(3):

    z = zs[j]
    

    rock     = pd.read_csv('../halo_props/halo_props_cdm_'+z+'_rock.csv.bz2')
    main     = pd.read_csv('../halo_props/halo_props_cdm_'+z+'_main.csv.bz2')
    
    rock1    = pd.read_csv('../halo_props/halo_props_sidm1_'+z+'_rock.csv.bz2')
    main1    = pd.read_csv('../halo_props/halo_props_sidm1_'+z+'_main.csv.bz2')
    
    # LOAD PARAMS
    
    S_rock = rock['c to a']
    Q_rock = rock['b to a']
    S_rock1 = rock1['c to a']
    Q_rock1 = rock1['b to a']
    
    S = main.c3D/main.a3D
    Q = main.c3D/main.a3D
    q = main.b2D/main.a2D
    S1 = main1.c3D/main1.a3D
    Q1 = main1.c3D/main1.a3D
    q1 = main1.b2D/main1.a2D
    
    Eratio  = (2.*main.EKin/abs(main.EPot))
    Eratio1 = (2.*main1.EKin/abs(main1.EPot))
    
    lgM = main.lgM
    lgM1 = main1.lgM
    
    rc = np.array(np.sqrt((main.xc - main.xc_rc)**2 + (main.yc - main.yc_rc)**2 + (main.zc - main.zc_rc)**2))
    rc1 = np.array(np.sqrt((main1.xc - main1.xc_rc)**2 + (main1.yc - main1.yc_rc)**2 + (main1.zc - main1.zc_rc)**2))
    offset = rc/main.r_max
    offset1 = rc1/main1.r_max
    
    
    # COMPARISON rock vs new params

    ax1[j].plot(S_rock,S,'.',label='CDM')
    ax1[j].plot(S_rock1,S1,'x',label='SIDM')
    ax1[j].plot([0,1],[0,1],'C7--')
    ax1[j].set_xlabel('c/a - ROCKSTAR')
    ax1[j].set_ylabel('c/a - new params')
    ax1[j].legend()
    
    
    ax2[j].plot(S_rock,S_rock1,'.',label='rock')
    ax2[j].plot(S,S1,'x',label='new params')
    ax2[j].plot([0,1],[0,1],'C7--')
    ax2[j].set_xlabel('c/a - CDM')
    ax2[j].set_ylabel('c/a - SIDM')
    ax2[j].legend()
    
    
    ax3[j].plot(Q_rock,Q,'.',label='CDM')
    ax3[j].plot(Q_rock1,Q1,'x',label='SIDM')
    ax3[j].plot([0,1],[0,1],'C7--')
    ax3[j].set_xlabel('b/a - ROCKSTAR')
    ax3[j].set_ylabel('b/a - new params')
    ax3[j].legend()
    
    
    ax4[j].plot(Q_rock,Q_rock1,'.',label='rock')
    ax4[j].plot(Q,Q1,'x',label='new params')
    ax4[j].plot([0,1],[0,1],'C7--')
    ax4[j].set_xlabel('b/a - CDM')
    ax4[j].set_ylabel('b/a - SIDM')
    ax4[j].legend()
    
    
    make_plot2(main.lgM,q1,nbins=4,color='C1',error=True,label='SIDM',plt=ax5[j])
    make_plot2(main.lgM,q,nbins=4,color='C0',error=True,label='CDM',plt=ax5[j])
    ax5[j].legend()
    ax5[j].set_xlabel('$\log(M_{vir})$')
    ax5[j].set_ylabel('$q_{2D}$')
    
    
    make_plot2(main.lgM,S_rock,nbins=4,color='C1',error=True,label='SIDM',plt=ax6[j])
    make_plot2(main.lgM,S_rock1,nbins=4,color='C0',error=True,label='CDM',plt=ax6[j])
    # make_plot2(halos.lgM[mask],halos.s[mask],nbins=4,color='C7',error=True,label='MICE')
    ax6[j].legend()
    ax6[j].set_xlabel('$\log(M_{vir})$')
    ax6[j].set_ylabel('$c/a$')
    
    
    make_plot2(main.lgM,S,nbins=4,color='C1',error=True,label='SIDM',plt=ax7[j])
    make_plot2(main.lgM,S1,nbins=4,color='C0',error=True,label='CDM',plt=ax7[j])
    # make_plot2(halos.lgM[mask],halos.s[mask],nbins=4,color='C7',error=True,label='MICE')
    ax7[j].legend()
    ax7[j].set_xlabel('$\log(M_{vir})$')
    ax7[j].set_ylabel('$c/a$')
    
    
    ax8[j].plot(offset,offset1,'.')
    ax8[j].plot([0,1.],[0,1],'C7--')
    ax8[j].set_xlabel('$r_c/r_{MAX}$ - CDM')
    ax8[j].set_ylabel('$r_c/r_{MAX}$ - SIDM')
    ax8[j].loglog()

    
    ax9[j].plot(Eratio,Eratio1,'C3.')
    ax9[j].plot([0,6],[0,6],'C7--')
    ax9[j].axvline(1.35)
    ax9[j].axhline(1.35)
    ax9[j].set_xlabel('$2K/U$ - CDM')
    ax9[j].set_ylabel('$2K/U$ - SIDM')
    ax9[j].axis([1,2,0.,4])
    
    
    ax10[j].scatter(S_rock,S,c=offset,vmax=0.1)
    ax10[j].plot([0,1],[0,1],'C7--')
    ax10[j].set_xlabel('c/a - ROCKSTAR')
    ax10[j].set_ylabel('c/a - new params')
    ax10[j].legend()

f1.savefig('../s_rock_new.png')
f2.savefig('../s_cdm_sidm.png')
f3.savefig('../q_rock_new.png')
f4.savefig('../q_cdm_sidm.png')
f5.savefig('../mass_q2d.png')
f6.savefig('../mass_Srock.png')
f7.savefig('../mass_S.png')
f8.savefig('../offset.png')    
f9.savefig('../Eratio.png')

im0 = ax10[0].scatter(S_rock,S,c=offset,vmax=0.1)
f10.colorbar(im0, ax=ax10, orientation='vertical', fraction=.05)
f10.savefig('../s_rock_new_offset.png')
