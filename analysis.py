import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import pandas as pd
from scipy import stats
from astropy.io import fits
from binned_plots import make_plot2

# halos = fits.open('/home/elizabeth/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/HS-lensing/HALO_Props_MICE.fits')[1].data        
halos = fits.open('/home/elizabeth/Documentos/proyectos/HALO-SHAPE/MICE/HS-lensing/HALO_Props_MICE.fits')[1].data        

zs = ['z0','z51','z96']
# z = 'z96'
# z = 'z51'
mask = (halos.z < 0.05)


z = zs[0]
    

rock     = pd.read_csv('../halo_props/halo_props_cdm_'+z+'_rock.csv.bz2')
main     = pd.read_csv('../halo_props/halo_props_cdm_'+z+'_main.csv.bz2')
main_fof = pd.read_csv('../halo_props/halo_props_fof_cdm_'+z+'_main.csv.bz2')

rock1     = pd.read_csv('../halo_props/halo_props_sidm1_'+z+'_rock.csv.bz2')
main1     = pd.read_csv('../halo_props/halo_props_sidm1_'+z+'_main.csv.bz2')
main1_fof = pd.read_csv('../halo_props/halo_props_fof_sidm1_'+z+'_main.csv.bz2')

    
    # LOAD PARAMS
    
S_rock = rock['c to a']
Q_rock = rock['b to a']
S1_rock = rock1['c to a']
Q1_rock = rock1['b to a']

S = main.c3D/main.a3D
Q = main.c3D/main.a3D
q = main.b2D/main.a2D
S1 = main1.c3D/main1.a3D
Q1 = main1.c3D/main1.a3D
q1 = main1.b2D/main1.a2D

S_fof  = main_fof.c3D/main_fof.a3D
Q_fof  = main_fof.c3D/main_fof.a3D
q_fof  = main_fof.b2D/main_fof.a2D
S1_fof = main1_fof.c3D/main1_fof.a3D
Q1_fof = main1_fof.c3D/main1_fof.a3D
q1_fof = main1_fof.b2D/main1_fof.a2D
    
Eratio  = (2.*main.EKin/abs(main.EPot))
Eratio1 = (2.*main1.EKin/abs(main1.EPot))

Eratio_fof  = (2.*main_fof.EKin/abs(main_fof.EPot))
Eratio1_fof = (2.*main1_fof.EKin/abs(main1_fof.EPot))

Eratio_mice = (2.*halos.K)/abs(halos.U)

lgM = main.lgM
lgM1 = main1.lgM

lgM_fof  = main_fof.lgM
lgM1_fof = main1_fof.lgM
    
rc_fof  = np.array(np.sqrt((main_fof.xc - main_fof.xc_rc)**2 + (main_fof.yc - main_fof.yc_rc)**2 + (main_fof.zc - main_fof.zc_rc)**2))
rc1_fof = np.array(np.sqrt((main1_fof.xc - main1_fof.xc_rc)**2 + (main1_fof.yc - main1_fof.yc_rc)**2 + (main1_fof.zc - main1_fof.zc_rc)**2))
offset_fof  = rc_fof/main_fof.r_max
offset1_fof = rc1_fof/main1_fof.r_max

make_plot2(main.lgM,S,nbins=4,color='C0',error=True,label='new par')
make_plot2(main.lgM,S_rock,nbins=4,color='C1',error=True,label='rockstar')
make_plot2(main_fof.lgM,S_fof,nbins=4,color='C2',error=True,label='fof')
make_plot2(halos.lgM[mask],halos.s[mask],nbins=4,color='C7',error=True,label='MICE')

make_plot2(main.lgM,S1,nbins=4,color='C0',error=True,label='new par',lt='--')
make_plot2(main.lgM,S1_rock,nbins=4,color='C1',error=True,label='rockstar',lt='--')
make_plot2(main_fof.lgM,S1_fof,nbins=4,color='C2',error=True,label='fof',lt='--')


plt.figure()
make_plot2(main_fof.lgM,S1_fof,nbins=4,color='C2',error=True,label='fof',lt='--')
make_plot2(halos.lgM[mask],halos.s[mask],nbins=4,color='C7',error=True,label='MICE')
make_plot2(main_fof.lgM,S_fof,nbins=4,color='C2',error=True,label='fof')


plt.figure()
make_plot2(main_fof.lgM[offset1_fof<0.1],S1_fof[offset1_fof<0.1],nbins=4,color='C2',error=True,label='fof',lt='--')
make_plot2(halos.lgM[mask*(halos.offset<0.1)],halos.s[mask*(halos.offset<0.1)],nbins=4,color='C7',error=True,label='MICE')
make_plot2(main_fof.lgM[offset_fof<0.1],S_fof[offset_fof<0.1],nbins=4,color='C2',error=True,label='fof')

plt.figure()
plt.hist(Eratio_fof,np.linspace(0.5,2.5,50),histtype='step',density=True,label='FOF')
plt.hist(Eratio_mice,np.linspace(0.5,2.5,50),histtype='step',density=True,label='MICE')
plt.hist(Eratio,np.linspace(0.5,2.5,50),histtype='step',density=True,label='Rockstar')
