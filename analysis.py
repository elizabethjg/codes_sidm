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
mask = (halos.z < 0.07)


z = zs[0]
    

rock      = pd.read_csv('../halo_props/halo_props_cdm_'+z+'_rock2.csv.bz2')
# main      = pd.read_csv('../halo_props/halo_props_cdm_'+z+'_main.csv.bz2')
main_fof  = pd.read_csv('../halo_props/halo_props_fof_cdm_'+z+'_main.csv.bz2')
main_rock = pd.read_csv('../halo_props/halo_props_rock2_cdm_'+z+'_main.csv.bz2')

rock1      = pd.read_csv('../halo_props/halo_props_sidm1_'+z+'_rock2.csv.bz2')
# main1      = pd.read_csv('../halo_props/halo_props_sidm1_'+z+'_main.csv.bz2')
main1_fof  = pd.read_csv('../halo_props/halo_props_fof_sidm1_'+z+'_main.csv.bz2')
main1_rock = pd.read_csv('../halo_props/halo_props_rock2_sidm1_'+z+'_main.csv.bz2')

    
    # LOAD PARAMS
    
S_rock  = rock['c_to_a']
Q_rock  = rock['b_to_a']
T_rock  = (1. - rock['b_to_a']**2)/(1. - rock['c_to_a']**2)

S1_rock = rock1['c_to_a']
Q1_rock = rock1['b_to_a']
T1_rock  = (1. - rock1['b_to_a']**2)/(1. - rock1['c_to_a']**2)

S_rock_2  = rock['c_to_a2']
Q_rock_2  = rock['b_to_a2']
T_rock_2  = (1. - rock['b_to_a2']**2)/(1. - rock['c_to_a2']**2)

S1_rock_2 = rock1['c_to_a2']
Q1_rock_2 = rock1['b_to_a2']
T1_rock_2  = (1. - rock1['b_to_a2']**2)/(1. - rock1['c_to_a2']**2)

S_fof  = main_fof.c3D/main_fof.a3D
Q_fof  = main_fof.c3D/main_fof.a3D
T_fof  = (main_fof.a3D**2 - main_fof.b3D**2)/(main_fof.a3D**2 - main_fof.c3D**2)
q_fof  = main_fof.b2D/main_fof.a2D

S1_fof = main1_fof.c3D/main1_fof.a3D
Q1_fof = main1_fof.c3D/main1_fof.a3D
T1_fof  = (main1_fof.a3D**2 - main1_fof.b3D**2)/(main1_fof.a3D**2 - main1_fof.c3D**2)
q1_fof = main1_fof.b2D/main1_fof.a2D

Sr_fof  = main_fof.c3Dr/main_fof.a3Dr
Qr_fof  = main_fof.c3Dr/main_fof.a3Dr
qr_fof  = main_fof.b2Dr/main_fof.a2Dr
Tr_fof  = (main_fof.a3Dr**2 - main_fof.b3Dr**2)/(main_fof.a3Dr**2 - main_fof.c3Dr**2)

S1r_fof = main1_fof.c3Dr/main1_fof.a3Dr
Q1r_fof = main1_fof.c3Dr/main1_fof.a3Dr
q1r_fof = main1_fof.b2Dr/main1_fof.a2Dr
T1r_fof  = (main1_fof.a3Dr**2 - main1_fof.b3Dr**2)/(main1_fof.a3Dr**2 - main1_fof.c3Dr**2)

S_rock2  = main_rock.c3D/main_rock.a3D
Q_rock2  = main_rock.c3D/main_rock.a3D
q_rock2  = main_rock.b2D/main_rock.a2D
T_rock2  = (main_rock.a3D**2 - main_rock.b3D**2)/(main_rock.a3D**2 - main_rock.c3D**2)

S1_rock2 = main1_rock.c3D/main1_rock.a3D
Q1_rock2 = main1_rock.c3D/main1_rock.a3D
q1_rock2 = main1_rock.b2D/main1_rock.a2D
T1_rock2  = (main1_rock.a3D**2 - main1_rock.b3D**2)/(main1_rock.a3D**2 - main1_rock.c3D**2)

Sr_rock2  = main_rock.c3Dr/main_rock.a3Dr
Qr_rock2  = main_rock.c3Dr/main_rock.a3Dr
qr_rock2  = main_rock.b2Dr/main_rock.a2Dr
Tr_rock2  = (main_rock.a3Dr**2 - main_rock.b3Dr**2)/(main_rock.a3Dr**2 - main_rock.c3Dr**2)

S1r_rock2 = main1_rock.c3Dr/main1_rock.a3Dr
Q1r_rock2 = main1_rock.c3Dr/main1_rock.a3Dr
q1r_rock2 = main1_rock.b2Dr/main1_rock.a2Dr
T1r_rock2 = (main1_rock.a3Dr**2 - main1_rock.b3Dr**2)/(main1_rock.a3Dr**2 - main1_rock.c3Dr**2)
    
Eratio_fof  = (2.*main_fof.EKin/abs(main_fof.EPot))
Eratio1_fof = (2.*main1_fof.EKin/abs(main1_fof.EPot))

Eratio_mice = (2.*halos.K)/abs(halos.U)

lgM = np.log10(rock.Mvir)
lgM1 = np.log10(rock1.Mvir)

lgM_fof  = main_fof.lgM
lgM1_fof = main1_fof.lgM
    
rc_fof  = np.array(np.sqrt((main_fof.xc - main_fof.xc_rc)**2 + (main_fof.yc - main_fof.yc_rc)**2 + (main_fof.zc - main_fof.zc_rc)**2))
rc1_fof = np.array(np.sqrt((main1_fof.xc - main1_fof.xc_rc)**2 + (main1_fof.yc - main1_fof.yc_rc)**2 + (main1_fof.zc - main1_fof.zc_rc)**2))
offset_fof  = rc_fof/main_fof.r_max
offset1_fof = rc1_fof/main1_fof.r_max

lgM_rock  = main_rock.lgM
lgM1_rock = main1_rock.lgM
    
rc_rock  = np.array(np.sqrt((main_rock.xc - main_rock.xc_rc)**2 + (main_rock.yc - main_rock.yc_rc)**2 + (main_rock.zc - main_rock.zc_rc)**2))
rc1_rock = np.array(np.sqrt((main1_rock.xc - main1_rock.xc_rc)**2 + (main1_rock.yc - main1_rock.yc_rc)**2 + (main1_rock.zc - main1_rock.zc_rc)**2))
offset_rock  = rc_rock/main_rock.r_max
offset1_rock = rc1_rock/main1_rock.r_max

plt.figure()
plt.xlabel('$\log M$')
plt.ylabel('$S = c/a$')
make_plot2(lgM,S_rock,nbins=4,color='C1',error=True,label='rockstar - rockstar')
make_plot2(lgM,S_rock_2,nbins=4,color='C0',error=True,label='rockstar - rockstar2')
make_plot2(main_fof.lgM-0.2,S_fof,nbins=4,color='C2',error=True,label='FOF - new par')
make_plot2(main_rock.lgM-0.2,S_rock2,nbins=4,color='C3',error=True,label='rockstar - new par')
make_plot2(halos.lgM[mask]-0.2,halos.s[mask],nbins=4,color='C7',error=True,label='FOF - new par (MICE)')
plt.legend()
make_plot2(lgM1,S1_rock,nbins=4,color='C1',error=True,label='rockstar',lt='--')
make_plot2(lgM1,S1_rock_2,nbins=4,color='C0',error=True,label='rockstar',lt='--')
make_plot2(main_fof.lgM-0.2,S1_fof,nbins=4,color='C2',error=True,label='fof',lt='--')
make_plot2(main_rock.lgM-0.2,S1_rock2,nbins=4,color='C3',error=True,label='fof',lt='--')

plt.figure()
plt.xlabel('$\log M$')
plt.ylabel('$T$')
make_plot2(lgM,T_rock,nbins=4,color='C1',error=True,label='rockstar - rockstar')
make_plot2(lgM,T_rock_2,nbins=4,color='C0',error=True,label='rockstar - rockstar2')
make_plot2(main_fof.lgM-0.2,T_fof,nbins=4,color='C2',error=True,label='FOF - new par')
make_plot2(main_rock.lgM-0.2,T_rock2,nbins=4,color='C3',error=True,label='rockstar - new par')
make_plot2(halos.lgM[mask]-0.2,((1.-halos.q**2)/(1.-halos.s**2))[mask],nbins=4,color='C7',error=True,label='FOF - new par (MICE)')
plt.legend()
make_plot2(lgM1,T1_rock,nbins=4,color='C1',error=True,label='rockstar',lt='--')
make_plot2(lgM1,T1_rock_2,nbins=4,color='C0',error=True,label='rockstar',lt='--')
make_plot2(main_fof.lgM-0.2,T1_fof,nbins=4,color='C2',error=True,label='fof',lt='--')
make_plot2(main_rock.lgM-0.2,T1_rock2,nbins=4,color='C3',error=True,label='fof',lt='--')


plt.figure()
plt.xlabel('$\log M_{FOF}$')
plt.ylabel('$S = c/a$')
make_plot2(halos.lgM[mask],halos.s[mask],nbins=4,color='C7',error=True,label='MICE')
make_plot2(main_rock.lgM,S1_rock2,nbins=4,color='C3',error=True,label='SIDM',lt='--')
make_plot2(main_fof.lgM,S1_fof,nbins=4,color='C2',error=True,label='SIDM',lt='--')
make_plot2(main_rock.lgM,S_rock2,nbins=4,color='C3',error=True,label='DM')
make_plot2(main_fof.lgM,S_fof,nbins=4,color='C2',error=True,label='DM')
plt.legend()

plt.figure()
plt.xlabel('$\log M_{FOF}$')
plt.ylabel('$S_r = c_r/a_r$')
make_plot2(halos.lgM[mask],halos.sr[mask],nbins=4,color='C7',error=True,label='MICE')
make_plot2(main_rock.lgM,S1r_rock2,nbins=4,color='C3',error=True,label='SIDM',lt='--')
make_plot2(main_fof.lgM,S1r_fof,nbins=4,color='C2',error=True,label='SIDM',lt='--')
make_plot2(main_rock.lgM,Sr_rock2,nbins=4,color='C3',error=True,label='DM')
make_plot2(main_fof.lgM,Sr_fof,nbins=4,color='C2',error=True,label='DM')
plt.legend()


doff = 0.1
mlim = 14.2

plt.figure()
plt.xlabel('$\log M_{FOF}$')
plt.ylabel('$S = c/a$')
make_plot2(halos.lgM[mask*(halos.offset<doff)],halos.s[mask*(halos.offset<doff)],nbins=4,color='C7',error=True,label='MICE')
make_plot2(main_fof.lgM[offset1_fof<doff],S1_fof[offset1_fof<doff],nbins=4,color='C2',error=True,label='fof',lt='--')
make_plot2(main_fof.lgM[offset_fof<doff],S_fof[offset_fof<doff],nbins=4,color='C2',error=True,label='fof')
plt.legend()

plt.figure()
plt.xlabel('$\log M_{FOF}$')
plt.ylabel('$S_r = c_r/a_r$')
make_plot2(halos.lgM[mask*(halos.offset<doff)],halos.sr[mask*(halos.offset<doff)],nbins=4,color='C7',error=True,label='MICE')
make_plot2(main_fof.lgM[offset1_fof<doff],S1r_fof[offset1_fof<doff],nbins=4,color='C2',error=True,label='fof',lt='--')
make_plot2(main_fof.lgM[offset_fof<doff],Sr_fof[offset_fof<doff],nbins=4,color='C2',error=True,label='fof')
plt.legend()

plt.figure()
plt.hist(halos.s[mask*(halos.offset<doff)*(halos.lgM<mlim)],np.linspace(0.2,1.,15),histtype='step',color='C7',density=True,label='MICE')
plt.hist(S_fof[(offset_fof<doff)*(main_fof.lgM<mlim)*(S_fof > 0.5)],np.linspace(0.2,1.,15),histtype='step',color='C2',density=True,label='CDM')
plt.hist(S1_fof[(offset1_fof<doff)*(main_fof.lgM<mlim)*(S1_fof > 0.5)],np.linspace(0.2,1.,15),histtype='step',color='C3',density=True,label='SIDM')
plt.axvline(np.mean(S_fof[(offset_fof<doff)*(main_fof.lgM<mlim)*(S_fof > 0.5)]),color='C2')
plt.axvline(np.mean(S1_fof[(offset1_fof<doff)*(main_fof.lgM<mlim)*(S1_fof > 0.5)]),color='C3')
plt.xlabel('$S$')
plt.ylabel('$N$')
plt.legend()

plt.figure()
plt.hist(halos.q2d[mask*(halos.offset<doff)*(halos.lgM<mlim)],np.linspace(0.2,1.,15),histtype='step',color='C7',density=True,label='MICE')
plt.hist(q_fof[(offset_fof<doff)*(main_fof.lgM<mlim)],np.linspace(0.2,1.,15),histtype='step',color='C2',density=True,label='CDM')
plt.hist(q1_fof[(offset1_fof<doff)*(main_fof.lgM<mlim)],np.linspace(0.2,1.,15),histtype='step',color='C3',density=True,label='SIDM')
plt.axvline(np.mean(q_fof[(offset_fof<doff)*(main_fof.lgM<mlim)]),color='C2')
plt.axvline(np.mean(q1_fof[(offset1_fof<doff)*(main_fof.lgM<mlim)]),color='C3')
plt.xlabel('$q$')
plt.ylabel('$N$')
plt.legend()


plt.figure()
plt.hist(halos.s[mask*(halos.offset<doff)*(halos.lgM>=mlim)],np.linspace(0.2,1.,15),histtype='step',color='C7',density=True)
plt.hist(S_fof[(offset_fof<doff)*(main_fof.lgM>=mlim)],np.linspace(0.2,1.,15),histtype='step',color='C2',density=True,label='CDM')
plt.hist(S1_fof[(offset1_fof<doff)*(main_fof.lgM>=mlim)],np.linspace(0.2,1.,15),histtype='step',color='C3',density=True,label='SIDM')
plt.axvline(np.mean(S_fof[(offset_fof<doff)*(main_fof.lgM>=mlim)]),color='C2')
plt.axvline(np.mean(S1_fof[(offset1_fof<doff)*(main_fof.lgM>=mlim)]),color='C3')
plt.legend()

print(np.mean(S_fof[(offset_fof<doff)*(main_fof.lgM<mlim)])/np.mean(S1_fof[(offset1_fof<doff)*(main_fof.lgM<mlim)]))
print(np.mean(S_fof[(offset_fof<doff)*(main_fof.lgM>=mlim)])/np.mean(S1_fof[(offset1_fof<doff)*(main_fof.lgM>=mlim)]))

plt.figure()
plt.hist(Eratio_fof,np.linspace(0.5,2.5,50),histtype='step',density=True,label='FOF')
plt.hist(Eratio_mice,np.linspace(0.5,2.5,50),histtype='step',density=True,label='MICE')
plt.hist(Eratio,np.linspace(0.5,2.5,50),histtype='step',density=True,label='Rockstar')
