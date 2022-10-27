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

def mask_border(x0,y0,z0):
    mask = (x0 > 2.)*(x0 < 118.)*(y0 > 2.)*(y0 < 118.)*(z0 > 2.)*(z0 < 118.)
    return mask

rock       = pd.read_csv('../halo_props/halo_props_cdm_'+z+'_rock2.csv.bz2')
rock_sh    = pd.read_csv('../halo_props/halo_props_sh_cdm_'+z+'_rock2.csv.bz2')
fof        = pd.read_csv('../halo_props/halo_props_cdm_'+z+'_fof.csv.bz2')
rock1      = pd.read_csv('../halo_props/halo_props_sidm1_'+z+'_rock2.csv.bz2')
rock1_sh   = pd.read_csv('../halo_props/halo_props_sh_sidm1_'+z+'_rock2.csv.bz2')
fof1       = pd.read_csv('../halo_props/halo_props_sidm1_'+z+'_fof.csv.bz2')

mrock     = mask_border(rock.x0,rock.y0,rock.z0)
mrock_sh  = mask_border(rock_sh.x0,rock_sh.y0,rock_sh.z0)
mfof      = mask_border(fof.x0,fof.y0,fof.z0)
mrock1    = mask_border(rock1.x0,rock1.y0,rock.z0)
mrock1_sh = mask_border(rock1_sh.x0,rock1_sh.y0,rock1_sh.z0)
mfof1     = mask_border(fof1.x0,fof1.y0,fof1.z0)

print('rock',len(mrock),mrock.sum())
print('rock_sh',len(mrock_sh),mrock_sh.sum())
print('fof',len(mfof),mfof.sum())

print('rock1',len(mrock1),mrock1.sum())
print('rock1_sh',len(mrock1_sh),mrock1_sh.sum())
print('fof1',len(mfof1),mfof1.sum())


rock       =  rock[mrock]    
rock_sh    =  rock_sh[mrock_sh] 
fof        =  fof[mfof]     
rock1      =  rock1[mrock1]   
rock1_sh   =  rock1_sh[mrock1_sh]
fof1       =  fof1[mfof1]    

main_fof  = pd.read_csv('../halo_props/halo_props_fof_cdm_'+z+'_main.csv.bz2')[mfof]
main_rock = pd.read_csv('../halo_props/halo_props_rock2_cdm_'+z+'_main.csv.bz2')[mrock]
main_v2 = pd.read_csv('../halo_props/halo_propsv2_rock2_cdm_'+z+'_main.csv.bz2')[mrock]
main_sh = pd.read_csv('../halo_props/halo_props_rock2_sh_cdm__main.csv.bz2')[mrock_sh]
main_it = pd.read_csv('../halo_props/halo_props_iterative_rock2_cdm_z0_main.csv.bz2')[mrock]

main1_fof  = pd.read_csv('../halo_props/halo_props_fof_sidm1_'+z+'_main.csv.bz2')[mfof1]
main1_rock = pd.read_csv('../halo_props/halo_props_rock2_sidm1_'+z+'_main.csv.bz2')[mrock1]
main1_v2 = pd.read_csv('../halo_props/halo_propsv2_rock2_sidm1_'+z+'_main.csv.bz2')[mrock1]
main1_sh = pd.read_csv('../halo_props/halo_props_rock2_sh_sidm1__main.csv.bz2')[mrock1_sh]
main1_it = pd.read_csv('../halo_props/halo_props_iterative_rock2_sidm1_z0_main.csv.bz2')[mrock1]

    
    # LOAD PARAMS

nhalos = rock_sh['N_sh']
nhalos1 = rock1_sh['N_sh']

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

S_sh  = main_sh.c3D/main_sh.a3D
Q_sh  = main_sh.c3D/main_sh.a3D
q_sh  = main_sh.b2D/main_sh.a2D
T_sh  = (main_sh.a3D**2 - main_sh.b3D**2)/(main_sh.a3D**2 - main_sh.c3D**2)

S1_sh = main1_sh.c3D/main1_sh.a3D
Q1_sh = main1_sh.c3D/main1_sh.a3D
q1_sh = main1_sh.b2D/main1_sh.a2D
T1_sh  = (main1_sh.a3D**2 - main1_sh.b3D**2)/(main1_sh.a3D**2 - main1_sh.c3D**2)

Sr_sh  = main_sh.c3Dr/main_sh.a3Dr
Qr_sh  = main_sh.c3Dr/main_sh.a3Dr
qr_sh  = main_sh.b2Dr/main_sh.a2Dr
Tr_sh  = (main_sh.a3Dr**2 - main_sh.b3Dr**2)/(main_sh.a3Dr**2 - main_sh.c3Dr**2)

S1r_sh = main1_sh.c3Dr/main1_sh.a3Dr
Q1r_sh = main1_sh.c3Dr/main1_sh.a3Dr
q1r_sh = main1_sh.b2Dr/main1_sh.a2Dr
T1r_sh = (main1_sh.a3Dr**2 - main1_sh.b3Dr**2)/(main1_sh.a3Dr**2 - main1_sh.c3Dr**2)

S_v2  = main_v2.c3D/main_v2.a3D
Q_v2  = main_v2.c3D/main_v2.a3D
q_v2  = main_v2.b2D/main_v2.a2D
T_v2  = (main_v2.a3D**2 - main_v2.b3D**2)/(main_v2.a3D**2 - main_v2.c3D**2)

S1_v2 = main1_v2.c3D/main1_v2.a3D
Q1_v2 = main1_v2.c3D/main1_v2.a3D
q1_v2 = main1_v2.b2D/main1_v2.a2D
T1_v2  = (main1_v2.a3D**2 - main1_v2.b3D**2)/(main1_v2.a3D**2 - main1_v2.c3D**2)

Sr_v2  = main_v2.c3Dr/main_v2.a3Dr
Qr_v2  = main_v2.c3Dr/main_v2.a3Dr
qr_v2  = main_v2.b2Dr/main_v2.a2Dr
Tr_v2  = (main_v2.a3Dr**2 - main_v2.b3Dr**2)/(main_v2.a3Dr**2 - main_v2.c3Dr**2)

S1r_v2 = main1_v2.c3Dr/main1_v2.a3Dr
Q1r_v2 = main1_v2.c3Dr/main1_v2.a3Dr
q1r_v2 = main1_v2.b2Dr/main1_v2.a2Dr
T1r_v2 = (main1_v2.a3Dr**2 - main1_v2.b3Dr**2)/(main1_v2.a3Dr**2 - main1_v2.c3Dr**2)

S_it  = main_it.c3D/main_it.a3D
Q_it  = main_it.c3D/main_it.a3D
q_it  = main_it.b2D/main_it.a2D
T_it  = (main_it.a3D**2 - main_it.b3D**2)/(main_it.a3D**2 - main_it.c3D**2)

S1_it = main1_it.c3D/main1_it.a3D
Q1_it = main1_it.c3D/main1_it.a3D
q1_it = main1_it.b2D/main1_it.a2D
T1_it  = (main1_it.a3D**2 - main1_it.b3D**2)/(main1_it.a3D**2 - main1_it.c3D**2)

Sr_it  = main_it.c3Dr/main_it.a3Dr
Qr_it  = main_it.c3Dr/main_it.a3Dr
qr_it  = main_it.b2Dr/main_it.a2Dr
Tr_it  = (main_it.a3Dr**2 - main_it.b3Dr**2)/(main_it.a3Dr**2 - main_it.c3Dr**2)

S1r_it = main1_it.c3Dr/main1_it.a3Dr
Q1r_it = main1_it.c3Dr/main1_it.a3Dr
q1r_it = main1_it.b2Dr/main1_it.a2Dr
T1r_it = (main1_it.a3Dr**2 - main1_it.b3Dr**2)/(main1_it.a3Dr**2 - main1_it.c3Dr**2)

mv2r  = ~np.isnan(main_v2.a3Dr)*(Sr_v2 > 0.)
mv2r1 = ~np.isnan(main1_v2.a3Dr)*(S1r_v2 > 0.)
mv2   = ~np.isnan(main_v2.a3D)*(S_v2 > 0.)
mv21  = ~np.isnan(main1_v2.a3D)*(S1_v2 > 0.)

mitr  = ~np.isnan(main_it.a3Dr)*(Sr_it > 0.)
mitr1 = ~np.isnan(main1_it.a3Dr)*(S1r_it > 0.)
mit   = ~np.isnan(main_it.a3D)*(S_it > 0.)
mit1  = ~np.isnan(main1_it.a3D)*(S1_it > 0.)

    
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

lgM_sh  = main_sh.lgM
lgM1_sh = main1_sh.lgM
    
rc_rock  = np.array(np.sqrt((main_rock.xc - main_rock.xc_rc)**2 + (main_rock.yc - main_rock.yc_rc)**2 + (main_rock.zc - main_rock.zc_rc)**2))
rc1_rock = np.array(np.sqrt((main1_rock.xc - main1_rock.xc_rc)**2 + (main1_rock.yc - main1_rock.yc_rc)**2 + (main1_rock.zc - main1_rock.zc_rc)**2))
offset_rock  = rc_rock/main_rock.r_max
offset1_rock = rc1_rock/main1_rock.r_max


nbins = 5
plt.figure()
plt.xlabel('$\log M$')
plt.ylabel('$S = c/a$')
# make_plot2(lgM,S_rock,nbins=nbins,color='C1',error=True,label='rockstar - rockstar')
make_plot2(lgM,S_rock_2,nbins=nbins,color='C0',error=True,label='rockstar - rockstar2')
# make_plot2(lgM_sh,S_sh,nbins=nbins,color='k',error=True,label='subhalos - new par')
# make_plot2(main_fof.lgM-0.2,S_fof,nbins=nbins,color='C2',error=True,label='FOF - new par')
make_plot2(lgM,S_rock2,nbins=nbins,color='C3',error=True,label='rockstar - new par sinit')
make_plot2(lgM[mv2],S_v2[mv2],nbins=nbins,color='C1',error=True,label='rockstar - new par rock')
make_plot2(lgM[mit],S_it[mit],nbins=nbins,color='C7',error=True,label='rockstar - new par it')
# make_plot2(halos.lgM[mask]-0.2,halos.s[mask],nbins=nbins,color='C7',error=True,label='FOF - new par (MICE)')
plt.legend()
# make_plot2(lgM1,S1_rock,nbins=nbins,color='C1',error=True,label='rockstar',lt='--')
make_plot2(lgM1,S1_rock_2,nbins=nbins,color='C0',error=True,label='rockstar',lt='--')
# make_plot2(lgM1_sh,S1_sh,nbins=nbins,color='k',error=True,label='subhalos - new par',lt='--')
# make_plot2(main_fof.lgM-0.2,S1_fof,nbins=nbins,color='C2',error=True,label='fof',lt='--')
make_plot2(lgM1,S1_rock2,nbins=nbins,color='C3',error=True,label='fof',lt='--')
make_plot2(lgM1[mv21],S1_v2[mv21],nbins=nbins,color='C1',error=True,label='fof',lt='--')
make_plot2(lgM1[mit1],S1_it[mit1],nbins=nbins,color='C7',error=True,label='fof',lt='--')
plt.axis([13.4,14.8,0.15,0.8])
plt.savefig('../S_lM.png')

plt.figure()
plt.xlabel('$\log M$')
plt.ylabel('$S_r = c/a$')
# make_plot2(lgM,S_rock,nbins=nbins,color='C1',error=True,label='rockstar - rockstar')
make_plot2(lgM,S_rock_2,nbins=nbins,color='C0',error=True,label='rockstar - rockstar2')
# make_plot2(lgM_sh,S_sh,nbins=nbins,color='k',error=True,label='subhalos - new par')
# make_plot2(main_fof.lgM-0.2,S_fof,nbins=nbins,color='C2',error=True,label='FOF - new par')
make_plot2(lgM,Sr_rock2,nbins=nbins,color='C3',error=True,label='rockstar - new par sinit')
make_plot2(lgM[mv2r],Sr_v2[mv2r],nbins=nbins,color='C1',error=True,label='rockstar - new par rock')
make_plot2(lgM[mitr],Sr_it[mitr],nbins=nbins,color='C7',error=True,label='rockstar - new par it')
# make_plot2(halos.lgM[mask]-0.2,halos.s[mask],nbins=nbins,color='C7',error=True,label='FOF - new par (MICE)')
plt.legend()
# make_plot2(lgM1,S1_rock,nbins=nbins,color='C1',error=True,label='rockstar',lt='--')
make_plot2(lgM1,S1_rock_2,nbins=nbins,color='C0',error=True,label='rockstar',lt='--')
# make_plot2(lgM1_sh,S1_sh,nbins=nbins,color='k',error=True,label='subhalos - new par',lt='--')
# make_plot2(main_fof.lgM-0.2,S1_fof,nbins=nbins,color='C2',error=True,label='fof',lt='--')
make_plot2(lgM1,S1r_rock2,nbins=nbins,color='C3',error=True,label='rockstar - new par',lt='--')
make_plot2(lgM1[mv2r1],S1r_v2[mv2r1],nbins=nbins,color='C1',error=True,label='fof',lt='--')
make_plot2(lgM1[mitr1],S1r_it[mitr1],nbins=nbins,color='C7',error=True,label='fof',lt='--')
plt.axis([13.4,14.8,0.15,0.8])
plt.savefig('../Sr_lM.png')

'''

plt.figure()
plt.xlabel('$\log M$')
plt.ylabel('$S_r = c/a$')
make_plot2(lgM,S_rock,nbins=nbins,color='C1',error=True,label='rockstar - rockstar')
make_plot2(lgM,S_rock_2,nbins=nbins,color='C0',error=True,label='rockstar - rockstar2')
make_plot2(lgM_sh,Sr_sh,nbins=nbins,color='k',error=True,label='subhalos - new par')
make_plot2(main_fof.lgM-0.2,Sr_fof,nbins=nbins,color='C2',error=True,label='FOF - new par')
make_plot2(lgM,Sr_rock2,nbins=nbins,color='C3',error=True,label='rockstar - new par')
make_plot2(halos.lgM[mask]-0.2,halos.sr[mask],nbins=nbins,color='C7',error=True,label='FOF - new par (MICE)')
plt.legend()
make_plot2(lgM1,S1_rock,nbins=nbins,color='C1',error=True,label='rockstar',lt='--')
make_plot2(lgM1,S1_rock_2,nbins=nbins,color='C0',error=True,label='rockstar',lt='--')
make_plot2(main_fof.lgM-0.2,S1r_fof,nbins=nbins,color='C2',error=True,label='fof',lt='--')
make_plot2(lgM1,S1r_rock2,nbins=nbins,color='C3',error=True,label='fof',lt='--')
make_plot2(lgM1_sh,S1r_sh,nbins=nbins,color='k',error=True,label='subhalos - new par',lt='--')
plt.axis([13.4,14.8,0.15,0.8])
plt.savefig('../Sr_lM.png')

plt.figure()
plt.xlabel('$\log M$')
plt.ylabel('$T$')
make_plot2(lgM,T_rock,nbins=nbins,color='C1',error=True,label='rockstar - rockstar')
make_plot2(lgM,T_rock_2,nbins=nbins,color='C0',error=True,label='rockstar - rockstar2')
make_plot2(main_fof.lgM-0.2,T_fof,nbins=nbins,color='C2',error=True,label='FOF - new par')
make_plot2(lgM,T_rock2,nbins=nbins,color='C3',error=True,label='rockstar - new par')
make_plot2(halos.lgM[mask]-0.2,((1.-halos.q**2)/(1.-halos.s**2))[mask],nbins=nbins,color='C7',error=True,label='FOF - new par (MICE)')
make_plot2(lgM_sh,T_sh,nbins=nbins,color='k',error=True,label='subhalos - new par')
plt.legend()
make_plot2(lgM1,T1_rock,nbins=nbins,color='C1',error=True,label='rockstar',lt='--')
make_plot2(lgM1,T1_rock_2,nbins=nbins,color='C0',error=True,label='rockstar',lt='--')
make_plot2(main1_fof.lgM-0.2,T1_fof,nbins=nbins,color='C2',error=True,label='fof',lt='--')
make_plot2(lgM1,T1_rock2,nbins=nbins,color='C3',error=True,label='fof',lt='--')
make_plot2(lgM1_sh,T1_sh,nbins=nbins,color='k',error=True,label='subhalos - new par',lt='--')
plt.axis([13.4,14.8,0.1,1.])
plt.savefig('../T_lM.png')

plt.figure()
plt.xlabel('$\log M$')
plt.ylabel('$T_r$')
make_plot2(lgM,T_rock,nbins=nbins,color='C1',error=True,label='rockstar - rockstar')
make_plot2(lgM,T_rock_2,nbins=nbins,color='C0',error=True,label='rockstar - rockstar2')
make_plot2(main_fof.lgM-0.2,Tr_fof,nbins=nbins,color='C2',error=True,label='FOF - new par')
make_plot2(lgM,Tr_rock2,nbins=nbins,color='C3',error=True,label='rockstar - new par')
make_plot2(halos.lgM[mask]-0.2,((1.-halos.q**2)/(1.-halos.s**2))[mask],nbins=nbins,color='C7',error=True,label='FOF - new par (MICE)')
make_plot2(lgM_sh,Tr_sh,nbins=nbins,color='k',error=True,label='subhalos - new par')
plt.legend()
make_plot2(lgM1,T1_rock,nbins=nbins,color='C1',error=True,label='rockstar',lt='--')
make_plot2(lgM1,T1_rock_2,nbins=nbins,color='C0',error=True,label='rockstar',lt='--')
make_plot2(main_fof.lgM-0.2,T1r_fof,nbins=nbins,color='C2',error=True,label='fof',lt='--')
make_plot2(lgM1,T1r_rock2,nbins=nbins,color='C3',error=True,label='fof',lt='--')
make_plot2(lgM1_sh,T1r_sh,nbins=nbins,color='k',error=True,label='subhalos - new par',lt='--')
plt.axis([13.4,14.8,0.1,1.])
plt.savefig('../Tr_lM.png')


# ONLY RELAXED

doff = 0.1

plt.figure()
plt.xlabel('$\log M$')
plt.ylabel('$S = c/a$')
make_plot2(lgM[offset_rock<doff],S_rock[offset_rock<doff],nbins=nbins,color='C1',error=True,label='rockstar - rockstar')
make_plot2(lgM[offset_rock<doff],S_rock_2[offset_rock<doff],nbins=nbins,color='C0',error=True,label='rockstar - rockstar2')
make_plot2(main_fof.lgM[offset_fof<doff]-0.2,S_fof[offset_fof<doff],nbins=nbins,color='C2',error=True,label='FOF - new par')
make_plot2(lgM[offset_rock<doff],S_rock2[offset_rock<doff],nbins=nbins,color='C3',error=True,label='rockstar - new par')
make_plot2(halos.lgM[mask*(halos.offset<doff)]-0.2,halos.s[mask*(halos.offset<doff)],nbins=nbins,color='C7',error=True,label='FOF - new par (MICE)')
plt.legend()
make_plot2(lgM1[offset1_rock<doff],S1_rock[offset1_rock<doff],nbins=nbins,color='C1',error=True,label='rockstar',lt='--')
make_plot2(lgM1[offset1_rock<doff],S1_rock_2[offset1_rock<doff],nbins=nbins,color='C0',error=True,label='rockstar',lt='--')
make_plot2(main1_fof.lgM[offset1_fof<doff]-0.2,S1_fof[offset1_fof<doff],nbins=nbins,color='C2',error=True,label='fof',lt='--')
make_plot2(lgM1[offset1_rock<doff],S1_rock2[offset1_rock<doff],nbins=nbins,color='C3',error=True,label='fof',lt='--')
plt.axis([13.4,14.8,0.15,0.8])
plt.savefig('../S_lM_relaxed.png')

plt.figure()
plt.xlabel('$\log M$')
plt.ylabel('$S_r = c/a$')
make_plot2(lgM[offset_rock<doff],S_rock[offset_rock<doff],nbins=nbins,color='C1',error=True,label='rockstar - rockstar')
make_plot2(lgM[offset_rock<doff],S_rock_2[offset_rock<doff],nbins=nbins,color='C0',error=True,label='rockstar - rockstar2')
make_plot2(main_fof.lgM[offset_fof<doff]-0.2,Sr_fof[offset_fof<doff],nbins=nbins,color='C2',error=True,label='FOF - new par')
make_plot2(lgM[offset_rock<doff],Sr_rock2[offset_rock<doff],nbins=nbins,color='C3',error=True,label='rockstar - new par')
make_plot2(halos.lgM[mask*(halos.offset<doff)]-0.2,halos.sr[mask*(halos.offset<doff)],nbins=nbins,color='C7',error=True,label='FOF - new par (MICE)')
plt.legend()
make_plot2(lgM1[offset1_rock<doff],S1_rock[offset1_rock<doff],nbins=nbins,color='C1',error=True,label='rockstar',lt='--')
make_plot2(lgM1[offset1_rock<doff],S1_rock_2[offset1_rock<doff],nbins=nbins,color='C0',error=True,label='rockstar',lt='--')
make_plot2(main1_fof.lgM[offset1_fof<doff]-0.2,S1r_fof[offset1_fof<doff],nbins=nbins,color='C2',error=True,label='fof',lt='--')
make_plot2(lgM1[offset1_rock<doff],S1r_rock2[offset1_rock<doff],nbins=nbins,color='C3',error=True,label='fof',lt='--')
plt.axis([13.4,14.8,0.15,0.8])
plt.savefig('../Sr_lM_relaxed.png')

plt.figure()
plt.xlabel('$\log M$')
plt.ylabel('$T$')
make_plot2(lgM[offset_rock<doff],T_rock[offset_rock<doff],nbins=nbins,color='C1',error=True,label='rockstar - rockstar')
make_plot2(lgM[offset_rock<doff],T_rock_2[offset_rock<doff],nbins=nbins,color='C0',error=True,label='rockstar - rockstar2')
make_plot2(main_fof.lgM[offset_fof<doff]-0.2,T_fof[offset_fof<doff],nbins=nbins,color='C2',error=True,label='FOF - new par')
make_plot2(lgM[offset_rock<doff]-0.2,T_rock2[offset_rock<doff],nbins=nbins,color='C3',error=True,label='rockstar - new par')
make_plot2(halos.lgM[mask]-0.2,((1.-halos.q**2)/(1.-halos.s**2))[mask],nbins=nbins,color='C7',error=True,label='FOF - new par (MICE)')
plt.legend()
make_plot2(lgM1[offset1_rock<doff],T1_rock[offset1_rock<doff],nbins=nbins,color='C1',error=True,label='rockstar',lt='--')
make_plot2(lgM1[offset1_rock<doff],T1_rock_2[offset1_rock<doff],nbins=nbins,color='C0',error=True,label='rockstar',lt='--')
make_plot2(main1_fof.lgM[offset1_fof<doff]-0.2,T1_fof[offset1_fof<doff],nbins=nbins,color='C2',error=True,label='fof',lt='--')
make_plot2(lgM1[offset1_rock<doff],T1_rock2[offset1_rock<doff],nbins=nbins,color='C3',error=True,label='fof',lt='--')
plt.axis([13.4,14.8,0.1,1.])
plt.savefig('../T_lM_relaxed.png')

plt.figure()
plt.xlabel('$\log M$')
plt.ylabel('$T_r$')
make_plot2(lgM[offset_rock<doff],T_rock[offset_rock<doff],nbins=nbins,color='C1',error=True,label='rockstar - rockstar')
make_plot2(lgM[offset_rock<doff],T_rock_2[offset_rock<doff],nbins=nbins,color='C0',error=True,label='rockstar - rockstar2')
make_plot2(main_fof.lgM[offset_fof<doff]-0.2,Tr_fof[offset_fof<doff],nbins=nbins,color='C2',error=True,label='FOF - new par')
make_plot2(lgM[offset_rock<doff],Tr_rock2[offset_rock<doff],nbins=nbins,color='C3',error=True,label='rockstar - new par')
make_plot2(halos.lgM[mask]-0.2,((1.-halos.q**2)/(1.-halos.s**2))[mask],nbins=nbins,color='C7',error=True,label='FOF - new par (MICE)')
plt.legend()
make_plot2(lgM1[offset1_rock<doff],T1_rock[offset1_rock<doff],nbins=nbins,color='C1',error=True,label='rockstar',lt='--')
make_plot2(lgM1[offset1_rock<doff],T1_rock_2[offset1_rock<doff],nbins=nbins,color='C0',error=True,label='rockstar',lt='--')
make_plot2(main1_fof.lgM[offset1_fof<doff]-0.2,T1r_fof[offset1_fof<doff],nbins=nbins,color='C2',error=True,label='fof',lt='--')
make_plot2(lgM1[offset1_rock<doff],T1r_rock2[offset1_rock<doff],nbins=nbins,color='C3',error=True,label='fof',lt='--')
plt.axis([13.4,14.8,0.1,1.])
plt.savefig('../Tr_lM_relaxed.png')


plt.figure()
plt.plot(lgM_sh,nhalos,'.',label='CDM')
plt.plot(lgM1_sh,nhalos1,'.',label='SIDM')
plt.legend()
plt.xlabel('$\log M_{vir}$')
plt.ylabel('$N_{sh}$')
plt.savefig('../nhalos_lgM.png')

plt.figure()
plt.plot(nhalos,S_sh,'.',label='CDM')
plt.plot(nhalos1,S1_sh,'.',label='SIDM')
plt.legend()
plt.ylabel('$S$')
plt.xlabel('$N_{sh}$')
plt.savefig('../nhalos_S.png')

minhalos = 120.
plt.figure()
plt.hist(S_sh[nhalos>minhalos],np.linspace(0.4,1.,15),histtype='step')
plt.hist(S1_sh[nhalos1>minhalos],np.linspace(0.4,1.,15),histtype='step')

minmass = 14.5
plt.figure()
plt.hist(S_sh[lgM_sh>minmass],np.linspace(0.4,1.,15),histtype='step')
plt.hist(S1_sh[lgM1_sh>minmass],np.linspace(0.4,1.,15),histtype='step')



mlim = 14.2

plt.figure()
plt.xlabel('$\log M_{FOF}$')
plt.ylabel('$S = c/a$')
make_plot2(halos.lgM[mask*(halos.offset<doff)],halos.s[mask*(halos.offset<doff)],nbins=nbins,color='C7',error=True,label='MICE')
make_plot2(main_fof.lgM[offset1_fof<doff],S1_fof[offset1_fof<doff],nbins=nbins,color='C2',error=True,label='fof',lt='--')
make_plot2(main_fof.lgM[offset_fof<doff],S_fof[offset_fof<doff],nbins=nbins,color='C2',error=True,label='fof')
plt.legend()

plt.figure()
plt.xlabel('$\log M_{FOF}$')
plt.ylabel('$S_r = c_r/a_r$')
make_plot2(halos.lgM[mask*(halos.offset<doff)],halos.sr[mask*(halos.offset<doff)],nbins=nbins,color='C7',error=True,label='MICE')
make_plot2(main_fof.lgM[offset1_fof<doff],S1r_fof[offset1_fof<doff],nbins=nbins,color='C2',error=True,label='fof',lt='--')
make_plot2(main_fof.lgM[offset_fof<doff],Sr_fof[offset_fof<doff],nbins=nbins,color='C2',error=True,label='fof')
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
'''
