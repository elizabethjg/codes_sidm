import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def q_75(y):
    return np.quantile(y, 0.75)

def q_25(y):
    return np.quantile(y, 0.25)

def binned(x,y,nbins=10):

    bined = stats.binned_statistic(x,y,statistic='mean', bins=nbins)
    x_b = 0.5*(bined.bin_edges[:-1] + bined.bin_edges[1:])
    ymean     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic='median', bins=nbins)
    x_b = 0.5*(bined.bin_edges[:-1] + bined.bin_edges[1:])
    q50     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic=q_25, bins=nbins)
    q25     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic=q_75, bins=nbins)
    q75     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic='count', bins=nbins)
    N     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic='std', bins=nbins)
    sigma = bined.statistic

    dig   = np.digitize(x,bined.bin_edges)
    mz    = np.ones(len(x))
    for j in range(nbins):
        mbin = dig == (j+1)
        mz[mbin] = y[mbin] >= q50[j]
    mz = mz.astype(bool)
    return x_b,q50,q25,q75,mz,ymean,sigma/np.sqrt(N)

def make_plot2(X,Y,color='C0',nbins=20,plt=plt,label='',error = False,lw=1,lt='-'):
    x,q50,q25,q75,nada,ymean,ers = binned(X,Y,nbins)
    if error:
        plt.plot(x,ymean,lt,color=color,label=label,lw=lw)
        plt.fill_between(x,ymean+ers,ymean-ers,color=color,alpha=0.2)
    else:
        plt.plot(x,q50,lt,color=color,label=label,lw=lw)
        plt.fill_between(x,q75,q25,color=color,alpha=0.2)

def load_parameters(rock, lgM, flag):

  if flag == 'standard':

    S_rock = rock['c3D']/rock['a3D']
    Q_rock = rock['c3D']/rock['a3D']
    q_rock = rock['b2D']/rock['a2D']
    T_rock = (rock['a3D']**2 - rock['b3D']**2)/(rock['a3D']**2 - rock['c3D']**2)

  elif flag == 'reducido':

    S_rock = rock['c3Dr']/rock['a3Dr']
    Q_rock = rock['c3Dr']/rock['a3Dr']
    q_rock = rock['b2Dr']/rock['a2Dr']
    T_rock = (rock['a3Dr']**2 - rock['b3Dr']**2)/(rock['a3Dr']**2 - rock['c3Dr']**2)

  elif flag == 'iterative_standard':

    S_rock = rock['c3D_it']/rock['a3D_it']
    Q_rock = rock['c3D_it']/rock['a3D_it']
    q_rock = rock['b2D_it']/rock['a2D_it']
    T_rock = (rock['a3D_it']**2 - rock['b3D_it']**2)/(rock['a3D_it']**2 - rock['c3D_it']**2)

  elif flag == 'iterative_reducido':

    S_rock = rock['c3Dr_it']/rock['a3Dr_it']
    Q_rock = rock['c3Dr_it']/rock['a3Dr_it']
    q_rock = rock['b2Dr_it']/rock['a2Dr_it']
    T_rock = (rock['a3Dr_it']**2 - rock['b3Dr_it']**2)/(rock['a3Dr_it']**2 - rock['c3Dr_it']**2)

  else:
    print("No existe el campo '%s'\n" % flag)
    assert(0)

  mask  = ~np.isnan(rock["a3Dr"])*(S_rock > 0.)

  return lgM[mask], S_rock[mask], Q_rock[mask], q_rock[mask], T_rock[mask]




zs = ['z0','z51','z96']
#mask = (halos.z < 0.07)
z = zs[0]

#def mask_border(x0,y0,z0):
#    mask = (x0 > 2.)*(x0 < 118.)*(y0 > 2.)*(y0 < 118.)*(z0 > 2.)*(z0 < 118.)
#    return mask

#halo_props_match_cdm_z0_rock2.csv.bz2
#halo_props_match_sidm1_z0_rock2.csv.bz2
#halo_propsv2_rock2_match_cdm_z0_main.csv.bz2
#halo_propsv2_rock2_match_cdm_z0_pro.csv.bz2
#halo_propsv2_rock2_match_sidm1_z0_main.csv.bz2
#halo_propsv2_rock2_match_sidm1_z0_pro.csv.bz2

folder      = "../halo_props"
rock_cdm    = pd.read_csv(folder + '/halo_props_match_cdm_'+z+'_rock2.csv.bz2')
rock_sidm1  = pd.read_csv(folder + '/halo_props_match_sidm1_'+z+'_rock2.csv.bz2')

#mrock     = mask_border(rock.x0,rock.y0,rock.z0)
#mrock_sh  = mask_border(rock_sh.x0,rock_sh.y0,rock_sh.z0)
#mrock1    = mask_border(rock1.x0,rock1.y0,rock.z0)
#mrock1_sh = mask_border(rock1_sh.x0,rock1_sh.y0,rock1_sh.z0)
#
#print('rock',len(mrock),mrock.sum())
#print('rock_sh',len(mrock_sh),mrock_sh.sum())
#
#print('rock1',len(mrock1),mrock1.sum())
#print('rock1_sh',len(mrock1_sh),mrock1_sh.sum())

#rock       =  rock[mrock]    
#rock_sh    =  rock_sh[mrock_sh] 
#rock1      =  rock1[mrock1]   
#rock1_sh   =  rock1_sh[mrock1_sh]

print(rock_cdm)
print(len(rock_cdm))

main_cdm_v2   = pd.read_csv(folder + '/halo_propsv2_rock2_match_cdm_'+z+'_main.csv.bz2')
main_sidm1_v2 = pd.read_csv(folder + '/halo_propsv2_rock2_match_sidm1_'+z+'_main.csv.bz2')

#nhalos = rock_sh['N_sh']
#nhalos1 = rock1_sh['N_sh']

# LOAD PARAMETERS
lg10M_cdm   = np.log10(rock_cdm["Mvir"])
lg10M_sidm1 = np.log10(rock_sidm1["Mvir"])

lgM_cdm, S_cdm, Q_cdm, q_cdm, T_cdm                 = load_parameters(main_cdm_v2, lg10M_cdm, flag='standard')
lgMr_cdm, Sr_cdm, Qr_cdm, qr_cdm, Tr_cdm             = load_parameters(main_cdm_v2, lg10M_cdm, flag='reducido')
lgM_it_cdm, S_it_cdm, Qr_it_cdm, q_it_cdm, T_it_cdm    = load_parameters(main_cdm_v2, lg10M_cdm, flag='iterative_standard')
lgMr_it_cdm, Sr_it_cdm, Qr_it_cdm, qr_it_cdm, Tr_it_cdm = load_parameters(main_cdm_v2, lg10M_cdm, flag='iterative_reducido')

lgM_sidm1, S_sidm1, Q_sidm1, q_sidm1, T_sidm1                 = load_parameters(main_sidm1_v2, lg10M_sidm1, flag='standard')
lgMr_sidm1, Sr_sidm1, Qr_sidm1, qr_sidm1, Tr_sidm1             = load_parameters(main_sidm1_v2, lg10M_sidm1, flag='reducido')
lgM_it_sidm1, S_it_sidm1, Qr_it_sidm1, q_it_sidm1, T_it_sidm1    = load_parameters(main_sidm1_v2, lg10M_sidm1, flag='iterative_standard')
lgMr_it_sidm1, Sr_it_sidm1, Qr_it_sidm1, qr_it_sidm1, Tr_it_sidm1 = load_parameters(main_sidm1_v2, lg10M_sidm1, flag='iterative_reducido')

#Eratio_it  = (2.*main_it.EKin/abs(main_it.EPot))
#Eratio1_it = (2.*main1_it.EKin/abs(main1_it.EPot))
#rc_rock  = np.array(np.sqrt((main_rock.xc - main_rock.xc_rc)**2 + (main_rock.yc - main_rock.yc_rc)**2 + (main_rock.zc - main_rock.zc_rc)**2))
#rc1_rock = np.array(np.sqrt((main1_rock.xc - main1_rock.xc_rc)**2 + (main1_rock.yc - main1_rock.yc_rc)**2 + (main1_rock.zc - main1_rock.zc_rc)**2))
#offset_rock  = rc_rock/main_rock.r_max
#offset1_rock = rc1_rock/main1_rock.r_max

nbins = 5
plt.figure()
plt.xlabel('$\log M$')
plt.ylabel('$S = c/a$')
# CDM
make_plot2(lgM_cdm, S_cdm, nbins=nbins, color='C0', error=True, label='CDM Standard')
make_plot2(lgM_it_cdm, S_it_cdm, nbins=nbins, color='C1', error=True, label='CDM Iterative Standard')
# SIDM1
make_plot2(lgM_sidm1, S_sidm1, nbins=nbins, color='C0', error=True, label='SIDM1 Standard', lt='--')
make_plot2(lgM_it_sidm1, S_it_sidm1, nbins=nbins, color='C1', error=True, label='SIDM1 Iterative Standard', lt='--')

plt.axis([13.4,14.8,0.15,0.8])
plt.legend(title='Rockstar Halos')
plt.savefig('S_lM.png')
plt.show()

nbins = 5
plt.figure()
plt.xlabel('$\log M$')
plt.ylabel('$S_r = c_r/a_r$')
# CDM
make_plot2(lgMr_cdm, Sr_cdm, nbins=nbins, color='C0', error=True, label='CDM Standard')
make_plot2(lgMr_it_cdm, Sr_it_cdm, nbins=nbins, color='C1', error=True, label='CDM Iterative Standard')
# SIDM1
make_plot2(lgMr_sidm1, Sr_sidm1, nbins=nbins, color='C0', error=True, label='SIDM1 Standard', lt='--')
make_plot2(lgMr_it_sidm1, Sr_it_sidm1, nbins=nbins, color='C1', error=True, label='SIDM1 Iterative Standard', lt='--')

plt.axis([13.4,14.8,0.15,0.8])
plt.legend(title='Rockstar Halos')
plt.savefig('Sr_lM.png')
plt.show()


