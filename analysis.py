import numpy as np
import pandas as pd
from scipy import stats
from astropy.io import fits

# halos = fits.open('/home/elizabeth/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/HS-lensing/HALO_Props_MICE.fits')[1].data        
halos = fits.open('/home/elizabeth/Documentos/proyectos/HALO-SHAPE/MICE/HS-lensing/HALO_Props_MICE.fits')[1].data        

z = 'z0'
# z = 'z96'
# z = 'z51'
mask = (halos.z < 0.1)

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
plt.figure()
plt.plot(S_rock,S,'.',label='CDM')
plt.plot(S_rock1,S1,'x',label='SIDM')
plt.plot([0,1],[0,1],'C7--')
plt.xlabel('c/a - ROCKSTAR')
plt.ylabel('c/a - new params')
plt.legend()
plt.savefig('../s_rock_new_'+z+'.png')

plt.figure()
plt.plot(S_rock,S_rock1,'.',label='rock')
plt.plot(S,S1,'x',label='new params')
plt.plot([0,1],[0,1],'C7--')
plt.xlabel('c/a - CDM')
plt.ylabel('c/a - SIDM')
plt.legend()
plt.savefig('../q_cdm_sidm_'+z+'.png')

plt.figure()
plt.plot(Q_rock,Q,'.',label='CDM')
plt.plot(Q_rock1,Q1,'x',label='SIDM')
plt.plot([0,1],[0,1],'C7--')
plt.xlabel('b/a - ROCKSTAR')
plt.ylabel('b/a - new params')
plt.legend()
plt.savefig('../s_rock_new_'+z+'.png')

plt.figure()
plt.plot(Q_rock,Q_rock1,'.',label='rock')
plt.plot(Q,Q1,'x',label='new params')
plt.plot([0,1],[0,1],'C7--')
plt.xlabel('b/a - CDM')
plt.ylabel('b/a - SIDM')
plt.legend()
plt.savefig('../q_cdm_sidm_'+z+'.png')

plt.figure()
make_plot2(main.lgM,q1,nbins=4,color='C1',error=True,label='SIDM')
make_plot2(main.lgM,q,nbins=4,color='C0',error=True,label='CDM')
# make_plot2(halos.lgM[mask],halos.q2d[mask],nbins=4,color='C7',error=True,label='MICE')
plt.legend()
plt.xlabel('$\log(M_{vir})$')
plt.ylabel('$q_{2D}$')
plt.savefig('../mass_q2d_'+z+'.png')

plt.figure()
make_plot2(main.lgM,S_rock,nbins=4,color='C1',error=True,label='SIDM')
make_plot2(main.lgM,S_rock1,nbins=4,color='C0',error=True,label='CDM')
# make_plot2(halos.lgM[mask],halos.s[mask],nbins=4,color='C7',error=True,label='MICE')
plt.legend()
plt.xlabel('$\log(M_{vir})$')
plt.ylabel('$c/a$')
plt.savefig('../mass_Srock_'+z+'.png')

plt.figure()
make_plot2(main.lgM,S,nbins=4,color='C1',error=True,label='SIDM')
make_plot2(main.lgM,S1,nbins=4,color='C0',error=True,label='CDM')
# make_plot2(halos.lgM[mask],halos.s[mask],nbins=4,color='C7',error=True,label='MICE')
plt.legend()
plt.xlabel('$\log(M_{vir})$')
plt.ylabel('$c/a$')
plt.savefig('../mass_S_'+z+'.png')
