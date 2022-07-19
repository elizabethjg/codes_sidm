import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import pandas as pd
from scipy import stats
from astropy.io import fits
from binned_plots import make_plot2

# halos = fits.open('/home/elizabeth/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/HS-lensing/HALO_Props_MICE.fits')[1].data        

zs = ['z0','z51','z96']


z = zs[0]
    
main_fof = pd.read_csv('../halo_props/halo_props_fof_cdm_'+z+'_main.csv.bz2')
pro_fof = np.loadtxt('../halo_props/halo_props_fof_cdm_'+z+'_pro.csv.bz2',skiprows=1,delimiter=',')

main1_fof = pd.read_csv('../halo_props/halo_props_fof_sidm1_'+z+'_main.csv.bz2')
pro1_fof = np.loadtxt('../halo_props/halo_props_fof_sidm1_'+z+'_pro.csv.bz2',skiprows=1,delimiter=',')

nrings = 25
mp = 0.013398587e10

r     = pro_fof[:,2:2+nrings]/1.e3
rho   = pro_fof[:,2+nrings:2+2*nrings]
rho_E = pro_fof[:,2+2*nrings:2+3*nrings]
S     = pro_fof[:,2+3*nrings:2+4*nrings]
S_E   = pro_fof[:,2+4*nrings:]

rho1   = pro1_fof[:,2+nrings:2+2*nrings]
rho_E1 = pro1_fof[:,2+2*nrings:2+3*nrings]
S1     = pro1_fof[:,2+3*nrings:2+4*nrings]
S_E1   = pro1_fof[:,2+4*nrings:]

plt.figure()
for j in range(len(r)):
    plt.plot(r[j],rho[j],'C7',alpha=0.3)
    plt.plot(r[j],rho1[j],'C8',alpha=0.3)    
plt.plot(r[0],np.average(rho,axis=0),'k',lw=2)
plt.plot(r[0],np.average(rho1,axis=0),'C2',lw=2)
plt.xlabel(r'$r$ [Mpc/h]')
plt.ylabel(r'$\rho$')
plt.loglog()


plt.figure()
for j in range(len(r)):
    plt.plot(r[j],rho_E[j],'C7',alpha=0.3)
    plt.plot(r[j],rho_E1[j],'C8',alpha=0.3)    
plt.plot(r[0],np.average(rho_E,axis=0),'k',lw=2)
plt.plot(r[0],np.average(rho_E1,axis=0),'C2',lw=2)
plt.xlabel(r'$r$ [Mpc/h]')
plt.ylabel(r'$\rho_E$')
plt.loglog()

plt.figure()
for j in range(len(r)):
    plt.plot(r[j],S[j],'C7',alpha=0.3)
    plt.plot(r[j],S1[j],'C8',alpha=0.3)    
plt.plot(r[0],np.average(S,axis=0),'k',lw=2)
plt.plot(r[0],np.average(S1,axis=0),'C2',lw=2)
plt.xlabel(r'$R$ [Mpc/h]')
plt.ylabel(r'$\Sigma$')
plt.loglog()


plt.figure()
for j in range(len(r)):
    plt.plot(r[j],S_E[j],'C7',alpha=0.3)
    plt.plot(r[j],S_E1[j],'C8',alpha=0.3)    
plt.plot(r[0],np.average(S_E,axis=0),'k',lw=2)
plt.plot(r[0],np.average(S_E1,axis=0),'C2',lw=2)
plt.xlabel(r'$R$ [Mpc/h]')
plt.ylabel(r'$\Sigma_E$')
plt.loglog()
