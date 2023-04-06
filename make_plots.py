import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import pylab
from astropy.io import fits
import corner
from matplotlib import cm
from binned_plots import make_plot2
from models_profiles import *
import matplotlib.pyplot as plt

folder = '../profiles3/'

def plot_zdist():
    
    S = fits.open('../MICE_sources_HSN_withextra.fits')[1].data
    
    
    f, ax = plt.subplots(1,1, figsize=(5,4),sharex = True,sharey=True)
    plt.hist(S.z_cgal_v,100,histtype='step',color='C3',lw=2.5)
    plt.xlabel('$z$')
    plt.ylabel('N [millions]')
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels([x/1.e6 for x in current_values])
    f.savefig('../zdist.pdf',bbox_inches='tight')



def plot_q_dist(DM,SIDM,ax,method=''):
    
    import seaborn as sns    
    

    qdm = eval('DM.q2d'+method)
    qsidm = eval('SIDM.q2d'+method)
                
    sns.kdeplot(qdm.byteswap().newbyteorder(),color='C7',ax=ax,label='CDM',clip=(0,1),bw_adjust=0.6)
    ax.axvline(np.mean(qdm),color='C7')
    ax.axvline(DM.q1h_2g,color='k',label='1h')
    ax.axvline(DM.q2h_2g,color='k',label='2h',ls='--')
    
    sns.kdeplot(qsidm.byteswap().newbyteorder(),color='C6',ax=ax,label='SIDM',clip=(0,1),bw_adjust=0.6)
    ax.axvline(np.mean(qsidm),color='C6')
    ax.axvline(SIDM.q1h_2g,color='C3')
    ax.axvline(SIDM.q2h_2g,color='C3',ls='--')
    ax.set_xlim(0.1,1)

    
 
def plt_profile_fitted_final(DM,SIDM,RIN,ROUT,axx3,fittype='_2h_2q'):

    ax,ax1,ax2 = axx3
            
    ##############    

    ax.plot(DM.r,DM.DS_T,'C7')
    # ax.plot(rplot,DS1h,'C1',label='1h')
    # ax.plot(rplot,DS2h,'C8',label='2h')
    ax.plot(DM.r,DM.DS_fit,'C3',label='1h+2h')
    ax.fill_between(DM.r,DM.DS_T+DM.e_DS_T,DM.DS_T-DM.e_DS_T,color='C7',alpha=0.4)
    ax.plot(SIDM.r,SIDM.DS_T,'C6')
    # ax.plot(rplot,DS1h,'C1',label='1h')
    # ax.plot(rplot,DS2h,'C8',label='2h')
    ax.plot(SIDM.r,SIDM.DS_fit,'C3--',label='1h+2h')
    ax.fill_between(SIDM.r,SIDM.DS_T+SIDM.e_DS_T,SIDM.DS_T-SIDM.e_DS_T,color='C6',alpha=0.4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\Delta\Sigma [h M_\odot/pc^2]$',labelpad=2)
    ax.set_xlabel('r [$h^{-1}$ Mpc]')
    ax.set_ylim(0.5,200)
    ax.set_xlim(0.1,5)
    ax.xaxis.set_ticks([0.1,1,5])
    ax.set_xticklabels([0.1,1,5])
    ax.yaxis.set_ticks([1,10,100])
    ax.set_yticklabels([1,10,100])
    ax.axvline(RIN/1000.,color='k',ls=':')
    ax.axvline(ROUT/1000.,color='k',ls=':')
    # ax.legend(loc=3,frameon=False,ncol=2)
    
    
    ax1.plot(DM.r,DM.GT,'C7',label='CDM')
    ax1.plot(DM.r,DM.GT1h+DM.GT2h,'C3')
    ax1.plot(DM.r,DM.GT1h,'C1')
    ax1.plot(DM.r,DM.GT2h,'C8')
    ax1.plot(SIDM.r,SIDM.GT,'C6--',label='SIDM')    
    ax1.plot(SIDM.r,SIDM.GT1h+SIDM.GT2h,'C3--')
    ax1.plot(SIDM.r,SIDM.GT1h,'C1--')
    ax1.plot(SIDM.r,SIDM.GT2h,'C8--')
    ax1.fill_between(DM.r,DM.GT+DM.e_GT,DM.GT-DM.e_GT,color='C7',alpha=0.4)
    ax1.fill_between(SIDM.r,SIDM.GT+SIDM.e_GT,DM.GT-SIDM.e_GT,color='C6',alpha=0.4)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('r [$h^{-1}$ Mpc]')
    ax1.set_ylabel(r'$\Gamma_T [h M_\odot/pc^2]$',labelpad=1.2)
    ax1.set_ylim(0.5,100)
    ax1.set_xlim(0.1,5)
    ax1.xaxis.set_ticks([0.1,1,5])
    ax1.set_xticklabels([0.1,1,5])
    ax1.yaxis.set_ticks([1,10,100])
    ax1.set_yticklabels([1,10,100])
    ax1.axvline(RIN/1000.,color='k',ls=':')
    ax1.axvline(ROUT/1000.,color='k',ls=':')
        
    ax2.plot([0,10],[0,0],'k')
    ax2.plot(DM.r,DM.GX,'C7',label='standard')
    ax2.plot(DM.r,DM.GX1h+DM.GX2h,'C3')
    ax2.plot(DM.r,DM.GX1h,'C1')
    ax2.plot(DM.r,DM.GX2h,'C8')
    ax2.plot(SIDM.r,SIDM.GX,'C6--',label='reduced')    
    ax2.plot(SIDM.r,SIDM.GX1h+SIDM.GX2h,'C3--')
    ax2.plot(SIDM.r,SIDM.GX1h,'C1--')
    ax2.plot(SIDM.r,SIDM.GX2h,'C8--')
    # ax2.legend(loc=3,frameon=False)
    ax2.fill_between(DM.r,DM.GX+DM.e_GX,DM.GX-DM.e_GX,color='C7',alpha=0.4)
    ax2.fill_between(SIDM.r,SIDM.GX+SIDM.e_GX,SIDM.GX-SIDM.e_GX,color='C6',alpha=0.4)
    ax2.set_xlabel('r [$h^{-1}$ Mpc]')
    ax2.set_ylabel(r'$\Gamma_\times [h M_\odot/pc^2]$',labelpad=1.2)
    ax2.set_xscale('log')
    ax2.set_xlim(0.1,5)
    ax2.set_ylim(-16,17)
    ax2.xaxis.set_ticks([0.1,1,5])
    ax2.set_xticklabels([0.1,1,5])

