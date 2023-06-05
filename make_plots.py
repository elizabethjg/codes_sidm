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
    
    eq1h_dm = np.percentile(DM.mcmc_q1h_2g[1500:], [16, 84])
    eq2h_dm = np.percentile(DM.mcmc_q2h_2g[1500:], [16, 84])
    eq1h_sidm = np.percentile(SIDM.mcmc_q1h_2g[1500:], [16, 84])
    eq2h_sidm = np.percentile(SIDM.mcmc_q2h_2g[1500:], [16, 84])

    qdm = eval('DM.q2d'+method)
    qsidm = eval('SIDM.q2d'+method)
                
    sns.kdeplot(qdm.byteswap().newbyteorder(),color='C7',ax=ax,label='CDM',clip=(0,1),bw_adjust=0.6,lw=2)
    ax.axvline(np.mean(qdm),color='C7')
    ax.axvline(DM.q1h_2g,color='k',label='1h')
    ax.axvline(DM.q2h_2g,color='k',label='2h',ls='--')
    ax.axvspan(eq1h_dm[0],eq1h_dm[1],color='k',alpha=0.1)
    ax.axvspan(eq2h_dm[0],eq2h_dm[1],color='k',alpha=0.1)
    
    sns.kdeplot(qsidm.byteswap().newbyteorder(),color='C6',ax=ax,label='SIDM',clip=(0,1),bw_adjust=0.6,lw=2)
    ax.axvline(np.mean(qsidm),color='C6')
    ax.axvline(SIDM.q1h_2g,color='C3')
    ax.axvline(SIDM.q2h_2g,color='C3',ls='--')
    ax.axvspan(eq1h_sidm[0],eq1h_sidm[1],color='C3',alpha=0.1)
    ax.axvspan(eq2h_sidm[0],eq2h_sidm[1],color='C3',alpha=0.1)    
    ax.set_xlim(0.1,1)

    ax.set_xlabel('$q$')
    ax.set_ylabel('$P(q)$')

def linear_compare_q(DM,SIDM,ax,j):    
    
    ax1,ax2 = ax
    
    eq_1h_dm = np.diff(np.percentile(DM.mcmc_q1h_2g[1500:], [16, 50, 84]))
    eq_2h_dm = np.diff(np.percentile(DM.mcmc_q2h_2g[1500:], [16, 50, 84]))
    eq_1h_sidm = np.diff(np.percentile(DM.mcmc_q1h_2g[1500:], [16, 50, 84]))
    eq_2h_sidm = np.diff(np.percentile(DM.mcmc_q2h_2g[1500:], [16, 50, 84]))

    e_ratio_1h = np.array([np.sqrt((eq_1h_dm/SIDM.q1h_2g)**2 + ((eq_1h_sidm*DM.q1h_2g)/SIDM.q1h_2g**2)**2)]).T
    e_ratio_2h = np.array([np.sqrt((eq_2h_dm/SIDM.q2h_2g)**2 + ((eq_2h_sidm*DM.q2h_2g)/SIDM.q2h_2g**2)**2)]).T
    
    ax1.errorbar(np.mean(DM.q2d_it),DM.q1h_2g,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='ks',markersize=10,label='CDM - standard')
    ax1.errorbar(np.mean(DM.q2dr_it),DM.q1h_2g,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C7D',markersize=10,label='CDM - reduced')
    ax1.errorbar(np.mean(SIDM.q2d_it),SIDM.q1h_2g,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C3s',markersize=10,label='SIDM - standard')
    ax1.errorbar(np.mean(SIDM.q2dr_it),SIDM.q1h_2g,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C6D',markersize=10,label='SIDM - reduced')
    ax1.plot([0.5,0.9],[0.5,0.9],'C7--')


    ax2.errorbar(np.mean(DM.q2d_it),DM.q2h_2g,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='ks',markersize=10,label='CDM - standard')
    ax2.errorbar(np.mean(DM.q2dr_it),DM.q2h_2g,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C7D',markersize=10,label='CDM - reduced')
    ax2.errorbar(np.mean(SIDM.q2d_it),SIDM.q2h_2g,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C3s',markersize=10,label='SIDM - standard')
    ax2.errorbar(np.mean(SIDM.q2dr_it),SIDM.q2h_2g,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C6D',markersize=10,label='SIDM - reduced')
    ax2.plot([0.12,0.75],[0.12,0.75],'C7--')
                 
    # ax2.plot(j+0.2,(DM.q1h_2g/SIDM.q1h_2g)-1,'C1o',label='1h',markersize=10)
    # ax2.plot(j+0.3,(DM.q2h_2g/SIDM.q2h_2g)-1,'C8o',label='2h',markersize=10)
    ax2.set_xlabel(r'\langle q \rangle - 1$')
    ax1.set_ylabel(r'$q_{1h}$')
    ax2.set_ylabel(r'$q_{2h}$')


def compare_q(DM,SIDM,ax,j):    
    
    ax1,ax2 = ax
    
    eq_1h_dm = np.diff(np.percentile(DM.mcmc_q1h_2g[1500:], [16, 50, 84]))
    eq_2h_dm = np.diff(np.percentile(DM.mcmc_q2h_2g[1500:], [16, 50, 84]))
    eq_1h_sidm = np.diff(np.percentile(DM.mcmc_q1h_2g[1500:], [16, 50, 84]))
    eq_2h_sidm = np.diff(np.percentile(DM.mcmc_q2h_2g[1500:], [16, 50, 84]))

    e_ratio_1h = np.array([np.sqrt((eq_1h_dm/SIDM.q1h_2g)**2 + ((eq_1h_sidm*DM.q1h_2g)/SIDM.q1h_2g**2)**2)]).T
    e_ratio_2h = np.array([np.sqrt((eq_2h_dm/SIDM.q2h_2g)**2 + ((eq_2h_sidm*DM.q2h_2g)/SIDM.q2h_2g**2)**2)]).T
    
    ax1.errorbar(j+0.,(DM.q1h_2g/np.mean(DM.q2d_it))-1,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='ks',markersize=10,label='CDM - standard')
    ax1.errorbar(j+0.1,(DM.q1h_2g/np.mean(DM.q2dr_it))-1,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C7D',markersize=10,label='CDM - reduced')
    ax1.errorbar(j+0.2,(SIDM.q1h_2g/np.mean(SIDM.q2d_it))-1,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C3s',markersize=10,label='SIDM - standard')
    ax1.errorbar(j+0.3,(SIDM.q1h_2g/np.mean(SIDM.q2dr_it))-1,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C6D',markersize=10,label='SIDM - reduced')

    ax1.set_ylim(-0.15,0.15)
    ax2.plot(j+0.,(np.mean(DM.q2d_it)/np.mean(SIDM.q2d_it))-1,'C4s',label=r'$\langle q \rangle$ - standard',markersize=10)
    ax2.plot(j+0.1,(np.mean(DM.q2dr_it)/np.mean(SIDM.q2dr_it))-1,'C3s',label=r'$\langle q \rangle$ - reduced',markersize=10)
    ax2.errorbar(j+0.2,(DM.q1h_2g/SIDM.q1h_2g)-1,
                 yerr=e_ratio_1h,
                 fmt='C1o',markersize=10,label='$q_{1h}$')
    ax2.errorbar(j+0.3,(DM.q2h_2g/SIDM.q2h_2g)-1,
                 yerr=e_ratio_2h,
                 fmt='C8o',markersize=10,label='$q_{2h}$')
    
    ax2.set_ylim(-0.3,0.3)
                 
    # ax2.plot(j+0.2,(DM.q1h_2g/SIDM.q1h_2g)-1,'C1o',label='1h',markersize=10)
    # ax2.plot(j+0.3,(DM.q2h_2g/SIDM.q2h_2g)-1,'C8o',label='2h',markersize=10)
    ax1.set_ylabel(r'$q_{1h} / \langle q \rangle - 1$')
    ax2.set_ylabel(r'$q_{CDM}/q_{SIDM} - 1$')

def corner_result(DM,SIDM,sname,name_tensor):

    mcmc_DM = np.array([DM.mcmc_q1h_2g[1500:],DM.mcmc_q2h_2g[1500:]]).T
    mcmc_SIDM = np.array([SIDM.mcmc_q1h_2g[1500:],SIDM.mcmc_q2h_2g[1500:]]).T
    mcmc_DM_ds = np.array([DM.mcmc_ds_lM[1500:],DM.mcmc_ds_c200[1500:]]).T
    mcmc_SIDM_ds = np.array([SIDM.mcmc_ds_lM[1500:],SIDM.mcmc_ds_c200[1500:]]).T

    f = corner.corner(mcmc_DM,labels=['$q_{1h}$','$q_{2h}$'],
                  smooth=1.,label_kwargs=({'fontsize':16}),
                  color='C7',truths=np.median(mcmc_DM,axis=0),truth_color='C7',
                  hist_kwargs=({'density':True}), levels=(0.99,0.9,0.6,0.3),
                  range=[(0.6,0.85),(0.12,0.75)])
    f = corner.corner(mcmc_SIDM,
                  smooth=1.,label_kwargs=({'fontsize':16}),
                  color='C6',truths=np.median(mcmc_SIDM,axis=0),truth_color='C6',
                  hist_kwargs=({'density':True}), levels=(0.99,0.9,0.6,0.3),
                  range=[(0.6,0.85),(0.12,0.75)],fig=f)

    axes = f.axes
    axes[1].text(0.5,0.5,sname,fontsize=16)
    f.savefig('../final_plots/corner_'+sname+'_'+name_tensor+'.pdf',bbox_inches='tight')

    f = corner.corner(mcmc_DM_ds,labels=['$\log M_{200}$','$c_{200}$'],
                  smooth=1.,label_kwargs=({'fontsize':16}),
                  color='C7',truths=np.median(mcmc_DM_ds,axis=0),truth_color='C7',
                  hist_kwargs=({'density':True}), levels=(0.99,0.9,0.6,0.3),
                  range=[(13.2,13.9),(4,8.2)])
    f = corner.corner(mcmc_SIDM_ds,
                  smooth=1.,label_kwargs=({'fontsize':16}),
                  color='C6',truths=np.median(mcmc_SIDM_ds,axis=0),truth_color='C6',
                  hist_kwargs=({'density':True}), levels=(0.99,0.9,0.6,0.3),
                  range=[(13.2,13.9),(4,8.2)],fig=f)

    axes = f.axes
    axes[1].text(0.5,0.5,sname,fontsize=16)
    f.savefig('../final_plots/corner_'+sname+'.pdf',bbox_inches='tight')
    
    
 
def plt_profile_fitted_final(DM,SIDM,RIN,ROUT,axx3,fittype='_2h_2q'):

    ax,ax1,ax2 = axx3
            
    ##############    

    ax.plot(DM.r,DM.DS_T,'C7',label='CDM')
    # ax.plot(rplot,DS1h,'C1',label='1h')
    # ax.plot(rplot,DS2h,'C8',label='2h')
    ax.plot(DM.r,DM.DS_fit,'C3')
    ax.fill_between(DM.r,DM.DS_T+DM.e_DS_T,DM.DS_T-DM.e_DS_T,color='C7',alpha=0.4)
    ax.plot(SIDM.r,SIDM.DS_T,'C6--',label='SIDM')
    # ax.plot(rplot,DS1h,'C1',label='1h')
    # ax.plot(rplot,DS2h,'C8',label='2h')
    ax.plot(SIDM.r,SIDM.DS_fit,'C3--')
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
    
    
    ax1.plot(DM.r,DM.GT,'C7')
    ax1.plot(DM.r,DM.GT1h+DM.GT2h,'C3',label='1h+2h')
    ax1.plot(DM.r,DM.GT1h,'C1',label='1h')
    ax1.plot(DM.r,DM.GT2h,'C8',label='2h')
    ax1.plot(SIDM.r,SIDM.GT,'C6--')    
    ax1.plot(SIDM.r,SIDM.GT1h+SIDM.GT2h,'C3--')
    ax1.plot(SIDM.r,SIDM.GT1h,'C1--')
    ax1.plot(SIDM.r,SIDM.GT2h,'C8--')
    ax1.fill_between(DM.r,DM.GT+DM.e_GT,DM.GT-DM.e_GT,color='C7',alpha=0.4)
    ax1.fill_between(SIDM.r,SIDM.GT+SIDM.e_GT,SIDM.GT-SIDM.e_GT,color='C6',alpha=0.4)
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

