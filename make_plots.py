import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import pylab
from astropy.io import fits
import corner
from matplotlib import cm
from binned_plots import make_plot2
from models_profiles import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
params = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.045, 'sigma8': 0.811, 'ns': 0.96}
folder = '../profiles3/'

def fit_qr(r,q,err_q):
    
    def qr(r,a,b):
        return b*r**a
    
    out = curve_fit(qr,r,q,sigma=err_q,absolute_sigma=True)
    a = out[0][0]
    b = out[0][1]
    
    return qr,a,b
    
        
    


def plot_zdist():
    
    S = fits.open('../MICE_sources_HSN_withextra.fits')[1].data
    
    
    f, ax = plt.subplots(1,1, figsize=(5,4),sharex = True,sharey=True)
    plt.hist(S.z_cgal_v,100,histtype='step',color='C3',lw=2.5)
    plt.xlabel('$z$')
    plt.ylabel('N [millions]')
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels([x/1.e6 for x in current_values])
    f.savefig('../zdist_v2.pdf',bbox_inches='tight')



def plot_q_dist(DM,SIDM,ax,method=''):
    
    import seaborn as sns    
    
    eq1h_dm = np.percentile(DM.mcmc_q1h_gt[3000:], [16, 84])
    eq2h_dm = np.percentile(DM.mcmc_q2h_gt[3000:], [16, 84])
    eq1h_sidm = np.percentile(SIDM.mcmc_q1h_gt[3000:], [16, 84])
    eq2h_sidm = np.percentile(SIDM.mcmc_q2h_gt[3000:], [16, 84])

    qdm = eval('DM.q2d'+method)
    qsidm = eval('SIDM.q2d'+method)
                
    sns.kdeplot(qdm.byteswap().newbyteorder(),color='C7',ax=ax,label='CDM',clip=(0,1),bw_adjust=0.6,lw=2)
    ax.axvline(np.mean(qdm),color='C7')
    ax.axvline(DM.q1h_gt,color='k',label='1h')
    ax.axvline(DM.q2h_gt,color='k',label='2h',ls='--')
    ax.axvspan(eq1h_dm[0],eq1h_dm[1],color='k',alpha=0.1)
    ax.axvspan(eq2h_dm[0],eq2h_dm[1],color='k',alpha=0.1)
    
    sns.kdeplot(qsidm.byteswap().newbyteorder(),color='C6',ax=ax,label='SIDM',clip=(0,1),bw_adjust=0.6,lw=2)
    ax.axvline(np.mean(qsidm),color='C6')
    ax.axvline(SIDM.q1h_gt,color='C3')
    ax.axvline(SIDM.q2h_gt,color='C3',ls='--')
    ax.axvspan(eq1h_sidm[0],eq1h_sidm[1],color='C3',alpha=0.1)
    ax.axvspan(eq2h_sidm[0],eq2h_sidm[1],color='C3',alpha=0.1)    
    ax.set_xlim(0.1,1)

    ax.set_xlabel('$q$')
    ax.set_ylabel('$P(q)$')

def linear_compare_q(DM,SIDM,ax,j):    
    
    ax1,ax2 = ax
    
    eq_1h_dm = np.diff(np.percentile(DM.mcmc_q1h_gt[3000:], [16, 50, 84]))
    eq_2h_dm = np.diff(np.percentile(DM.mcmc_q2h_gt[3000:], [16, 50, 84]))
    eq_1h_sidm = np.diff(np.percentile(DM.mcmc_q1h_gt[3000:], [16, 50, 84]))
    eq_2h_sidm = np.diff(np.percentile(DM.mcmc_q2h_gt[3000:], [16, 50, 84]))

    e_ratio_1h = np.array([np.sqrt((eq_1h_dm/SIDM.q1h_gt)**2 + ((eq_1h_sidm*DM.q1h_gt)/SIDM.q1h_gt**2)**2)]).T
    e_ratio_2h = np.array([np.sqrt((eq_2h_dm/SIDM.q2h_gt)**2 + ((eq_2h_sidm*DM.q2h_gt)/SIDM.q2h_gt**2)**2)]).T
    
    ax1.errorbar(np.mean(DM.q2d_it),DM.q1h_gt,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='ks',markersize=10,label='CDM - standard')
    ax1.errorbar(np.mean(DM.q2dr_it),DM.q1h_gt,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C7D',markersize=10,label='CDM - reduced')
    ax1.errorbar(np.mean(SIDM.q2d_it),SIDM.q1h_gt,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C3s',markersize=10,label='SIDM - standard')
    ax1.errorbar(np.mean(SIDM.q2dr_it),SIDM.q1h_gt,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C6D',markersize=10,label='SIDM - reduced')
    ax1.plot([0.5,0.9],[0.5,0.9],'C7--')


    ax2.errorbar(np.mean(DM.q2d_it),DM.q2h_gt,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='ks',markersize=10,label='CDM - standard')
    ax2.errorbar(np.mean(DM.q2dr_it),DM.q2h_gt,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C7D',markersize=10,label='CDM - reduced')
    ax2.errorbar(np.mean(SIDM.q2d_it),SIDM.q2h_gt,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C3s',markersize=10,label='SIDM - standard')
    ax2.errorbar(np.mean(SIDM.q2dr_it),SIDM.q2h_gt,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C6D',markersize=10,label='SIDM - reduced')
    ax2.plot([0.12,0.75],[0.12,0.75],'C7--')
                 
    # ax2.plot(j+0.2,(DM.q1h_gt/SIDM.q1h_gt)-1,'C1o',label='1h',markersize=10)
    # ax2.plot(j+0.3,(DM.q2h_gt/SIDM.q2h_gt)-1,'C8o',label='2h',markersize=10)
    ax2.set_xlabel(r'\langle q \rangle - 1$')
    ax1.set_ylabel(r'$q_{1h}$')
    ax2.set_ylabel(r'$q_{2h}$')


def compare_q(DM,SIDM,ax,j,method='2g'):    
    
    ax1,ax2 = ax
    
    eq_1h_dm = np.diff(np.percentile(eval('DM.mcmc_q1h_'+method)[3000:], [16, 50, 84]))
    eq_2h_dm = np.diff(np.percentile(eval('DM.mcmc_q2h_'+method)[3000:], [16, 50, 84]))
    eq_1h_sidm = np.diff(np.percentile(eval('SIDM.mcmc_q1h_'+method)[3000:], [16, 50, 84]))
    eq_2h_sidm = np.diff(np.percentile(eval('SIDM.mcmc_q2h_'+method)[3000:], [16, 50, 84]))

    e_ratio_1h = np.array([np.sqrt((eq_1h_dm/eval('SIDM.q1h_'+method))**2 + ((eq_1h_sidm*eval('DM.q1h_'+method))/eval('SIDM.q1h_'+method)**2)**2)]).T
    e_ratio_2h = np.array([np.sqrt((eq_2h_dm/eval('SIDM.q2h_'+method))**2 + ((eq_2h_sidm*eval('DM.q2h_'+method))/eval('SIDM.q2h_'+method)**2)**2)]).T
    
    ax1.errorbar(j+0.,(eval('DM.q1h_'+method)/np.mean(DM.q2d_it))-1,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='ks',markersize=10,label='CDM - standard')
    ax1.errorbar(j+0.1,(eval('DM.q1h_'+method)/np.mean(DM.q2dr_it))-1,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C7D',markersize=10,label='CDM - reduced')
    ax1.errorbar(j+0.2,(eval('SIDM.q1h_'+method)/np.mean(SIDM.q2d_it))-1,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C3s',markersize=10,label='SIDM - standard')
    ax1.errorbar(j+0.3,(eval('SIDM.q1h_'+method)/np.mean(SIDM.q2dr_it))-1,
                 yerr=np.array([eq_1h_dm]).T,
                 fmt='C6D',markersize=10,label='SIDM - reduced')

    ax1.set_ylim(-0.15,0.15)
    ax2.plot(j+0.,(np.mean(DM.q2d_it)/np.mean(SIDM.q2d_it))-1,'C4s',label=r'$\langle q \rangle$ - standard',markersize=10)
    ax2.plot(j+0.1,(np.mean(DM.q2dr_it)/np.mean(SIDM.q2dr_it))-1,'C3s',label=r'$\langle q \rangle$ - reduced',markersize=10)
    ax2.errorbar(j+0.2,(eval('DM.q1h_'+method)/eval('SIDM.q1h_'+method))-1,
                 yerr=e_ratio_1h,
                 fmt='C1o',markersize=10,label='$q_{1h}$')
    ax2.errorbar(j+0.3,(eval('DM.q2h_'+method)/eval('SIDM.q2h_'+method))-1,
                 yerr=e_ratio_2h,
                 fmt='C8o',markersize=10,label='$q_{2h}$')
    
    ax2.set_ylim(-0.3,0.3)
                 
    # ax2.plot(j+0.2,(DM.q1h_gt/SIDM.q1h_gt)-1,'C1o',label='1h',markersize=10)
    # ax2.plot(j+0.3,(DM.q2h_gt/SIDM.q2h_gt)-1,'C8o',label='2h',markersize=10)
    ax1.set_ylabel(r'$q_{1h} / \langle q \rangle - 1$')
    ax2.set_ylabel(r'$q_{CDM}/q_{SIDM} - 1$')


def compare_qr(DM,SIDM,ax,j,method='2g'):    
    
    ax1,ax2 = ax
    
    qr, a_dm, b_dm = fit_qr(DM.rs,DM.qs,DM.err_qs)
    qr, a_sidm, b_sidm = fit_qr(SIDM.rs,SIDM.qs,SIDM.err_qs)
    
    ea_1h_dm = np.diff(np.percentile(eval('DM.mcmc_a_'+method)[3000:], [16, 50, 84]))
    eb_1h_dm = np.diff(np.percentile(eval('DM.mcmc_b_'+method)[3000:], [16, 50, 84]))
    eq_2h_dm = np.diff(np.percentile(eval('DM.mcmc_q2hr_'+method)[3000:], [16, 50, 84]))
    ea_1h_sidm = np.diff(np.percentile(eval('SIDM.mcmc_a_'+method)[3000:], [16, 50, 84]))
    eb_1h_sidm = np.diff(np.percentile(eval('SIDM.mcmc_b_'+method)[3000:], [16, 50, 84]))
    eq_2h_sidm = np.diff(np.percentile(eval('SIDM.mcmc_q2hr_'+method)[3000:], [16, 50, 84]))

    e_ratio_a = np.array([np.sqrt((ea_1h_dm/eval('SIDM.a_'+method))**2 + ((ea_1h_sidm*eval('DM.a_'+method))/eval('SIDM.a_'+method)**2)**2)]).T
    e_ratio_b = np.array([np.sqrt((eb_1h_dm/eval('SIDM.b_'+method))**2 + ((eb_1h_sidm*eval('DM.b_'+method))/eval('SIDM.b_'+method)**2)**2)]).T
    e_ratio_2h = np.array([np.sqrt((eq_2h_dm/eval('SIDM.q2hr_'+method))**2 + ((eq_2h_sidm*eval('DM.q2hr_'+method))/eval('SIDM.q2hr_'+method)**2)**2)]).T
    
    ax1.errorbar(j+0.,(eval('DM.a_'+method)/a_dm)-1,
                 yerr=np.array([ea_1h_dm]).T,
                 fmt='ks',markersize=10,label='CDM - a')
    ax1.errorbar(j+0.1,(eval('DM.b_'+method)/b_dm)-1,
                 yerr=np.array([eb_1h_dm]).T,
                 fmt='C7D',markersize=10,label='CDM - b')
    ax1.errorbar(j+0.2,(eval('SIDM.a_'+method)/a_sidm)-1,
                 yerr=np.array([ea_1h_sidm]).T,
                 fmt='C3s',markersize=10,label='SIDM - a')
    ax1.errorbar(j+0.3,(eval('SIDM.b_'+method)/b_sidm)-1,
                 yerr=np.array([eb_1h_sidm]).T,
                 fmt='C6D',markersize=10,label='SIDM - b')

    ax1.set_ylim(-0.5,0.5)
    
    ax2.plot(j+0.,(a_dm/a_sidm)-1,'C4s',label=r'$a$',markersize=10)
    ax2.plot(j+0.1,(b_dm/b_sidm)-1,'C3s',label=r'$b$',markersize=10)
    ax2.errorbar(j+0.2,(eval('DM.a_'+method)/eval('SIDM.a_'+method))-1,
                 yerr=e_ratio_a,
                 fmt='C1o',markersize=10,label='$a$')
    ax2.errorbar(j+0.3,(eval('DM.b_'+method)/eval('SIDM.b_'+method))-1,
                 yerr=e_ratio_b,
                 fmt='C9o',markersize=10,label='$b$')
    ax2.errorbar(j+0.3,(eval('DM.q2hr_'+method)/eval('SIDM.q2hr_'+method))-1,
                 yerr=e_ratio_b,
                 fmt='C8o',markersize=10,label='$q2h$')
    
    ax2.set_ylim(-0.5,0.5)
                 
    # ax2.plot(j+0.2,(DM.q1h_gt/SIDM.q1h_gt)-1,'C1o',label='1h',markersize=10)
    # ax2.plot(j+0.3,(DM.q2h_gt/SIDM.q2h_gt)-1,'C8o',label='2h',markersize=10)
    ax1.set_ylabel(r'$q_{1h} / \langle q \rangle - 1$')
    ax2.set_ylabel(r'$q_{CDM}/q_{SIDM} - 1$')


def corner_result(DM,SIDM,sname,name_tensor):

    for method in ['gt','gx','2g']:

        mcmc_DM = np.array([eval('DM.mcmc_q1h_'+method)[3000:],eval('DM.mcmc_q2h_'+method)[3000:]]).T
        mcmc_SIDM = np.array([eval('SIDM.mcmc_q1h_'+method)[3000:],eval('SIDM.mcmc_q2h_'+method)[3000:]]).T
        
        ##########
        f = corner.corner(mcmc_DM,labels=['$q_{1h}$','$q_{2h}$'],
                    smooth=1.,label_kwargs=({'fontsize':16}),
                    color='C7',truths=np.median(mcmc_DM,axis=0),truth_color='C7',
                    hist_kwargs=({'density':True}), levels=(0.99,0.9,0.6,0.3),
                    range=[(0.,1.),(0.,1.)])
        f = corner.corner(mcmc_SIDM,
                    smooth=1.,label_kwargs=({'fontsize':16}),
                    color='C6',truths=np.median(mcmc_SIDM,axis=0),truth_color='C6',
                    hist_kwargs=({'density':True}), levels=(0.99,0.9,0.6,0.3),
                    range=[(0.,1.),(0.,1.)],fig=f)
    
        axes = f.axes
        axes[1].text(0.5,0.5,sname,fontsize=16)
        # f.savefig('../final_plots/corner_'+sname+'_'+name_tensor+'_'+method+'.pdf',bbox_inches='tight')
        f.savefig('../final_plots/corner_'+sname+'_'+name_tensor+'_'+method+'.png',bbox_inches='tight')
        
        ##########
        mcmc_DM = np.array([eval('DM.mcmc_a_'+method)[3000:],eval('DM.mcmc_b_'+method)[3000:],eval('DM.mcmc_q2h_'+method)[3000:]]).T
        mcmc_SIDM = np.array([eval('SIDM.mcmc_a_'+method)[3000:],eval('SIDM.mcmc_b_'+method)[3000:],eval('SIDM.mcmc_q2h_'+method)[3000:]]).T

        f1 = corner.corner(mcmc_DM,labels=['$a$','$b$','$q_{2h}$'],
                    smooth=1.,label_kwargs=({'fontsize':16}),
                    color='C7',truths=np.median(mcmc_DM,axis=0),truth_color='C7',
                    hist_kwargs=({'density':True}), levels=(0.99,0.9,0.6,0.3),
                    range=[(-0.5,0.),(0.4,1.),(0.12,0.95)])
        f1 = corner.corner(mcmc_SIDM,
                    smooth=1.,label_kwargs=({'fontsize':16}),
                    color='C6',truths=np.median(mcmc_SIDM,axis=0),truth_color='C6',
                    hist_kwargs=({'density':True}), levels=(0.99,0.9,0.6,0.3),
                    range=[(-0.5,0.),(0.4,1.),(0.12,0.95)],fig=f1)
    
        axes = f.axes
        axes[1].text(0.5,0.5,sname,fontsize=16)
        # f.savefig('../final_plots/corner_'+sname+'_'+name_tensor+'_r_'+method+'.pdf',bbox_inches='tight')
        f.savefig('../final_plots/corner_'+sname+'_'+name_tensor+'_r_'+method+'.png',bbox_inches='tight')

    ##########
    mcmc_DM_ds = np.array([DM.mcmc_ds_lM[3000:],DM.mcmc_ds_c200[3000:]]).T
    mcmc_SIDM_ds = np.array([SIDM.mcmc_ds_lM[3000:],SIDM.mcmc_ds_c200[3000:]]).T


    f = corner.corner(mcmc_DM_ds,labels=['$\log M_{200}$','$c_{200}$'],
                  smooth=1.,label_kwargs=({'fontsize':16}),
                  color='C7',truths=np.median(mcmc_DM_ds,axis=0),truth_color='C7',
                  hist_kwargs=({'density':True}), levels=(0.99,0.9,0.6,0.3),
                  range=[(12,15),(3.5,8.2)])
    f = corner.corner(mcmc_SIDM_ds,
                  smooth=1.,label_kwargs=({'fontsize':16}),
                  color='C6',truths=np.median(mcmc_SIDM_ds,axis=0),truth_color='C6',
                  hist_kwargs=({'density':True}), levels=(0.99,0.9,0.6,0.3),
                  range=[(12,15),(3.5,8.2)],fig=f)

    axes = f.axes
    axes[1].text(0.5,0.5,sname,fontsize=16)
    f.savefig('../final_plots/corner_'+sname+'_v2.pdf',bbox_inches='tight')
    f.savefig('../final_plots/corner_'+sname+'_v2.png',bbox_inches='tight')
    
    
 
def plt_profile_fitted_final(DM,SIDM,RIN,ROUT,axx3):

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
    ax1.plot(SIDM.r,SIDM.GT,'C6--')    
    ax1.fill_between(DM.r,DM.GT+DM.e_GT,DM.GT-DM.e_GT,color='C7',alpha=0.4)
    ax1.fill_between(SIDM.r,SIDM.GT+SIDM.e_GT,SIDM.GT-SIDM.e_GT,color='C6',alpha=0.4)
    
    ax1.plot(DM.r,DM.GT1h+DM.GT2h,'C3',label='1h+2h')
    ax1.plot(DM.r,DM.GT1h,'C1',label='1h')
    ax1.plot(DM.r,DM.GT2h,'C8',label='2h')
    ax1.plot(SIDM.r,SIDM.GT1h+SIDM.GT2h,'C3--')
    ax1.plot(SIDM.r,SIDM.GT1h,'C1--')
    ax1.plot(SIDM.r,SIDM.GT2h,'C8--')
    # ax1.plot(DM.r,DM.GT1h_fit2+DM.GT2h_fit2,'C3',label='1h+2h')
    # ax1.plot(DM.r,DM.GT1h_fit2,'C1',label='1h')
    # ax1.plot(DM.r,DM.GT2h_fit2,'C8',label='2h')
    # ax1.plot(SIDM.r,SIDM.GT1h_fit2+SIDM.GT2h_fit2,'C3--')
    # ax1.plot(SIDM.r,SIDM.GT1h_fit2,'C1--')
    # ax1.plot(SIDM.r,SIDM.GT2h_fit2,'C8--')
    
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
    ax2.plot(SIDM.r,SIDM.GX,'C6--',label='reduced')    
    ax2.fill_between(DM.r,DM.GX+DM.e_GX,DM.GX-DM.e_GX,color='C7',alpha=0.4)
    ax2.fill_between(SIDM.r,SIDM.GX+SIDM.e_GX,SIDM.GX-SIDM.e_GX,color='C6',alpha=0.4)
    
    ax2.plot(DM.r,DM.GX1h+DM.GX2h,'C3')
    ax2.plot(DM.r,DM.GX1h,'C1')
    ax2.plot(DM.r,DM.GX2h,'C8')
    ax2.plot(SIDM.r,SIDM.GX1h+SIDM.GX2h,'C3--')
    ax2.plot(SIDM.r,SIDM.GX1h,'C1--')
    ax2.plot(SIDM.r,SIDM.GX2h,'C8--')
    # ax2.plot(DM.r,DM.GX1h_fit2+DM.GX2h_fit2,'C3')
    # ax2.plot(DM.r,DM.GX1h_fit2,'C1')
    # ax2.plot(DM.r,DM.GX2h_fit2,'C8')
    # ax2.plot(SIDM.r,SIDM.GX1h_fit2+SIDM.GX2h_fit2,'C3--')
    # ax2.plot(SIDM.r,SIDM.GX1h_fit2,'C1--')
    # ax2.plot(SIDM.r,SIDM.GX2h_fit2,'C8--')
    
    # ax2.legend(loc=3,frameon=False)
    ax2.set_xlabel('r [$h^{-1}$ Mpc]')
    ax2.set_ylabel(r'$\Gamma_\times [h M_\odot/pc^2]$',labelpad=1.2)
    ax2.set_xscale('log')
    ax2.set_xlim(0.1,5)
    ax2.set_ylim(-16,17)
    ax2.xaxis.set_ticks([0.1,1,5])
    ax2.set_xticklabels([0.1,1,5])

def plt_profile_fitted_final_new(DM,SIDM,RIN,ROUT,axx3):


    a_dm, b_dm, q2h_dm = DM.a_2g, DM.b_2g, DM.q2hr_2g
    q1h_dm = b_dm*DM.r**a_dm
    e1h_dm   = (1.-q1h_dm)/(1.+q1h_dm)
    e2h_dm   = (1.-q2h_dm)/(1.+q2h_dm)
        
    a_sidm, b_sidm, q2h_sidm = SIDM.a_2g, SIDM.b_2g, SIDM.q2hr_2g
    q1h_sidm = b_sidm*SIDM.r**a_sidm
    e1h_sidm   = (1.-q1h_sidm)/(1.+q1h_sidm)
    e2h_sidm   = (1.-q2h_sidm)/(1.+q2h_sidm)


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
    ax1.plot(SIDM.r,SIDM.GT,'C6--')    
    ax1.fill_between(DM.r,DM.GT+DM.e_GT,DM.GT-DM.e_GT,color='C7',alpha=0.4)
    ax1.fill_between(SIDM.r,SIDM.GT+SIDM.e_GT,SIDM.GT-SIDM.e_GT,color='C6',alpha=0.4)
    
    ax1.plot(DM.r,DM.GT1h+DM.GT2h,'C3',label='1h+2h')
    ax1.plot(DM.r,DM.GT1h,'C1',label='1h')
    ax1.plot(DM.r,DM.GT2h,'C8',label='2h')
    ax1.plot(SIDM.r,SIDM.GT1h+SIDM.GT2h,'C3--')
    ax1.plot(SIDM.r,SIDM.GT1h,'C1--')
    ax1.plot(SIDM.r,SIDM.GT2h,'C8--')
    # ax1.plot(DM.r,DM.GT1hr_fit2+DM.GT2hr_fit2,'C3',label='1h+2h')
    # ax1.plot(DM.r,DM.GT1hr_fit2,'C1',label='1h')
    # ax1.plot(DM.r,DM.GT2hr_fit2,'C8',label='2h')
    # ax1.plot(SIDM.r,SIDM.GT1hr_fit2+SIDM.GT2hr_fit2,'C3--')
    # ax1.plot(SIDM.r,SIDM.GT1hr_fit2,'C1--')
    # ax1.plot(SIDM.r,SIDM.GT2hr_fit2,'C8--')
    
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
    ax2.plot(SIDM.r,SIDM.GX,'C6--',label='reduced')    
    ax2.fill_between(DM.r,DM.GX+DM.e_GX,DM.GX-DM.e_GX,color='C7',alpha=0.4)
    ax2.fill_between(SIDM.r,SIDM.GX+SIDM.e_GX,SIDM.GX-SIDM.e_GX,color='C6',alpha=0.4)
    
    ax2.plot(DM.r,DM.GX1h+DM.GX2h,'C3')
    ax2.plot(DM.r,DM.GX1h,'C1')
    ax2.plot(DM.r,DM.GX2h,'C8')
    ax2.plot(SIDM.r,SIDM.GX1h+SIDM.GX2h,'C3--')
    ax2.plot(SIDM.r,SIDM.GX1h,'C1--')
    ax2.plot(SIDM.r,SIDM.GX2h,'C8--')
    # ax2.plot(DM.r,DM.GX1hr_fit2+DM.GX2h_fit2,'C3')
    # ax2.plot(DM.r,DM.GX1hr_fit2,'C1')
    # ax2.plot(DM.r,DM.GX2hr_fit2,'C8')
    # ax2.plot(SIDM.r,SIDM.GX1hr_fit2+SIDM.GX2hr_fit2,'C3--')
    # ax2.plot(SIDM.r,SIDM.GX1hr_fit2,'C1--')
    # ax2.plot(SIDM.r,SIDM.GX2hr_fit2,'C8--')
    
    # ax2.legend(loc=3,frameon=False)
    ax2.set_xlabel('r [$h^{-1}$ Mpc]')
    ax2.set_ylabel(r'$\Gamma_\times [h M_\odot/pc^2]$',labelpad=1.2)
    ax2.set_xscale('log')
    ax2.set_xlim(0.1,5)
    ax2.set_ylim(-16,17)
    ax2.xaxis.set_ticks([0.1,1,5])
    ax2.set_xticklabels([0.1,1,5])
