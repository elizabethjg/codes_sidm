import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import pylab
from astropy.io import fits
# import corner
from matplotlib import cm
# from binned_plots import make_plot2
# from models_profiles import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
params = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.045, 'sigma8': 0.811, 'ns': 0.96}
folder = '../profiles3/'
from stacked import *


def q_comparison_it(DM,SIDM,ax):

    plt.figure(figsize=(6,4))
    plt.plot(DM.q2dr,DM.q2dr_it,'C3o',alpha=0.1)
    # plt.plot(SIDM.q2dr,SIDM.q2dr_it,'C1o',alpha=0.1)
    plt.plot([0.5,1],[0.5,1],'C7--')
    plt.axis([0.65,1,0.45,1])
    plt.xlabel('$q$')
    plt.ylabel('$q_{it}$')

def fit_qr(r,q,err_q):
    
    def qr(r,a,b):
        return b*r**a
    
    out = curve_fit(qr,r,q,sigma=err_q,absolute_sigma=True)
    a = out[0][0]
    b = out[0][1]
    
    return qr,a,b
    
        
def qplot(DM,SIDM,ax,samp):
    



    eq_ratio   = np.sqrt((DM.err_qs_all/SIDM.qs_all)**2 + ((DM.qs_all/(SIDM.qs_all**2))*SIDM.err_qs_all)**2)
    eq_ratio_b = np.sqrt((DM.err_qs_all/SIDM.qs)**2 + ((DM.qs_all/(SIDM.qs**2))*SIDM.err_qs)**2)


    ax.plot(DM.rs,np.zeros(len(DM.rs)),'C7')    
    ax.fill_between(DM.rs,
                    (DM.qs/SIDM.qs-1)+eq_ratio_b,
                    (DM.qs/SIDM.qs-1)-eq_ratio_b,
                    color='C2',alpha=0.3)
    ax.plot(DM.rs,DM.qs/SIDM.qs-1,'C2--',label='bound particles')
    
    ax.fill_between(DM.rs,
                    (DM.qs_all/SIDM.qs_all-1)+eq_ratio,
                    (DM.qs_all/SIDM.qs_all-1)-eq_ratio,
                    color='C4',alpha=0.3)
    ax.plot(DM.rs,DM.qs_all/SIDM.qs_all-1,'C4',label='all particles')


    # ax.plot(DM.rs,np.rad2deg(DM.fi_all),'k',label='DM - all')
    # ax.plot(DM.rs,np.rad2deg(DM.fi),'k--',label='DM - bound')
    # ax.plot(DM.rs,np.rad2deg(SIDM.fi_all),'C6',label='SIDM - all')
    # ax.plot(DM.rs,np.rad2deg(SIDM.fi),'C6--',label='SIDM - bound')

    
    ax.text(0.3,0.03,samp)
    
    ax.set_xscale('log')
    ax.set_xticks([0.2,0.5,1.0,4.0])
    ax.set_xticklabels(['0.2','0.5','1.0','4.0'])
    ax.set_xlabel('$r[h^{-1} Mpc]$')




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
    # ax2.plot(j+0.,(np.mean(DM.q2d_it)/np.mean(SIDM.q2d_it))-1,'C4D',label=r'$\langle q \rangle$ - standard',markersize=10)
    ax2.errorbar(j+0.2,(eval('DM.q1h_'+method)/eval('SIDM.q1h_'+method))-1,
                 yerr=e_ratio_1h,
                 fmt='C2o',markersize=10,label='$q_{1h}$')
    ax2.errorbar(j+0.3,(eval('DM.q2h_'+method)/eval('SIDM.q2h_'+method))-1,
                 yerr=e_ratio_2h,
                 fmt='C4o',markersize=10,label='$q_{2h}$')
    ax2.errorbar(j+0.1,(np.mean(DM.q2dr_it)/np.mean(SIDM.q2dr_it))-1,yerr=0.01,fmt='C3D',label=r'$\langle q \rangle$ - particle distribution',markersize=10)
    # ax2.errorbar(j+0.1,(np.mean(DM.qs[-1])/np.mean(SIDM.qs[-1]))-1,yerr=0.01,fmt='ks',label=r'$q(2 Mpc h^{-1})$ - stacked particle distribution',markersize=10)
    
    ax2.set_ylim(-0.3,0.3)
                 
    # ax2.plot(j+0.2,(DM.q1h_gt/SIDM.q1h_gt)-1,'C1o',label='1h',markersize=10)
    # ax2.plot(j+0.3,(DM.q2h_gt/SIDM.q2h_gt)-1,'C8o',label='2h',markersize=10)
    ax1.set_ylabel(r'$q_{1h} / \langle q \rangle - 1$')
    ax2.set_ylabel(r'$CDM/SIDM - 1$')


def stacked_particle(DM,SIDM,ax,samp):
    
    mr = DM.rs <= 1.0
    qr, a_dm, b_dm = fit_qr(DM.rs[mr],DM.qs[mr],DM.err_qs[mr])
    qr, a_sidm, b_sidm = fit_qr(SIDM.rs[mr],SIDM.qs[mr],SIDM.err_qs[mr])
    ax.fill_between(DM.rs[mr],(DM.qs+DM.err_qs)[mr],(DM.qs-DM.err_qs)[mr],color='C7',alpha=0.4)
    ax.fill_between(SIDM.rs[mr],(SIDM.qs+SIDM.err_qs)[mr],(SIDM.qs-SIDM.err_qs)[mr],color='C6',alpha=0.4)
    ax.plot(DM.rs[mr],qr(DM.rs[mr],a_dm,b_dm),'k',label=r'$\alpha = $'+f'{np.round(a_dm,3)}, $q_0 = ${np.round(b_dm,2)}')
    ax.plot(SIDM.rs[mr],qr(SIDM.rs[mr],a_sidm,b_sidm),'C3',label=r'$\alpha = $'+f'{np.round(a_sidm,3)}, $q_0 = ${np.round(b_sidm,2)}')
    ax.set_xscale('log')
    ax.text(0.2,0.85,samp)
    ax.legend(frameon=False,loc=1)
    ax.set_xticks([0.3,0.5,1.0])
    ax.set_xticklabels(['0.3','0.5','1.0'])
    ax.set_xlabel('$r[h^{-1} Mpc]$')
    
def stacked_lens_fit(DM,SIDM,ax,samp,j,Rvir):
    
    def logq(r,alpha,lq0):
        return lq0 + alpha*np.log10(r)

    err_alpha = np.diff(np.percentile(DM.mcmc_a_2g_fb[3000:], [16, 50, 84]))
    err_q0    = np.diff(np.percentile(DM.mcmc_a_2g_fb[3000:], [16, 50, 84]))
    err_q2h_dm   = np.diff(np.percentile(DM.mcmc_q2hr_2g_fb[3000:], [16, 50, 84]))
    
    q0    = DM.b_2g_fb
    alpha = DM.a_2g_fb
    
    lq0  = np.log10(q0)
    elq0 = err_q0/(q0*np.log(10.))
    
    label_dm = r'CDM: $\alpha = '+str(np.round(alpha,3))+', q_0 = '+str(np.round(q0,3))+', q_{2h} = '+str(np.round(DM.q2hr_2g_fb,2))+'$'
    
    
    mr = DM.rs <= Rvir    
    ax.plot(DM.rs[mr],10**(logq(DM.rs[mr],alpha,lq0)),'k',
            label='CDM')
    ax.plot(DM.rs[~mr],10**(logq(DM.rs[~mr],alpha,lq0)),'k',alpha=0.5)
    
    
    ax.fill_between(DM.rs[mr], 
                    10**(logq(DM.rs[mr],alpha+err_alpha[1],lq0+elq0[1])), 
                    10**(logq(DM.rs[mr],alpha-err_alpha[0],lq0-elq0[0])), 
                    interpolate=True, 
                    color=f'C7',alpha=0.3)    
    ax.fill_between(DM.rs[~mr], 
                    10**(logq(DM.rs[~mr],alpha+err_alpha[1],lq0+elq0[1])), 
                    10**(logq(DM.rs[~mr],alpha-err_alpha[0],lq0-elq0[0])), 
                    interpolate=True, 
                    color=f'C7',alpha=0.1)    

    ax.fill_between(SIDM.rs[~mr],
                    np.ones(len(DM.rs[~mr]))*(DM.q2hr_2g_fb+err_q2h_dm[1]),
                    np.ones(len(DM.rs[~mr]))*(DM.q2hr_2g_fb-err_q2h_dm[0]),
                    interpolate=True, 
                    color=f'C7',alpha=0.3) 
    ax.fill_between(SIDM.rs[mr],
                    np.ones(len(DM.rs[mr]))*(DM.q2hr_2g_fb+err_q2h_dm[1]),
                    np.ones(len(DM.rs[mr]))*(DM.q2hr_2g_fb-err_q2h_dm[0]),
                    interpolate=True, 
                    color=f'C7',alpha=0.1) 


    err_alpha = np.diff(np.percentile(SIDM.mcmc_a_2g_fb[3000:], [16, 50, 84]))
    err_q0 = np.diff(np.percentile(SIDM.mcmc_a_2g_fb[3000:], [16, 50, 84]))
    err_q2h_sidm   = np.diff(np.percentile(SIDM.mcmc_q2hr_2g_fb[3000:], [16, 50, 84]))
    
    q0    = SIDM.b_2g_fb
    alpha = SIDM.a_2g_fb
    
    lq0  = np.log10(q0)
    elq0 = err_q0/(q0*np.log(10.))    
    
    label_sidm = r'SIDM: $\alpha = '+str(np.round(alpha,3))+', q_0 = '+str(np.round(q0,3))+', q_{2h} = '+str(np.round(SIDM.q2hr_2g_fb,2))+'$'
    
    ax.plot(SIDM.rs[mr],10**(logq(SIDM.rs[mr],alpha,lq0)),f'C3',label='SIDM')
    ax.plot(SIDM.rs[~mr],10**(logq(SIDM.rs[~mr],alpha,lq0)),f'C3',alpha=0.5)
    
    ax.plot(SIDM.rs[~mr],np.ones(len(SIDM.rs[~mr]))*(DM.q2hr_2g_fb),f'k:',label='$q_{2h}$')
    ax.plot(SIDM.rs[mr],np.ones(len(SIDM.rs[mr]))*(DM.q2hr_2g_fb),f'k:',alpha=0.5)
    
    ax.plot(SIDM.rs[~mr],np.ones(len(SIDM.rs[~mr]))*(SIDM.q2hr_2g_fb),f'C3:')
    ax.plot(SIDM.rs[mr],np.ones(len(SIDM.rs[mr]))*(SIDM.q2hr_2g_fb),f'C3:',alpha=0.5)
    
    
    ax.fill_between(DM.rs[mr], 
                    10**(logq(DM.rs[mr],alpha+err_alpha[1],lq0+elq0[1])), 
                    10**(logq(DM.rs[mr],alpha-err_alpha[0],lq0-elq0[0])), 
                    interpolate=True, 
                    color=f'C6',alpha=0.3) 
    ax.fill_between(DM.rs[~mr], 
                    10**(logq(DM.rs[~mr],alpha+err_alpha[1],lq0+elq0[1])), 
                    10**(logq(DM.rs[~mr],alpha-err_alpha[0],lq0-elq0[0])), 
                    interpolate=True, 
                    color=f'C6',alpha=0.1) 
                    
    ax.fill_between(SIDM.rs[~mr],
                    np.ones(len(SIDM.rs[~mr]))*(SIDM.q2hr_2g_fb+err_q2h_sidm[1]),
                    np.ones(len(SIDM.rs[~mr]))*(SIDM.q2hr_2g_fb-err_q2h_sidm[0]),
                    interpolate=True, 
                    color=f'C6',alpha=0.3) 
    ax.fill_between(SIDM.rs[mr],
                    np.ones(len(SIDM.rs[mr]))*(SIDM.q2hr_2g_fb+err_q2h_sidm[1]),
                    np.ones(len(SIDM.rs[mr]))*(SIDM.q2hr_2g_fb-err_q2h_sidm[0]),
                    interpolate=True, 
                    color=f'C6',alpha=0.1) 
    
    ax.text(0.18,0.85,samp)                
    ax.text(0.18,0.47,label_dm,fontsize=10)
    ax.text(0.18,0.43,label_sidm,fontsize=10)      
    ax.set_xscale('log')
    ax.set_ylim(0.42,0.89)
    ax.set_xticks([0.2,0.5,1.0,2.0,4.])
    ax.set_xticklabels(['0.2','0.5','1.0','2.0','4.0'])
    ax.set_xlabel('$r[h^{-1} Mpc]$')



def compare_qr(DM,SIDM,ax,j,method='2g_fb'):    
    
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

    ax1.set_ylim(-0.8,0.5)
    
    ax2.errorbar(j+0.2,(eval('DM.a_'+method)/eval('SIDM.a_'+method))-1,
                 yerr=e_ratio_a,
                 fmt='C9o',markersize=10,label=r'$\alpha$')
    ax2.errorbar(j+0.3,(eval('DM.b_'+method)/eval('SIDM.b_'+method))-1,
                 yerr=e_ratio_b,
                 fmt='C2o',markersize=10,label=r'$q_0$')
    ax2.errorbar(j+0.3,(eval('DM.q2hr_'+method)/eval('SIDM.q2hr_'+method))-1,
                 yerr=e_ratio_b,
                 fmt='C4o',markersize=10,label='$q2h$')
    ax2.errorbar(j+0.,(a_dm/a_sidm)-1,yerr=0.01,fmt='C7s',label=r'$\alpha$ - stacked particle distribution',markersize=10)
    ax2.errorbar(j+0.1,(b_dm/b_sidm)-1,yerr=0.01,fmt='ks',label=r'$q_0$ - stacked particle distribution',markersize=10)
    
    ax2.set_ylim(-1.2,0.5)
                 
    # ax2.plot(j+0.2,(DM.q1h_gt/SIDM.q1h_gt)-1,'C1o',label='1h',markersize=10)
    # ax2.plot(j+0.3,(DM.q2h_gt/SIDM.q2h_gt)-1,'C8o',label='2h',markersize=10)
    ax1.set_ylabel(r'$q_{1h} / \langle q \rangle - 1$')
    ax2.set_ylabel(r'$CDM/SIDM - 1$')
    
    return [b_dm,a_dm,b_sidm,a_sidm]


def corner_result(DM,SIDM,sname,name_tensor):

    for method in ['2g']:

        mcmc_DM = np.array([eval('DM.mcmc_q1h_'+method)[3000:],eval('DM.mcmc_q2h_'+method)[3000:]]).T
        mcmc_SIDM = np.array([eval('SIDM.mcmc_q1h_'+method)[3000:],eval('SIDM.mcmc_q2h_'+method)[3000:]]).T
        
        ##########
        f = corner.corner(mcmc_DM,labels=['$q_{1h}$','$q_{2h}$'],
                    smooth=1.,label_kwargs=({'fontsize':16}),
                    color='C7',truths=np.median(mcmc_DM,axis=0),truth_color='C7',
                    hist_kwargs=({'density':True}), levels=(0.85,0.6,0.3))#,
                    # range=[(0.55,0.75),(0.5,0.9)])
        f = corner.corner(mcmc_SIDM,
                    smooth=1.,label_kwargs=({'fontsize':16}),
                    color='C6',truths=np.median(mcmc_SIDM,axis=0),truth_color='C6',
                    hist_kwargs=({'density':True}), levels=(0.85,0.6,0.3),fig=f)#,
                    # range=[(0.55,0.75),(0.5,0.9)],fig=f)
    
        axes = f.axes
        axes[1].text(0.5,0.5,sname,fontsize=16)
        f.savefig('../final_plots/corner_'+sname+'_'+name_tensor+'_'+method+'.pdf',bbox_inches='tight')
        f.savefig('../final_plots/corner_'+sname+'_'+name_tensor+'_'+method+'.png',bbox_inches='tight')
        
        ##########
        mcmc_DM = np.array([eval('DM.mcmc_a_'+method+'_fb')[3000:],eval('DM.mcmc_b_'+method+'_fb')[3000:],eval('DM.mcmc_q2hr_'+method+'_fb')[3000:]]).T
        mcmc_SIDM = np.array([eval('SIDM.mcmc_a_'+method+'_fb')[3000:],eval('SIDM.mcmc_b_'+method+'_fb')[3000:],eval('SIDM.mcmc_q2hr_'+method+'_fb')[3000:]]).T

        f1 = corner.corner(mcmc_DM,labels=[r'$\alpha$',r'$q_0$','$q_{2h}$'],
                    smooth=1.,label_kwargs=({'fontsize':16}),
                    color='C7',truths=np.median(mcmc_DM,axis=0),truth_color='C7',
                    hist_kwargs=({'density':True}), levels=(0.85,0.6,0.3))#,
                    # range=[(-0.2,0.05),(0.4,0.75),(0.5,0.9)])
        f1 = corner.corner(mcmc_SIDM,
                    smooth=1.,label_kwargs=({'fontsize':16}),
                    color='C6',truths=np.median(mcmc_SIDM,axis=0),truth_color='C6',
                    hist_kwargs=({'density':True}), levels=(0.85,0.6,0.3),fig=f1)#,
                    # range=[(-0.2,0.05),(0.45,0.75),(0.5,0.9)],fig=f1)
    
        axes = f1.axes
        axes[1].text(0.5,0.5,sname,fontsize=16)
        f1.savefig('../final_plots/corner_'+sname+'_'+name_tensor+'_r_'+method+'.pdf',bbox_inches='tight')
        f1.savefig('../final_plots/corner_'+sname+'_'+name_tensor+'_r_'+method+'.png',bbox_inches='tight')

    ##########
    mcmc_DM_ds = np.array([DM.mcmc_ds_lM[1500:],DM.mcmc_ds_c200[1500:]]).T
    mcmc_SIDM_ds = np.array([SIDM.mcmc_ds_lM[1500:],SIDM.mcmc_ds_c200[1500:]]).T


    f = corner.corner(mcmc_DM_ds,labels=['$\log M_{200}$','$c_{200}$'],
                  smooth=3.,label_kwargs=({'fontsize':16}),
                  color='C7',truths=np.median(mcmc_DM_ds,axis=0),truth_color='C7',
                  hist_kwargs=({'density':True}), levels=(0.85,0.6,0.3))#,
                  # range=[(13,14.4),(4.,7.5)])
    f = corner.corner(mcmc_SIDM_ds,
                  smooth=3.,label_kwargs=({'fontsize':16}),
                  color='C6',truths=np.median(mcmc_SIDM_ds,axis=0),truth_color='C6',
                  hist_kwargs=({'density':True}), levels=(0.85,0.6,0.3),fig=f)#,
                  # range=[(13,14.4),(4.,7.5)],fig=f)

    axes = f.axes
    axes[1].text(0.5,0.5,sname,fontsize=16)
    f.savefig('../final_plots/corner_'+sname+'_v2.pdf',bbox_inches='tight')
    f.savefig('../final_plots/corner_'+sname+'_v2.png',bbox_inches='tight')
    
    
 
def plt_profile_fitted_final(DM,SIDM,RIN,ROUT,axx3,jdx):

    ax,ax1,ax2 = axx3
    
    ##############    
    gl = 2
    ax.plot(DM.r,DM.DS_T,'C7')
    ax.plot(DM.r,DM.DS1h_fit,'C1')
    ax.plot(DM.r,DM.DS2h_fit,'C8')
    ax.plot(DM.r,DM.DS_fit,'C3',
           label = r'$\chi^2_{red} = $'+f'{np.round(chi_red(DM.DS_fit,DM.DS_T,DM.e_DS_T,gl),2)}')
    ax.fill_between(DM.r,DM.DS_T+DM.e_DS_T,DM.DS_T-DM.e_DS_T,color='C7',alpha=0.4)
    ax.plot(SIDM.r,SIDM.DS_T,'C6--')
    ax.plot(SIDM.r,SIDM.DS1h_fit,'C1--')
    ax.plot(SIDM.r,SIDM.DS2h_fit,'C8--')
    ax.plot(SIDM.r,SIDM.DS_fit,'C3--',
            label = r'$\chi^2_{red} = $'+f'{np.round(chi_red(SIDM.DS_fit,SIDM.DS_T,SIDM.e_DS_T,gl),2)}')
    ax.fill_between(SIDM.r,SIDM.DS_T+SIDM.e_DS_T,SIDM.DS_T-SIDM.e_DS_T,color='C6',alpha=0.4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if jdx == 0 and jdx == 2:
        ax.set_ylabel(r'$\Delta\Sigma [h M_\odot/pc^2]$',labelpad=0.1)
    ax.set_xlabel('r [$h^{-1}$ Mpc]')
    ax.set_ylim(0.5,200)
    ax.set_xlim(0.1,5)
    ax.xaxis.set_ticks([0.1,1,3])
    ax.set_xticklabels([0.1,1,3])
    ax.yaxis.set_ticks([1,10,100])
    ax.set_yticklabels([1,10,100])
    ax.legend(loc=3,frameon=False,fontsize=10)
    
    
    ax1.plot(DM.r,DM.GT,'C7')
    ax1.plot(SIDM.r,SIDM.GT,'C6--')    
    ax1.fill_between(DM.r,DM.GT+DM.e_GT,DM.GT-DM.e_GT,color='C7',alpha=0.4)
    ax1.fill_between(SIDM.r,SIDM.GT+SIDM.e_GT,SIDM.GT-SIDM.e_GT,color='C6',alpha=0.4)
    
    ax1.plot(DM.r,DM.GT1h_fit2+DM.GT2h_fit2,'C3',
             label = r'$\chi^2_{red} = $'+f'{np.round(chi_red(DM.GT1h_fit2+DM.GT2h_fit2,DM.GT,DM.e_GT,gl),2)}')
    ax1.plot(DM.r,DM.GT1h_fit2,'C1')
    ax1.plot(DM.r,DM.GT2h_fit2,'C8')
    ax1.plot(SIDM.r,SIDM.GT1h_fit2+SIDM.GT2h_fit2,'C3--',
             label = r'$\chi^2_{red} = $'+f'{np.round(chi_red(SIDM.GT1h_fit2+SIDM.GT2h_fit2,SIDM.GT,SIDM.e_GT,gl),2)}')
    ax1.plot(SIDM.r,SIDM.GT1h_fit2,'C1--')
    ax1.plot(SIDM.r,SIDM.GT2h_fit2,'C8--')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('r [$h^{-1}$ Mpc]')
    if jdx == 0:
        ax1.set_ylabel(r'$\Gamma_T [h M_\odot/pc^2]$',labelpad=0.1,fontsize=10)
    ax1.set_ylim(0.5,100)
    ax1.set_xlim(0.1,5)
    ax1.xaxis.set_ticks([0.1,1,3])
    ax1.set_xticklabels([0.1,1,3])
    ax1.yaxis.set_ticks([1,10,100])
    ax1.set_yticklabels([1,10,100])
    ax1.legend(loc=1,frameon=False,fontsize=10)
    
    ax2.plot([0,10],[0,0],'k')
    ax2.plot(DM.r,DM.GX,'C7')
    ax2.plot(SIDM.r,SIDM.GX,'C6--')    
    ax2.fill_between(DM.r,DM.GX+DM.e_GX,DM.GX-DM.e_GX,color='C7',alpha=0.4)
    ax2.fill_between(SIDM.r,SIDM.GX+SIDM.e_GX,SIDM.GX-SIDM.e_GX,color='C6',alpha=0.4)
    ax2.plot(DM.r,DM.GX1h_fit2+DM.GX2h_fit2,'C3',
             label = r'$\chi^2_{red} = $'+f'{np.round(chi_red(DM.GX1h_fit2+DM.GX2h_fit2,DM.GX,DM.e_GX,gl),2)}')
    ax2.plot(DM.r,DM.GX1h_fit2,'C1')
    ax2.plot(DM.r,DM.GX2h_fit2,'C8')
    ax2.plot(SIDM.r,SIDM.GX1h_fit2+SIDM.GX2h_fit2,'C3--',
            label = r'$\chi^2_{red} = $'+f'{np.round(chi_red(SIDM.GX1h_fit2+SIDM.GX2h_fit2,SIDM.GX,SIDM.e_GX,gl),2)}')
    ax2.plot(SIDM.r,SIDM.GX1h_fit2,'C1--')
    ax2.plot(SIDM.r,SIDM.GX2h_fit2,'C8--')
    
    # ax2.legend(loc=3,frameon=False)
    ax2.set_xlabel('r [$h^{-1}$ Mpc]')
    if jdx == 0:
        ax2.set_ylabel(r'$\Gamma_\times [h M_\odot/pc^2]$',labelpad=0.1,fontsize=10)
    ax2.set_xscale('log')
    ax2.set_xlim(0.1,5)
    ax2.set_ylim(-6,16)
    ax2.xaxis.set_ticks([0.1,1,3])
    ax2.set_xticklabels([0.1,1,3])
    ax2.legend(loc=1,frameon=False,fontsize=10)

def plt_profile_fitted_final_new(DM,SIDM,RIN,ROUT,axx3,jdx):


    a_dm, b_dm, q2h_dm = DM.a_2g_fb, DM.b_2g_fb, DM.q2hr_2g_fb
    q1h_dm = b_dm*DM.r**a_dm
    e1h_dm   = (1.-q1h_dm)/(1.+q1h_dm)
    e2h_dm   = (1.-q2h_dm)/(1.+q2h_dm)
        
    a_sidm, b_sidm, q2h_sidm = SIDM.a_2g_fb, SIDM.b_2g_fb, SIDM.q2hr_2g_fb
    q1h_sidm = b_sidm*SIDM.r**a_sidm
    e1h_sidm   = (1.-q1h_sidm)/(1.+q1h_sidm)
    e2h_sidm   = (1.-q2h_sidm)/(1.+q2h_sidm)

    Gterms_dm = quadrupoles_from_map_model(M200=10**DM.lM200_ds,c200=DM.c200_ds,
                                        resolution=2000,
                                        RIN=100.,ROUT=5000.,
                                        ndots=20
                                        )

    Gterms_sidm = quadrupoles_from_map_model(M200=10**SIDM.lM200_ds,c200=SIDM.c200_ds,
                                        resolution=2000,
                                        RIN=100.,ROUT=5000.,
                                        ndots=20
                                        )
    G1h_dm = Gterms_dm(a_dm,b_dm)
    G1h_sidm = Gterms_sidm(a_sidm,b_sidm)
    

    ax1,ax2 = axx3
            
    ##############        
    gl = 3
    model_GT_dm = G1h_dm['GT'] + e2h_dm*Gterms_dm.GT_2h
    model_GT_sidm = G1h_sidm['GT'] + e2h_sidm*Gterms_sidm.GT_2h
    ax1.plot(DM.r,DM.GT,'C7')
    ax1.plot(SIDM.r,SIDM.GT,'C6--')    
    ax1.fill_between(DM.r,DM.GT+DM.e_GT,DM.GT-DM.e_GT,color='C7',alpha=0.4)
    ax1.fill_between(SIDM.r,SIDM.GT+SIDM.e_GT,SIDM.GT-SIDM.e_GT,color='C6',alpha=0.4)
    ax1.plot(DM.r,G1h_dm['GT'] + e2h_dm*Gterms_dm.GT_2h,'C3',
             label = r'$\chi^2_{red} = $'+f'{np.round(chi_red(model_GT_dm,DM.GT,DM.e_GT,gl),2)}')
    ax1.plot(DM.r,G1h_dm['GT'],'C1')
    ax1.plot(DM.r,e2h_dm*Gterms_dm.GT_2h,'C8')
    ax1.plot(SIDM.r,G1h_sidm['GT'] + e2h_sidm*Gterms_sidm.GT_2h,'C3--',
             label = r'$\chi^2_{red} = $'+f'{np.round(chi_red(model_GT_sidm,SIDM.GT,SIDM.e_GT,gl),2)}')
    ax1.plot(SIDM.r,G1h_sidm['GT'],'C1--')
    ax1.plot(SIDM.r,e2h_sidm*Gterms_sidm.GT_2h,'C8--')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('r [$h^{-1}$ Mpc]')
    if jdx == 0:
        ax1.set_ylabel(r'$\Gamma_T [h M_\odot/pc^2]$',labelpad=0.1,fontsize=10)
    ax1.set_ylim(0.5,100)
    ax1.set_xlim(0.1,5)
    ax1.xaxis.set_ticks([0.1,1,4])
    ax1.set_xticklabels([0.1,1,4])
    ax1.yaxis.set_ticks([1,10,100])
    ax1.set_yticklabels([1,10,100])
    ax1.legend(loc=1,frameon=False,fontsize=10)
    
    model_GX_dm = G1h_dm['GX'] + e2h_dm*Gterms_dm.GX_2h
    model_GX_sidm = G1h_sidm['GX'] + e2h_sidm*Gterms_sidm.GX_2h
    ax2.plot([0,10],[0,0],'k')
    ax2.plot(DM.r,DM.GX,'C7')
    ax2.plot(SIDM.r,SIDM.GX,'C6--')    
    ax2.fill_between(DM.r,DM.GX+DM.e_GX,DM.GX-DM.e_GX,color='C7',alpha=0.4)
    ax2.fill_between(SIDM.r,SIDM.GX+SIDM.e_GX,SIDM.GX-SIDM.e_GX,color='C6',alpha=0.4)
    
    ax2.plot(DM.r,G1h_dm['GX'] + e2h_dm*Gterms_dm.GX_2h,'C3',
             label = r'$\chi^2_{red} = $'+f'{np.round(chi_red(model_GX_dm,DM.GX,DM.e_GX,gl),2)}')
    ax2.plot(DM.r,G1h_dm['GX'],'C1')
    ax2.plot(DM.r,e2h_dm*Gterms_dm.GX_2h,'C8')
    ax2.plot(SIDM.r,G1h_sidm['GX'] + e2h_sidm*Gterms_sidm.GX_2h,'C3--',
             label = r'$\chi^2_{red} = $'+f'{np.round(chi_red(model_GX_sidm,SIDM.GX,SIDM.e_GX,gl),2)}')
    ax2.plot(SIDM.r,G1h_sidm['GX'],'C1--')
    ax2.plot(SIDM.r,e2h_sidm*Gterms_sidm.GX_2h,'C8--')
    
    # ax2.legend(loc=3,frameon=False)
    ax2.set_xlabel('r [$h^{-1}$ Mpc]')
    if jdx == 0:
        ax2.set_ylabel(r'$\Gamma_\times [h M_\odot/pc^2]$',labelpad=0.1,fontsize=10)
    ax2.set_xscale('log')
    ax2.set_xlim(0.1,5)
    ax2.set_ylim(-6,16)
    ax2.xaxis.set_ticks([0.1,1,4])
    ax2.set_xticklabels([0.1,1,4])
    ax2.legend(loc=1,frameon=False,fontsize=10)


def make_radial_plot():

    
    Gterms = quadrupoles_from_map_model(M200=10**14,c200=5.,
                                        resolution=2000,
                                        RIN=100.,ROUT=5000.,
                                        ndots=20
                                        )
    q0 = 0.6
    G1h = Gterms(-0.1,q0)
    e = (1. - q0)/(1. + q0)
    
    GT_func,GX_func = GAMMA_components(Gterms.R,0.,ellip=e,M200 = 10**14,c200=5,cosmo_params=params)   
    
    fig, ax = plt.subplots(4,1, figsize=(6,8),sharex = True,gridspec_kw={'height_ratios': [4,2,4,2]})    

    fig.subplots_adjust(hspace=0,wspace=0)
    
    ax[0].plot(Gterms.R,GT_func,'C4',label='$q = 0.6$',lw=2)
    ax[0].plot(Gterms.R,G1h['GT'],'C2',label=r'$q_0 = 0.6$, $\alpha = -0.1$',lw=2)
    ax[2].plot(Gterms.R,G1h['GX'],'C2',label=r'$q_0 = 0.6$, $\alpha = -0.1$',lw=2)
    ax[2].plot(Gterms.R,GX_func,'C4',label='$q = 0.6$',lw=2)
    
    ax[2].plot(Gterms.R,np.zeros(len(Gterms.R)),'C7')
    ax[1].plot(Gterms.R,np.zeros(len(Gterms.R)),'C7')
    ax[3].plot(Gterms.R,np.zeros(len(Gterms.R)),'C7')

    ax[1].plot(Gterms.R,G1h['GT']-GT_func,'k--',lw=2)
    ax[3].plot(Gterms.R,G1h['GX']-GX_func,'k--',lw=2)
    
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    
    ax[0].legend(frameon=False)
    
    ax[3].set_xlabel('r [$h^{-1}$ Mpc]')
    ax[0].set_ylabel(r'$\Gamma_\times [h M_\odot/pc^2]$')
    ax[2].set_ylabel(r'$\Gamma_T [h M_\odot/pc^2]$')
    ax[1].set_ylabel(r'Difference')
    ax[3].set_ylabel(r'Difference')
    ax[3].xaxis.set_ticks([0.1,1,3])
    ax[3].set_xticklabels([0.1,1,3])
    
    fig.savefig('../final_plots/comparison_radial.pdf',bbox_inches='tight')



def make_3D_plot(j):
    
    
    import h5py
    
    path_all = '/mnt/projects/lensing/SIDM_project/cuadrados/CDM_10/'
    path_halo = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/rockstar/CDM/'
    
    halo = h5py.File(path_halo+'halo_'+str(j)+'.hdf5','r') 
    
    X = np.array(halo['X']) - halo.attrs['x0']
    Y = np.array(halo['Y']) - halo.attrs['y0']
    Z = np.array(halo['Z']) - halo.attrs['z0']


    allp = h5py.File(path_all+'halo_'+str(j)+'.hdf5','r') 
    
    Xp = np.array(allp['X']) - allp.attrs['x0 center']
    Yp = np.array(allp['Y']) - allp.attrs['y0 center']
    Zp = np.array(allp['Z']) - allp.attrs['z0 center']

    print('Making plot...')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(Xp,Yp,Zp,'k,',alpha=0.005)
    ax.plot(X,Y,Z,'C3,',alpha=0.01)
    ax.set_xlabel('X [Mpc]')
    ax.set_ylabel('Y [Mpc]')
    ax.set_zlabel('Z [Mpc]')
    fig.savefig('../particles_halo_'+str(j)+'.png',bbox_inches='tight')
    
make_3D_plot(12)

