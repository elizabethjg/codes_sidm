import sys
import os 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from make_plots import *
# from stacked import fit_quadrupoles_2terms_qrfunc
import emcee
from models_profiles import GAMMA_components_parallel
params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}

def fit_S2_2terms(R,s2,e_s2,S2,S2_2h):
    
    def log_likelihood(data_model, R, s2, e_s2):
        
        q1h, q2h = data_model
                
        e1h   = (1.-q1h)/(1.+q1h)
        e2h   = (1.-q2h)/(1.+q2h)
        
        sigma2 = e_s2**2
        model = e1h*S2 + e2h*S2_2h
        L = -0.5 * np.sum((model - s2)**2 / sigma2 + np.log(2.*np.pi*sigma2))

        return L
    
    
    def log_probability(data_model, R, profiles, eprofiles):
        
        q1h, q2h = data_model
        
        if 0. < q1h < 1. and 0. < q2h < 1.:
            return log_likelihood(data_model, R, profiles, eprofiles)
            
        return -np.inf
    
    # initializing
    
    pos = np.array([np.random.uniform(0.6,0.9,15),
                    np.random.uniform(0.1,0.5,15)]).T
    
    nwalkers, ndim = pos.shape
    
    #-------------------
    # running emcee
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(R,s2,e_s2))
                                    # pool = pool)
                    
    sampler.run_mcmc(pos, 250, progress=True)
    
    mcmc_out = sampler.get_chain(flat=True).T
    
    return np.median(mcmc_out[0][1500:]),np.median(mcmc_out[1][1500:]),mcmc_out[0],mcmc_out[1]


def fit_quadrupoles_2terms(R,gt,gx,egt,egx,GT,GX,GT_2h,GX_2h,fit_components):
    
    print('fitting components: ',fit_components)
    def log_likelihood(data_model, R, profiles, eprofiles):
        
        q1h, q2h = data_model
        
        gt, gx   = profiles
        egt, egx = eprofiles
        
        e1h   = (1.-q1h)/(1.+q1h)
        e2h   = (1.-q2h)/(1.+q2h)
        
        sigma2 = egt**2
        mGT = e1h*GT + e2h*GT_2h
        LGT = -0.5 * np.sum((mGT - gt)**2 / sigma2 + np.log(2.*np.pi*sigma2))
        
        mGX = e1h*GX + e2h*GX_2h
        sigma2 = egx**2
        LGX = -0.5 * np.sum((mGX - gx)**2 / sigma2 + np.log(2.*np.pi*sigma2))
        
        if fit_components == 'both':
            L = LGT +  LGX
        if fit_components == 'tangential':
            L = LGT
        if fit_components == 'cross':
            L = LGX

        return L
    
    
    def log_probability(data_model, R, profiles, eprofiles):
        
        q1h, q2h = data_model
        
        if 0. < q1h < 1. and 0. < q2h < 1.:
            return log_likelihood(data_model, R, profiles, eprofiles)
            
        return -np.inf
    
    # initializing
    
    pos = np.array([np.random.uniform(0.6,0.9,15),
                    np.random.uniform(0.1,0.5,15)]).T
    
    nwalkers, ndim = pos.shape
    
    #-------------------
    # running emcee
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(R,[gt,gx],[egt,egx]))
                                    # pool = pool)
                    
    sampler.run_mcmc(pos, 250, progress=True)
    
    mcmc_out = sampler.get_chain(flat=True).T
    
    return np.median(mcmc_out[0][1500:]),np.median(mcmc_out[1][1500:]),mcmc_out[0],mcmc_out[1]

def fit_quadrupoles_2terms_qrfunc(R,gt,gx,egt,egx,GT,GX,GT_2h,GX_2h,fit_components):
    
    print('fitting components: ',fit_components)
    def log_likelihood(data_model, R, profiles, eprofiles):
        
        a, b, q2h = data_model
        q1h = b*R**a
        
        gt, gx   = profiles
        egt, egx = eprofiles
        
        e1h   = (1.-q1h)/(1.+q1h)
        e2h   = (1.-q2h)/(1.+q2h)
        
        sigma2 = egt**2
        mGT = e1h*GT + e2h*GT_2h
        LGT = -0.5 * np.sum((mGT - gt)**2 / sigma2 + np.log(2.*np.pi*sigma2))
        
        mGX = e1h*GX + e2h*GX_2h
        sigma2 = egx**2
        LGX = -0.5 * np.sum((mGX - gx)**2 / sigma2 + np.log(2.*np.pi*sigma2))
        
        if fit_components == 'both':
            L = LGT +  LGX
        if fit_components == 'tangential':
            L = LGT
        if fit_components == 'cross':
            L = LGX

        return L
    
    
    def log_probability(data_model, R, profiles, eprofiles):
        
        a, b, q2h = data_model
        
        if -0.5 < a < 0. and 0. < b < 1. and 0. < q2h < 1.:
            return log_likelihood(data_model, R, profiles, eprofiles)
            
        return -np.inf
    
    # initializing
    
    pos = np.array([np.random.uniform(-0.1,0.,15),
                    np.random.uniform(0.6,0.9,15),
                    np.random.uniform(0.1,0.5,15)]).T
    
    nwalkers, ndim = pos.shape
    
    #-------------------
    # running emcee
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(R,[gt,gx],[egt,egx]))
                                    # pool = pool)
                    
    sampler.run_mcmc(pos, 1000, progress=True)
    
    mcmc_out = sampler.get_chain(flat=True).T
    
    return np.median(mcmc_out[0][1500:]),np.median(mcmc_out[1][1500:]),np.median(mcmc_out[2][1500:]),mcmc_out[0],mcmc_out[1],mcmc_out[2]


def fit_gamma_components(DF):

    # FIT SHEAR QUADRUPOLE PROFILES
            
    GT_func,GX_func = GAMMA_components_parallel(DF.r,0.,ellip=1.,M200 = 10**DF.lM200_ds,c200=DF.c200_ds,cosmo_params=params,terms='1h',ncores=10)
    GT_2h_func,GX_2h_func = GAMMA_components_parallel(DF.r,0.,ellip=1.,M200 = 10**DF.lM200_ds,c200=DF.c200_ds,cosmo_params=params,terms='2h',ncores=10)
    S2    = S2_quadrupole(DF.r,0.,M200 = 10**DF.lM200_ds,c200=DF.c200_ds,terms='1h',cosmo_params=params)
    S2_2h = S2_quadrupole(DF.r,0.,M200 = 10**DF.lM200_ds,c200=DF.c200_ds,terms='2h',cosmo_params=params)
                                                                             
    q_s2,q2h_q_s2,mcmc_q_s2,mcmc_q2h_q_s2 = fit_S2_2terms(DF.r,DF.S2,DF.e_S2,S2,S2_2h)
    
    q_x,q2h_q_x,mcmc_q_x,mcmc_q2h_q_x = fit_quadrupoles_2terms(DF.r,DF.GT,DF.GX,DF.e_GT,DF.e_GX,GT_func,GX_func,GT_2h_func,GX_2h_func,'cross')
    a_x,b_x,q2h_ab_x,mcmc_a_x,mcmc_b_x,mcmc_q2h_ab_x = fit_quadrupoles_2terms_qrfunc(DF.r,DF.GT,DF.GX,DF.e_GT,DF.e_GX,GT_func,GX_func,GT_2h_func,GX_2h_func,'cross')

    q_t,q2h_q_t,mcmc_q_t,mcmc_q2h_q_t = fit_quadrupoles_2terms(DF.r,DF.GT,DF.GX,DF.e_GT,DF.e_GX,GT_func,GX_func,GT_2h_func,GX_2h_func,'tangential')
    a_t,b_t,q2h_ab_t,mcmc_a_t,mcmc_b_t,mcmc_q2h_ab_t = fit_quadrupoles_2terms_qrfunc(DF.r,DF.GT,DF.GX,DF.e_GT,DF.e_GX,GT_func,GX_func,GT_2h_func,GX_2h_func,'tangential')

    q,q2h_q,mcmc_q,mcmc_q2h_q = fit_quadrupoles_2terms(DF.r,DF.GT,DF.GX,DF.e_GT,DF.e_GX,GT_func,GX_func,GT_2h_func,GX_2h_func,'both')
    a,b,q2h_ab,mcmc_a,mcmc_b,mcmc_q2h_ab = fit_quadrupoles_2terms_qrfunc(DF.r,DF.GT,DF.GX,DF.e_GT,DF.e_GX,GT_func,GX_func,GT_2h_func,GX_2h_func,'both')
    a,b,q2h,mcmc_a,mcmc_b,mcmc_q2h = fit_quadrupoles_2terms_qrfunc(DF.r,DF.GT,DF.GX,DF.e_GT,DF.e_GX,GT_func,GX_func,GT_2h_func,GX_2h_func,'both')
    
    
    return a,b,q2h,mcmc_a,mcmc_b,mcmc_q2h


class pack():

    def __init__(self, name):
      
      self.name = name
      
      try:
        self.load()
      except OSError as err:
        print("OS error:", err)
        
    def load(self):
      # Esta funcion carga el archivo
      # estan guardados todos esos campos
      # self.main es el dataframe de los halos maskareado para ese tipo de muestra

      f = open(self.name, 'rb')
      self.DS1h_fit     = np.load(f) 
      self.DS2h_fit     = np.load(f) 
      self.DS_fit       = np.load(f) 
      self.lM200_ds     = np.load(f)      
      self.c200_ds      = np.load(f)       
      self.e_c200_ds    = np.load(f)     
      self.e_lM200_ds   = np.load(f)    
      self.mcmc_ds_lM   = np.load(f)    
      self.mcmc_ds_c200 = np.load(f)  
      self.q1h_2g       = np.load(f)        
      self.q2h_2g       = np.load(f)        
      self.mcmc_q1h_2g  = np.load(f)   
      self.mcmc_q2h_2g  = np.load(f)   
      self.GT1h_fit2    = np.load(f)     
      self.GX1h_fit2    = np.load(f)     
      self.GT2h_fit2    = np.load(f)     
      self.GX2h_fit2    = np.load(f)     
      self.q1h_gt       = np.load(f)        
      self.q2h_gt       = np.load(f)        
      self.mcmc_q1h_gt  = np.load(f)   
      self.mcmc_q2h_gt  = np.load(f)   
      self.GT1h         = np.load(f)          
      self.GT2h         = np.load(f)          
      self.q1h_gx       = np.load(f)        
      self.q2h_gx       = np.load(f)        
      self.mcmc_q1h_gx  = np.load(f)   
      self.mcmc_q2h_gx  = np.load(f)   
      self.GX1h         = np.load(f)          
      self.GX2h         = np.load(f)          
      self.r            = np.load(f) 
      self.DS_T         = np.load(f) 
      self.e_DS_T       = np.load(f)
      self.GT           = np.load(f)
      self.e_GT         = np.load(f)
      self.GX           = np.load(f)
      self.e_GX         = np.load(f)
      self.S            = np.load(f)
      self.e_S          = np.load(f)
      self.S2           = np.load(f)
      self.e_S2         = np.load(f)
      self.rs           = np.load(f) 
      self.qs           = np.load(f) 
      self.err_qs       = np.load(f) 
      self.main         = pickle.loads(np.load(f, allow_pickle=True).item())
      f.close()

      # guardo estos atributos extras para graficarlos
      self.q2d          = np.concatenate((self.main.b2D_xy/self.main.a2D_xy,self.main.b2D_zx/self.main.a2D_zx,self.main.b2D_yz/self.main.a2D_yz))
      self.q2dr         = np.concatenate((self.main.b2Dr_xy/self.main.a2Dr_xy,self.main.b2Dr_zx/self.main.a2Dr_zx,self.main.b2Dr_yz/self.main.a2Dr_yz))
      self.q2d_it       = np.concatenate((self.main.b2D_it_xy/self.main.a2D_it_xy,self.main.b2D_it_zx/self.main.a2D_it_zx,self.main.b2D_it_yz/self.main.a2D_it_yz))
      self.q2dr_it      = np.concatenate((self.main.b2Dr_it_xy/self.main.a2Dr_it_xy,self.main.b2Dr_it_zx/self.main.a2Dr_it_zx,self.main.b2Dr_it_yz/self.main.a2Dr_it_yz))

# Globales
typetensor = ['standard', 'reduced']
folders_list = ["total", "relajados", \
                "relajados_cerca", "relajados_lejos", \
                "relajados_masivos", "relajados_no_masivos"]

lhs = ["Total", "Relaxed", \
                "Denser", "Isolated", \
                "Higher-mass", "Lower-mass"]

# La carpeta donde estan guarados los objetos
input_folder  = "./arreglos/"      

ti = time.time()

# FIT QUADRUPOLES
def analyse_new_model():
    
    a_dm   = []
    b_dm   = []
    q2h_dm = []
    mcmc_a_dm   = []
    mcmc_b_dm   = []
    mcmc_q2h_dm = []
    
    a_sidm   = []
    b_sidm   = []
    q2h_sidm = []
    mcmc_a_sidm = []
    mcmc_b_sidm = []
    mcmc_q2h_sidm = []

    
    for idx, name_tensor in enumerate(typetensor):

        f, ax_all = plt.subplots(6,3, figsize=(14,16),sharex = True)
        f.subplots_adjust(hspace=0)                

        
        for jdx, name_folder in enumerate(folders_list):
            
            row, col = jdx // 2, jdx % 2
        
            filename_DM = input_folder + "%s_DM_%s.npy" % (name_folder, name_tensor)
            # llama a la clase pack que lee el archivo y lo carga
            DM = pack(filename_DM)
            
            a,b,q2h,mcmc_a,mcmc_b,mcmc_q2h = fit_gamma_components(DM)
            
            data_model_dm = [a,b,q2h]
            
            a_dm   += [a]
            b_dm   += [b]
            q2h_dm += [q2h]
            mcmc_a_dm   += [mcmc_a]
            mcmc_b_dm   += [mcmc_b]
            mcmc_q2h_dm += [mcmc_q2h]
    
        
            filename_SIDM = input_folder + "%s_SIDM_%s.npy" % (name_folder, name_tensor)
            # llama a la clase pack que lee el archivo y lo carga
            SIDM = pack(filename_SIDM)
            a,b,q2h,mcmc_a,mcmc_b,mcmc_q2h = fit_gamma_components(SIDM)
            data_model_sidm = [a,b,q2h]
            
            a_sidm   += [a]
            b_sidm   += [b]
            q2h_sidm += [q2h]
            mcmc_a_sidm   += [mcmc_a]
            mcmc_b_sidm   += [mcmc_b]
            mcmc_q2h_sidm += [mcmc_q2h]
            
            ax_all[jdx,0].text(1,100,lhs[jdx],fontsize=14)
            plt_profile_fitted_final_new(DM,SIDM,0,5000,ax_all[jdx],data_model_dm, data_model_sidm)
            ax_all[0,0].legend(loc=3,frameon=False,fontsize=10)
            ax_all[0,1].legend(loc=3,frameon=False,fontsize=10)
        
        f.savefig('../final_plots/profile_'+name_tensor+'_new_model_cross.pdf',bbox_inches='tight')
    

    
    fa, axa = plt.subplots(6,2, figsize=(14,16),sharex = True, sharey = True)
    fa.subplots_adjust(hspace=0,wspace=0)
    axa = axa.flatten()
    fb, axb = plt.subplots(6,2, figsize=(14,16),sharex = True, sharey = True)
    fb.subplots_adjust(hspace=0,wspace=0)
    axb = axb.flatten()
    fq2h, axq2h = plt.subplots(6,2, figsize=(14,16),sharex = True, sharey = True)
    fq2h.subplots_adjust(hspace=0,wspace=0)
    axq2h = axq2h.flatten()


    for j in range(12):
        axa[j].plot(mcmc_a_dm[j],alpha=0.5)
        axa[j].plot(mcmc_a_sidm[j],alpha=0.5)
        axb[j].plot(mcmc_b_dm[j],alpha=0.5)
        axb[j].plot(mcmc_b_sidm[j],alpha=0.5)
        axq2h[j].plot(mcmc_q2h_dm[j],alpha=0.5)
        axq2h[j].plot(mcmc_q2h_sidm[j],alpha=0.5)
        
        
        axa[j].set_ylabel('a')
        axa[j].set_xlabel('N')
        axb[j].set_ylabel('b')
        axb[j].set_xlabel('N')
        axq2h[j].set_ylabel('q2h')
        axq2h[j].set_xlabel('N')
    

# El primer for es sobre una lista que tiene los tipos de tensores
# standard o reducido
for idx, name_tensor in enumerate(typetensor):
    
    f, ax_all = plt.subplots(6,3, figsize=(14,16),sharex = True)
    f.subplots_adjust(hspace=0)

    '''
    fdist, ax_dist = plt.subplots(3,2, figsize=(10,8),sharex = True,sharey = True)
    fdist.subplots_adjust(hspace=0,wspace=0)

    fdistr, ax_distr = plt.subplots(3,2, figsize=(10,8),sharex = True,sharey = True)
    fdistr.subplots_adjust(hspace=0,wspace=0)
    '''
    
    fdist_it, ax_dist_it = plt.subplots(3,2, figsize=(10,8),sharex = True,sharey = True)
    fdist_it.subplots_adjust(hspace=0,wspace=0)

    fdistr_it, ax_distr_it = plt.subplots(3,2, figsize=(10,8),sharex = True,sharey = True)
    fdistr_it.subplots_adjust(hspace=0,wspace=0)

    fcomp, ax_comp = plt.subplots(2,1, figsize=(10,10),sharex = True)
    fcomp.subplots_adjust(hspace=0,wspace=0)

    fcomp2, ax_comp2 = plt.subplots(1,2, figsize=(10,6))

    
    ax_comp[0].axhspan(-0.05,0.05,color='C7',alpha=0.1)
    ax_comp[0].plot([0,7],[0,0],'k--')
    ax_comp[1].axhspan(-0.05,0.05,color='C7',alpha=0.1)
    ax_comp[1].plot([0,7],[0,0],'k--')
    

  # El segundo for es sobre una lista que tiene los nombres de las submuestras
  # total, relajados, relajados con el 5to vecino cerca, etc.
    for jdx, name_folder in enumerate(folders_list):
        
        row, col = jdx // 2, jdx % 2
    
        filename_DM = input_folder + "%s_DM_%s.npy" % (name_folder, name_tensor)
        # llama a la clase pack que lee el archivo y lo carga
        DM = pack(filename_DM)
    
        filename_SIDM = input_folder + "%s_SIDM_%s.npy" % (name_folder, name_tensor)
        # llama a la clase pack que lee el archivo y lo carga
        SIDM = pack(filename_SIDM)
        
        corner_result(DM,SIDM,lhs[jdx],name_tensor)
    
        
        plt_profile_fitted_final(DM,SIDM,0,5000,ax_all[jdx])
        ax_all[jdx,0].text(1,100,lhs[jdx],fontsize=14)
        
        '''
        plot_q_dist(DM,SIDM,ax_dist[row,col],method='')
        ax_dist[row,col].text(0.2,4,lhs[jdx],fontsize=14)

        plot_q_dist(DM,SIDM,ax_distr[row,col],method='r')
        ax_distr[row,col].text(0.2,4,lhs[jdx],fontsize=14)
        '''
        
        plot_q_dist(DM,SIDM,ax_dist_it[row,col],method='_it')
        ax_dist_it[row,col].text(0.2,4,lhs[jdx],fontsize=14)

        plot_q_dist(DM,SIDM,ax_distr_it[row,col],method='r_it')
        ax_distr_it[row,col].text(0.2,4,lhs[jdx],fontsize=14)
        
        compare_q(DM,SIDM,ax_comp,jdx+1)
        linear_compare_q(DM,SIDM,ax_comp2,jdx+1)
        
        if jdx == 0:
            ax_comp[0].legend(loc=2,frameon=False,fontsize=10)
            ax_comp[1].legend(loc=2,frameon=False,fontsize=10)
            
    ax_comp[1].set_xticks(np.arange(jdx+1)+1)
    ax_comp[1].set_xticklabels(lhs)


    ax_all[0,0].legend(loc=3,frameon=False,fontsize=10)
    ax_all[0,1].legend(loc=3,frameon=False,fontsize=10)
    # ax_dist[0,0].legend(loc=3,frameon=False,fontsize=10)
    # ax_distr[0,0].legend(loc=3,frameon=False,fontsize=10)
    ax_dist_it[0,0].legend(loc=3,frameon=False,fontsize=10)
    ax_distr_it[0,0].legend(loc=3,frameon=False,fontsize=10)

    f.savefig('../final_plots/profile_'+name_tensor+'_v2.pdf',bbox_inches='tight')
    fcomp.savefig('../final_plots/compare_'+name_tensor+'_v2.pdf',bbox_inches='tight')
    fcomp2.savefig('../final_plots/compare2_'+name_tensor+'_v2.pdf',bbox_inches='tight')
    # fdist.savefig('../final_plots/dist_std_'+name_tensor+'.png',bbox_inches='tight')
    fdist_it.savefig('../final_plots/dist_std_it_'+name_tensor+'.png',bbox_inches='tight')
    # fdistr.savefig('../final_plots/dist_red_'+name_tensor+'.png',bbox_inches='tight')
    fdistr_it.savefig('../final_plots/dist_red_it_'+name_tensor+'.png',bbox_inches='tight')
    fdistr_it.savefig('../final_plots/dist_red_it_'+name_tensor+'_v2.pdf',bbox_inches='tight')
    
tf = time.time()
print("Total Time %.2f\n" % (tf-ti))
