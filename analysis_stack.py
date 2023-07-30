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
params = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.045, 'sigma8': 0.811, 'ns': 0.96}

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
      self.DS1h_fit        = np.load(f)
      self.DS2h_fit        = np.load(f)
      self.DS_fit          = np.load(f)
      self.lM200_ds        = np.load(f)
      self.c200_ds         = np.load(f)
      self.e_c200_ds       = np.load(f)
      self.e_lM200_ds      = np.load(f)
      self.mcmc_ds_lM      = np.load(f)
      self.mcmc_ds_c200    = np.load(f)
      self.q1h_2g          = np.load(f)
      self.q2h_2g          = np.load(f)
      self.mcmc_q1h_2g     = np.load(f)
      self.mcmc_q2h_2g     = np.load(f)
      self.GT1h_fit2       = np.load(f)
      self.GX1h_fit2       = np.load(f)
      self.GT2h_fit2       = np.load(f)
      self.GX2h_fit2       = np.load(f)
      self.q1h_gt          = np.load(f)
      self.q2h_gt          = np.load(f)
      self.mcmc_q1h_gt     = np.load(f)
      self.mcmc_q2h_gt     = np.load(f)
      self.GT1h            = np.load(f)
      self.GT2h            = np.load(f)
      self.q1h_gx          = np.load(f)
      self.q2h_gx          = np.load(f)
      self.mcmc_q1h_gx     = np.load(f)
      self.mcmc_q2h_gx     = np.load(f)
      self.GX1h            = np.load(f)
      self.GX2h            = np.load(f)
      self.a_2g            = np.load(f)
      self.b_2g            = np.load(f)
      self.q2hr_2g         = np.load(f)
      self.mcmc_a_2g       = np.load(f)
      self.mcmc_b_2g       = np.load(f)
      self.mcmc_q2hr_2g    = np.load(f)
      self.GT1hr_fit2      = np.load(f)
      self.GX1hr_fit2      = np.load(f)
      self.GT2hr_fit2      = np.load(f)
      self.GX2hr_fit2      = np.load(f)
      self.a_gx            = np.load(f)
      self.b_gx            = np.load(f)
      self.q2hr_gx         = np.load(f)
      self.mcmc_a_gx       = np.load(f)
      self.mcmc_b_gx       = np.load(f)
      self.mcmc_q2hr_gx    = np.load(f)
      self.GX1hr           = np.load(f)
      self.GX2hr           = np.load(f)
      self.a_gt            = np.load(f)
      self.b_gt            = np.load(f)
      self.q2hr_gt         = np.load(f)
      self.mcmc_a_gt       = np.load(f)
      self.mcmc_b_gt       = np.load(f)
      self.mcmc_q2hr_gt    = np.load(f)
      self.GT1hr           = np.load(f)
      self.GT2hr           = np.load(f)
      self.r               = np.load(f)
      self.DS_T            = np.load(f)
      self.e_DS_T          = np.load(f)
      self.GT              = np.load(f)
      self.e_GT            = np.load(f)
      self.GX              = np.load(f)
      self.e_GX            = np.load(f)
      self.S               = np.load(f)
      self.e_S             = np.load(f)
      self.S2              = np.load(f)
      self.e_S2            = np.load(f)
      self.rs              = np.load(f)
      self.qs              = np.load(f)
      self.err_qs          = np.load(f)
      self.a_2g_fb         = np.load(f)
      self.b_2g_fb         = np.load(f)
      self.q2hr_2g_fb      = np.load(f)
      self.mcmc_a_2g_fb    = np.load(f)
      self.mcmc_b_2g_fb    = np.load(f)
      self.mcmc_q2hr_2g_fb = np.load(f)
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

q_1h_T_dm = np.array([])
q_1h_X_dm = np.array([])
q_2h_T_dm = np.array([])
q_2h_X_dm = np.array([])
q_1h_T_sidm = np.array([])
q_1h_X_sidm = np.array([])
q_2h_T_sidm = np.array([])
q_2h_X_sidm = np.array([])

a_T_dm = np.array([])
a_X_dm = np.array([])
b_T_dm = np.array([])
b_X_dm = np.array([])
q_2hr_T_dm = np.array([])
q_2hr_X_dm = np.array([])
a_T_sidm = np.array([])
a_X_sidm = np.array([])
b_T_sidm = np.array([])
b_X_sidm = np.array([])
q_2hr_T_sidm = np.array([])
q_2hr_X_sidm = np.array([])
    

# El primer for es sobre una lista que tiene los tipos de tensores
# standard o reducido
for idx, name_tensor in enumerate(typetensor):
    
    
    f, ax_all = plt.subplots(6,3, figsize=(14,16),sharex = True)
    f.subplots_adjust(hspace=0)

    f2, ax_all2 = plt.subplots(6,3, figsize=(14,16),sharex = True)
    f2.subplots_adjust(hspace=0)
    
    fdist_it, ax_dist_it = plt.subplots(3,2, figsize=(10,8),sharex = True,sharey = True)
    fdist_it.subplots_adjust(hspace=0,wspace=0)

    fdistr_it, ax_distr_it = plt.subplots(3,2, figsize=(10,8),sharex = True,sharey = True)
    fdistr_it.subplots_adjust(hspace=0,wspace=0)

    fcomp_x, ax_comp_x = plt.subplots(2,1, figsize=(10,10),sharex = True)
    fcomp_x.subplots_adjust(hspace=0,wspace=0)

    fcomp_t, ax_comp_t = plt.subplots(2,1, figsize=(10,10),sharex = True)
    fcomp_t.subplots_adjust(hspace=0,wspace=0)

    fcomp_2g, ax_comp_2g = plt.subplots(2,1, figsize=(10,10),sharex = True)
    fcomp_2g.subplots_adjust(hspace=0,wspace=0)

    fcomp_x2, ax_comp_x2 = plt.subplots(2,1, figsize=(10,10),sharex = True)
    fcomp_x2.subplots_adjust(hspace=0,wspace=0)

    fcomp_t2, ax_comp_t2 = plt.subplots(2,1, figsize=(10,10),sharex = True)
    fcomp_t2.subplots_adjust(hspace=0,wspace=0)

    fcomp_2g2, ax_comp_2g2 = plt.subplots(2,1, figsize=(10,10),sharex = True)
    fcomp_2g2.subplots_adjust(hspace=0,wspace=0)

    for ind in [0,1]:
        ax_comp_x[ind].axhspan(-0.05,0.05,color='C7',alpha=0.1)
        ax_comp_x[ind].plot([0,7],[0,0],'k--')
        ax_comp_x2[ind].axhspan(-0.05,0.05,color='C7',alpha=0.1)
        ax_comp_x2[ind].plot([0,7],[0,0],'k--')
        ax_comp_t[ind].axhspan(-0.05,0.05,color='C7',alpha=0.1)
        ax_comp_t[ind].plot([0,7],[0,0],'k--')
        ax_comp_t2[ind].axhspan(-0.05,0.05,color='C7',alpha=0.1)
        ax_comp_t2[ind].plot([0,7],[0,0],'k--')
        ax_comp_2g[ind].axhspan(-0.05,0.05,color='C7',alpha=0.1)
        ax_comp_2g[ind].plot([0,7],[0,0],'k--')
        ax_comp_2g2[ind].axhspan(-0.05,0.05,color='C7',alpha=0.1)
        ax_comp_2g2[ind].plot([0,7],[0,0],'k--')
    

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
        
        q_1h_T_dm = np.append(q_1h_T_dm,DM.q1h_gt)
        q_1h_X_dm = np.append(q_1h_X_dm,DM.q1h_gx)
        q_2h_T_dm = np.append(q_2h_T_dm,DM.q2h_gt)
        q_2h_X_dm = np.append(q_2h_X_dm,DM.q2h_gx)
        q_1h_T_sidm = np.append(q_1h_T_sidm,SIDM.q1h_gt)
        q_1h_X_sidm = np.append(q_1h_X_sidm,SIDM.q1h_gx)
        q_2h_T_sidm = np.append(q_2h_T_sidm,SIDM.q2h_gt)
        q_2h_X_sidm = np.append(q_2h_X_sidm,SIDM.q2h_gx)

        a_T_dm       = np.append(a_T_dm,DM.a_gt)
        a_X_dm       = np.append(a_X_dm,DM.a_gx)
        b_T_dm       = np.append(b_T_dm,DM.b_gt)
        b_X_dm       = np.append(b_X_dm,DM.b_gx)
        q_2hr_T_dm   = np.append(q_2hr_T_dm,DM.q2hr_gt)
        q_2hr_X_dm   = np.append(q_2hr_X_dm,DM.q2hr_gx)
        a_T_sidm     = np.append(a_T_sidm,SIDM.a_gt)
        a_X_sidm     = np.append(a_X_sidm,SIDM.a_gx)
        b_T_sidm     = np.append(b_T_sidm,SIDM.b_gt)
        b_X_sidm     = np.append(b_X_sidm,SIDM.b_gx)
        q_2hr_T_sidm = np.append(q_2hr_T_sidm,SIDM.q2hr_gt)
        q_2hr_X_sidm = np.append(q_2hr_X_sidm,SIDM.q2hr_gx)
        
        corner_result(DM,SIDM,lhs[jdx],name_tensor)
            
        plt_profile_fitted_final(DM,SIDM,0,5000,ax_all[jdx])
        ax_all[jdx,0].text(1,100,lhs[jdx],fontsize=14)

        plt_profile_fitted_final_new(DM,SIDM,0,5000,ax_all2[jdx])
        ax_all2[jdx,0].text(1,100,lhs[jdx],fontsize=14)
                
        plot_q_dist(DM,SIDM,ax_dist_it[row,col],method='_it')
        ax_dist_it[row,col].text(0.2,4,lhs[jdx],fontsize=14)

        plot_q_dist(DM,SIDM,ax_distr_it[row,col],method='r_it')
        ax_distr_it[row,col].text(0.2,4,lhs[jdx],fontsize=14)
        
        compare_q(DM,SIDM,ax_comp_x,jdx+1,method='gx')
        compare_q(DM,SIDM,ax_comp_t,jdx+1,method='gt')
        compare_q(DM,SIDM,ax_comp_2g,jdx+1,method='2g')

        compare_qr(DM,SIDM,ax_comp_x2,jdx+1,method='gx')
        compare_qr(DM,SIDM,ax_comp_t2,jdx+1,method='gt')
        compare_qr(DM,SIDM,ax_comp_2g2,jdx+1,method='2g')

        
        if jdx == 0:
            ax_comp_x[0].legend(loc=2,frameon=False,fontsize=10)
            ax_comp_x[1].legend(loc=2,frameon=False,fontsize=10)
            ax_comp_t[0].legend(loc=2,frameon=False,fontsize=10)
            ax_comp_t[1].legend(loc=2,frameon=False,fontsize=10)
            ax_comp_2g[0].legend(loc=2,frameon=False,fontsize=10)
            ax_comp_2g[1].legend(loc=2,frameon=False,fontsize=10)

            ax_comp_x2[0].legend(loc=2,frameon=False,fontsize=10)
            ax_comp_x2[1].legend(loc=2,frameon=False,fontsize=10)
            ax_comp_t2[0].legend(loc=2,frameon=False,fontsize=10)
            ax_comp_t2[1].legend(loc=2,frameon=False,fontsize=10)
            ax_comp_2g2[0].legend(loc=2,frameon=False,fontsize=10)
            ax_comp_2g2[1].legend(loc=2,frameon=False,fontsize=10)
            
    ax_comp_x[1].set_xticks(np.arange(jdx+1)+1)
    ax_comp_x[1].set_xticklabels(lhs)
    ax_comp_t[1].set_xticks(np.arange(jdx+1)+1)
    ax_comp_t[1].set_xticklabels(lhs)
    ax_comp_2g[1].set_xticks(np.arange(jdx+1)+1)
    ax_comp_2g[1].set_xticklabels(lhs)

    ax_comp_x2[1].set_xticks(np.arange(jdx+1)+1)
    ax_comp_x2[1].set_xticklabels(lhs)
    ax_comp_t2[1].set_xticks(np.arange(jdx+1)+1)
    ax_comp_t2[1].set_xticklabels(lhs)
    ax_comp_2g2[1].set_xticks(np.arange(jdx+1)+1)
    ax_comp_2g2[1].set_xticklabels(lhs)


    ax_all[0,0].legend(loc=3,frameon=False,fontsize=10)
    ax_all[0,1].legend(loc=3,frameon=False,fontsize=10)
    ax_dist_it[0,0].legend(loc=3,frameon=False,fontsize=10)
    ax_distr_it[0,0].legend(loc=3,frameon=False,fontsize=10)

    fdist_it.savefig('../final_plots/dist_'+name_tensor+'.png',bbox_inches='tight')
    fdistr_it.savefig('../final_plots/dist_'+name_tensor+'_r.png',bbox_inches='tight')

    f.savefig('../final_plots/profile2_'+name_tensor+'.png',bbox_inches='tight')
    f2.savefig('../final_plots/profile2_'+name_tensor+'_v2.png',bbox_inches='tight')
    # fcomp_2g.savefig('../final_plots/compare_'+name_tensor+'_v2.pdf',bbox_inches='tight')
    fcomp_2g.savefig('../final_plots/compare_'+name_tensor+'.png',bbox_inches='tight')
    fcomp_t.savefig('../final_plots/compare_'+name_tensor+'_t.png',bbox_inches='tight')
    fcomp_x.savefig('../final_plots/compare_'+name_tensor+'_x.png',bbox_inches='tight')

    fcomp_2g2.savefig('../final_plots/compare_'+name_tensor+'_v2.png',bbox_inches='tight')
    fcomp_t2.savefig('../final_plots/compare_'+name_tensor+'_t_v2.png',bbox_inches='tight')
    fcomp_x2.savefig('../final_plots/compare_'+name_tensor+'_x_v2.png',bbox_inches='tight')

    
tf = time.time()
print("Total Time %.2f\n" % (tf-ti))
