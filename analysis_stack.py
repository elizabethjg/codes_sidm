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
# typetensor = ['standard', 'reduced']
typetensor = ['reduced']
folders_list = ["total", "relajados", \
                "relajados_cerca", "relajados_lejos", \
                "relajados_masivos", "relajados_no_masivos"]

lhs = ["Total", "Relaxed", \
                "Clustered", "Isolated", \
                "Massive", "Lower-mass"]

# La carpeta donde estan guarados los objetos
input_folder  = "./arreglos/"      

ti = time.time()

lM200_dm = np.array([])
c200_dm  = np.array([])
q1h_dm   = np.array([])
q2h_dm   = np.array([])
q1hr_dm  = np.array([])
ar_dm    = np.array([])
q2hr_dm  = np.array([])
q_dm     = np.array([])
q0_dm    = np.array([])
a_dm     = np.array([])

lM200_sidm = np.array([])
c200_sidm  = np.array([])
q1h_sidm   = np.array([])
q2h_sidm   = np.array([])
q1hr_sidm  = np.array([])
ar_sidm    = np.array([])
q2hr_sidm  = np.array([])
q_sidm     = np.array([])
q0_sidm    = np.array([])
a_sidm     = np.array([])

    

# El primer for es sobre una lista que tiene los tipos de tensores
# standard o reducido
for idx, name_tensor in enumerate(typetensor):
    
    
    f, ax_all = plt.subplots(2,4, figsize=(12,5),sharex = True,sharey = True)
    ax_all = ax_all.flatten()
    ax_all[2].axis('off')
    ax_all[3].axis('off')
    ax_all = np.append(ax_all[:2],ax_all[4:])
    f.subplots_adjust(hspace=0,wspace=0)

    ft, ax_T = plt.subplots(2,6, figsize=(14,5),sharex = True)
    ax_T = ax_T.T
    ft.subplots_adjust(hspace=0,wspace=0)

    fx, ax_X = plt.subplots(2,6, figsize=(14,5),sharex = True)
    ax_X = ax_X.T
    fx.subplots_adjust(hspace=0,wspace=0)

    fstack, ax_stack = plt.subplots(2,4, figsize=(14,6),sharex = True,sharey = True)
    ax_stack = ax_stack.flatten()
    ax_stack[2].axis('off')
    ax_stack[3].axis('off')
    ax_stack = np.append(ax_stack[:2],ax_stack[4:])
    fstack.subplots_adjust(hspace=0,wspace=0)
    
    frad, ax_rad = plt.subplots(2,4, figsize=(14,6),sharex = True,sharey = True)
    ax_rad = ax_rad.flatten()
    ax_rad[2].axis('off')
    ax_rad[3].axis('off')
    ax_rad = np.append(ax_rad[:2],ax_rad[4:])
    frad.subplots_adjust(hspace=0,wspace=0)
    
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

    fcomp_2g2, ax_comp_2g2 = plt.subplots(2,1, figsize=(10,10),sharex = True)
    fcomp_2g2.subplots_adjust(hspace=0,wspace=0)

    for ind in [0,1]:
        ax_comp_x[ind].axhspan(-0.05,0.05,color='C7',alpha=0.1)
        ax_comp_x[ind].plot([0,7],[0,0],'k--')
        ax_comp_t[ind].axhspan(-0.05,0.05,color='C7',alpha=0.1)
        ax_comp_t[ind].plot([0,7],[0,0],'k--')
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


        # MAKE PROFILE
        ax_T[jdx,0].plot(0.11,50,'w,',label=lhs[jdx])
        ax_X[jdx,0].plot(0.11,50,'w,',label=lhs[jdx])
        plt_profile_fitted_final(DM,SIDM,0,5000,[ax_all[jdx],ax_T[jdx,0],ax_T[jdx,1]],jdx)
        # plt_profile_fitted_final_new(DM,SIDM,0,5000,[ax_X[jdx,0],ax_X[jdx,1]],jdx)
        ax_all[jdx].text(0.5,100,lhs[jdx],fontsize=14)
        
        
        
        corner_result(DM,SIDM,lhs[jdx],name_tensor)
        
        stacked_particle(DM,SIDM,ax_stack[jdx],lhs[jdx])
        
                
        plot_q_dist(DM,SIDM,ax_dist_it[row,col],method='_it')
        ax_dist_it[row,col].text(0.2,4,lhs[jdx],fontsize=14)

        plot_q_dist(DM,SIDM,ax_distr_it[row,col],method='r_it')
        ax_distr_it[row,col].text(0.2,4,lhs[jdx],fontsize=14)
        
        compare_q(DM,SIDM,ax_comp_x,jdx+1,method='gx')
        compare_q(DM,SIDM,ax_comp_t,jdx+1,method='gt')
        compare_q(DM,SIDM,[ax_comp_2g[0],ax_comp_2g2[0]],jdx+1,method='2g')

        fit = compare_qr(DM,SIDM,[ax_comp_2g[1],ax_comp_2g2[1]],jdx+1,method='2g_fb')

        
        if jdx == 0:
            ax_comp_x[0].legend(loc=2,frameon=False,fontsize=10,ncol=2)
            ax_comp_x[1].legend(loc=2,frameon=False,fontsize=10,ncol=2)
            ax_comp_t[0].legend(loc=2,frameon=False,fontsize=10,ncol=2)
            ax_comp_t[1].legend(loc=2,frameon=False,fontsize=10,ncol=2)
            ax_comp_2g[0].legend(loc=2,frameon=False,fontsize=10,ncol=2)
            ax_comp_2g[1].legend(loc=2,frameon=False,fontsize=10,ncol=2)

            ax_comp_2g2[0].legend(loc=2,frameon=False,fontsize=10,ncol=2)
            ax_comp_2g2[1].legend(loc=2,frameon=False,fontsize=10,ncol=2)
            
        qplot(DM,SIDM,ax_rad[jdx])
        
        lM200_dm = np.append(lM200_dm,DM.lM200_ds)
        c200_dm  = np.append(c200_dm,DM.c200_ds) 
        q1h_dm   = np.append(q1h_dm,DM.q1h_2g)  
        q2h_dm   = np.append(q2h_dm,DM.q2h_2g)    
        q1hr_dm  = np.append(q1hr_dm,DM.b_2g_fb) 
        ar_dm    = np.append(ar_dm,DM.a_2g_fb) 
        q2hr_dm  = np.append(q2hr_dm,DM.q2hr_2g_fb) 
        q_dm     = np.append(q_dm,np.mean(DM.q2dr_it))
        q0_dm    = np.append(q0_dm,fit[0])
        a_dm     = np.append(a_dm ,fit[1])
        
        lM200_sidm = np.append(lM200_sidm,SIDM.lM200_ds)
        c200_sidm  = np.append(c200_sidm,SIDM.c200_ds) 
        q1h_sidm   = np.append(q1h_sidm,SIDM.q1h_2g)  
        q2h_sidm   = np.append(q2h_sidm,SIDM.q2h_2g)    
        q1hr_sidm  = np.append(q1hr_sidm,SIDM.b_2g_fb) 
        ar_sidm    = np.append(ar_sidm,SIDM.a_2g_fb) 
        q2hr_sidm  = np.append(q2hr_sidm,SIDM.q2hr_2g_fb)
        q_sidm     = np.append(q_sidm,np.mean(SIDM.q2dr_it))
        q0_sidm    = np.append(q0_sidm,fit[2])
        a_sidm     = np.append(a_sidm ,fit[3])
            
            
    ax_comp_x[1].set_xticks(np.arange(jdx+1)+1)
    ax_comp_x[1].set_xticklabels(lhs)
    ax_comp_t[1].set_xticks(np.arange(jdx+1)+1)
    ax_comp_t[1].set_xticklabels(lhs)
    ax_comp_2g[1].set_xticks(np.arange(jdx+1)+1)
    ax_comp_2g[1].set_xticklabels(lhs)

    ax_comp_2g2[1].set_xticks(np.arange(jdx+1)+1)
    ax_comp_2g2[1].set_xticklabels(lhs)


    ax_dist_it[0,0].legend(loc=3,frameon=False,fontsize=10)
    ax_distr_it[0,0].legend(loc=3,frameon=False,fontsize=10)

    for j in range(len(ax_stack)):
        if j == 0 or j == 2:
            ax_stack[j].set_ylabel('q')

        if j !=0:
            ax_T[j,0].yaxis.set_ticklabels([])
            ax_X[j,0].yaxis.set_ticklabels([])
            ax_T[j,1].yaxis.set_ticklabels([])
            ax_X[j,1].yaxis.set_ticklabels([])


    fstack.savefig('../final_plots/stack_'+name_tensor+'.pdf',bbox_inches='tight')
    fdist_it.savefig('../final_plots/dist_'+name_tensor+'.png',bbox_inches='tight')
    fdistr_it.savefig('../final_plots/dist_'+name_tensor+'_r.png',bbox_inches='tight')

    f.savefig('../final_plots/profile2_'+name_tensor+'.pdf',bbox_inches='tight')
    ft.savefig('../final_plots/profile_'+name_tensor+'_T.pdf',bbox_inches='tight')
    fx.savefig('../final_plots/profile_'+name_tensor+'_X.pdf',bbox_inches='tight')
    fcomp_2g.savefig('../final_plots/compare_'+name_tensor+'.png',bbox_inches='tight')
    fcomp_t.savefig('../final_plots/compare_'+name_tensor+'_t.png',bbox_inches='tight')
    fcomp_x.savefig('../final_plots/compare_'+name_tensor+'_x.png',bbox_inches='tight')

    fcomp_2g2.savefig('../final_plots/compare_'+name_tensor+'_v2.pdf',bbox_inches='tight')
    
    

    
tf = time.time()
print("Total Time %.2f\n" % (tf-ti))
