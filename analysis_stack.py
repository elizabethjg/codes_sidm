import sys
import os 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from make_plots import *

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

    f.savefig('../final_plots/profile_'+name_tensor+'.pdf',bbox_inches='tight')
    fcomp.savefig('../final_plots/compare_'+name_tensor+'.pdf',bbox_inches='tight')
    fcomp2.savefig('../final_plots/compare2_'+name_tensor+'.pdf',bbox_inches='tight')
    # fdist.savefig('../final_plots/dist_std_'+name_tensor+'.png',bbox_inches='tight')
    fdist_it.savefig('../final_plots/dist_std_it_'+name_tensor+'.png',bbox_inches='tight')
    # fdistr.savefig('../final_plots/dist_red_'+name_tensor+'.png',bbox_inches='tight')
    fdistr_it.savefig('../final_plots/dist_red_it_'+name_tensor+'.png',bbox_inches='tight')
    fdistr_it.savefig('../final_plots/dist_red_it_'+name_tensor+'.pdf',bbox_inches='tight')
    
tf = time.time()
print("Total Time %.2f\n" % (tf-ti))
