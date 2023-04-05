import sys
import os 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

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
# La carpeta donde estan guarados los objetos
input_folder  = "./arreglos/"      
# La carpeta donde se van guardar los graficos
output_folder = "./graficos/"
ROUT = 5000.

if not os.path.exists(output_folder):
  os.makedirs(output_folder)

ti = time.time()

# El primer for es sobre una lista que tiene los tipos de tensores
# standard o reducido
for idx, name_tensor in enumerate(typetensor):

  fig_S = plt.figure(figsize=(12,12))
  gs_S  = fig_S.add_gridspec(3, 2, hspace=0, wspace=0)
  fig_S.suptitle(name_tensor.title(), fontsize=15, y=0.9)

  fig_S2 = plt.figure(figsize=(12,12))
  gs_S2  = fig_S2.add_gridspec(3, 2, hspace=0, wspace=0)
  fig_S2.suptitle(name_tensor.title(), fontsize=15, y=0.9)

  fig_GX = plt.figure(figsize=(12,12))
  gs_GX  = fig_GX.add_gridspec(3, 2, hspace=0, wspace=0)
  fig_GX.suptitle(name_tensor.title(), fontsize=15, y=0.9)

  fig_GT = plt.figure(figsize=(12,12))
  gs_GT  = fig_GT.add_gridspec(3, 2, hspace=0, wspace=0)
  fig_GT.suptitle(name_tensor.title(), fontsize=15, y=0.9)  
  
  fig_q2d = plt.figure(figsize=(12,12))
  gs_q2d  = fig_q2d.add_gridspec(3, 2, hspace=0, wspace=0)
  fig_q2d.suptitle(name_tensor.title() + ' DM', fontsize=15, y=0.9)  
  
  fig_q2d_it = plt.figure(figsize=(12,12))
  gs_q2d_it  = fig_q2d_it.add_gridspec(3, 2, hspace=0, wspace=0)
  fig_q2d_it.suptitle(name_tensor.title() + ' DM', fontsize=15, y=0.9)

  fig_q2d1 = plt.figure(figsize=(12,12))
  gs_q2d1  = fig_q2d1.add_gridspec(3, 2, hspace=0, wspace=0)
  fig_q2d1.suptitle(name_tensor.title() + ' SIDM', fontsize=15, y=0.9)  
  
  fig_q2d1_it = plt.figure(figsize=(12,12))
  gs_q2d1_it  = fig_q2d1_it.add_gridspec(3, 2, hspace=0, wspace=0)
  fig_q2d1_it.suptitle(name_tensor.title() + ' SIDM', fontsize=15, y=0.9) 

  # El segundo for es sobre una lista que tiene los nombres de las submuestras
  # total, relajados, relajados con el 5to vecino cerca, etc.
  for jdx, name_folder in enumerate(folders_list):
    
    filename_DM = input_folder + "%s_DM_%s.npy" % (name_folder, name_tensor)
    # llama a la clase pack que lee el archivo y lo carga
    DM = pack(filename_DM)

    filename_SIDM = input_folder + "%s_SIDM_%s.npy" % (name_folder, name_tensor)
    # llama a la clase pack que lee el archivo y lo carga
    SIDM = pack(filename_SIDM)
   
    row, col = jdx // 2, jdx % 2
    ax_S  = fig_S.add_subplot(gs_S[row, col])
    ax_S2 = fig_S2.add_subplot(gs_S2[row, col])
    ax_GX = fig_GX.add_subplot(gs_GX[row, col])
    ax_GT = fig_GT.add_subplot(gs_GT[row, col])
    ax_q2d = fig_q2d.add_subplot(gs_q2d[row, col])
    ax_q2d_it = fig_q2d_it.add_subplot(gs_q2d_it[row, col])
    ax_q2d1 = fig_q2d1.add_subplot(gs_q2d1[row, col])
    ax_q2d1_it = fig_q2d1_it.add_subplot(gs_q2d1_it[row, col])
  
    if jdx == 0:
      #################################
      ax_S.plot( SIDM.r , SIDM.S,'C3',  label='SIDM',lw=4)
      ax_S.plot(   DM.r ,   DM.S,'k',   label='CDM',lw=4)
      ax_S.plot( SIDM.r,  SIDM.S,'C5-.',label='SIDM - all part',lw=4)
      ax_S.plot(   DM.r,    DM.S,'C7-.',label='CDM - all part', lw=4)
      #################################
      #################################
      ax_S2.plot( SIDM.r , SIDM.S2,'C3',  label='SIDM',lw=4)
      ax_S2.plot(   DM.r ,   DM.S2,'k',   label='CDM',lw=4)
      ax_S2.plot( SIDM.r,  SIDM.S2,'C5-.',label='SIDM - all part',lw=4)
      ax_S2.plot(   DM.r,    DM.S2,'C7-.',label='CDM - all part', lw=4)
      #################################
      #################################
      ax_GX.plot( SIDM.r , SIDM.GX, 'C3', label='SIDM',lw=4)
      ax_GX.plot(   DM.r ,   DM.GX,'k',label='CDM',lw=4)
      #################################
      #################################
      ax_GT.plot( SIDM.r , SIDM.GT1h,'C3',label='SIDM',lw=4)
      ax_GT.plot( SIDM.r , SIDM.GT1h_fit2,'C3--',alpha=0.5,lw=3)
      ax_GT.plot(   DM.r ,   DM.GT1h,'k' ,label='CDM', lw=4)
      ax_GT.plot(   DM.r ,   DM.GT1h_fit2,'k--',alpha=0.5,label='fit simultaneously',lw=3)
      #################################
      #################################
      ax_q2d.hist(DM.q2d ,np.linspace(0.35,1,20),histtype='step',density=True,label='std',lw=3)
      ax_q2d.hist(DM.q2dr,np.linspace(0.35,1,20),histtype='step',density=True,label='red',lw=3)
      ax_q2d.axvline(np.mean(DM.q2d ),color='C0',lw=3,ls='--')
      ax_q2d.axvline(np.mean(DM.q2dr),color='C1',lw=3,ls='--')
      ax_q2d.axvline(DM.q1h_2g,label='fit G',color='C3',lw=3)
      ax_q2d.axvline(DM.q1h_gt,label='fit GT',color='C4',lw=3)
      ax_q2d.axvline(DM.q1h_gx,label='fit GX',color='C5',lw=3)
      ax_q2d.axvline(DM.q2h_2g,label='fit G',color='C3',lw=3,ls='--')
      ax_q2d.axvline(DM.q2h_gt,label='fit GT',color='C4',lw=3,ls='--')
      ax_q2d.axvline(DM.q2h_gx,label='fit GX',color='C5',lw=3,ls='--')
      #################################
      #################################
      ax_q2d_it.hist(DM.q2d_it ,np.linspace(0.35,1,20),histtype='step',density=True,label='std - it',lw=3)
      ax_q2d_it.hist(DM.q2dr_it,np.linspace(0.35,1,20),histtype='step',density=True,label='red - it',lw=3)
      ax_q2d_it.axvline(np.mean(DM.q2d_it ),color='C0',lw=3,ls='--')
      ax_q2d_it.axvline(np.mean(DM.q2dr_it),color='C1',lw=3,ls='--')
      ax_q2d_it.axvline(DM.q1h_2g,label='fit G', color='C3',lw=3)
      ax_q2d_it.axvline(DM.q1h_gt,label='fit GT',color='C4',lw=3)
      ax_q2d_it.axvline(DM.q1h_gx,label='fit GX',color='C5',lw=3)
      ax_q2d_it.axvline(DM.q2h_2g,label='fit G', color='C3',lw=3,ls='--')
      ax_q2d_it.axvline(DM.q2h_gt,label='fit GT',color='C4',lw=3,ls='--')
      ax_q2d_it.axvline(DM.q2h_gx,label='fit GX',color='C5',lw=3,ls='--')
      #################################
      #################################
      ax_q2d1.hist(SIDM.q2d, np.linspace(0.35,1,20),histtype='step',density=True,label='std',lw=3)
      ax_q2d1.hist(SIDM.q2dr,np.linspace(0.35,1,20),histtype='step',density=True,label='red',lw=3)
      ax_q2d1.axvline(np.mean(SIDM.q2d), color='C0',lw=3,ls='--')
      ax_q2d1.axvline(np.mean(SIDM.q2dr),color='C1',lw=3,ls='--')
      ax_q2d1.axvline(SIDM.q1h_2g,label='fit G', color='C3',lw=3)
      ax_q2d1.axvline(SIDM.q1h_gt,label='fit GT',color='C4',lw=3)
      ax_q2d1.axvline(SIDM.q1h_gx,label='fit GX',color='C5',lw=3)
      ax_q2d1.axvline(SIDM.q2h_2g,label='fit G', color='C3',lw=3,ls='--')
      ax_q2d1.axvline(SIDM.q2h_gt,label='fit GT',color='C4',lw=3,ls='--')
      ax_q2d1.axvline(SIDM.q2h_gx,label='fit GX',color='C5',lw=3,ls='--')
      ################################# 
      #################################
      ax_q2d1_it.hist(SIDM.q2d_it, np.linspace(0.35,1,20),histtype='step',density=True,label='std - it',lw=3)
      ax_q2d1_it.hist(SIDM.q2dr_it,np.linspace(0.35,1,20),histtype='step',density=True,label='red - it',lw=3)
      ax_q2d1_it.axvline(np.mean(SIDM.q2d_it), color='C0',lw=3,ls='--')
      ax_q2d1_it.axvline(np.mean(SIDM.q2dr_it),color='C1',lw=3,ls='--')
      ax_q2d1_it.axvline(SIDM.q1h_2g,label='fit G', color='C3',lw=3)
      ax_q2d1_it.axvline(SIDM.q1h_gt,label='fit GT',color='C4',lw=3)
      ax_q2d1_it.axvline(SIDM.q1h_gx,label='fit GX',color='C5',lw=3)
      ax_q2d1_it.axvline(SIDM.q2h_2g,label='fit G', color='C3',lw=3,ls='--')
      ax_q2d1_it.axvline(SIDM.q2h_gt,label='fit GT',color='C4',lw=3,ls='--')
      ax_q2d1_it.axvline(SIDM.q2h_gx,label='fit GX',color='C5',lw=3,ls='--')
      ################################# 
    else: 
      #################################
      ax_S.plot( SIDM.r , SIDM.S,'C3',  lw=4)
      ax_S.plot(   DM.r ,   DM.S,'k',   lw=4)
      ax_S.plot( SIDM.r,  SIDM.S,'C5-.',lw=4)
      ax_S.plot(   DM.r,    DM.S,'C7-.',lw=4)
      #################################
      #################################
      ax_S2.plot( SIDM.r , SIDM.S2,'C3',  lw=4)
      ax_S2.plot(   DM.r ,   DM.S2,'k',   lw=4)
      ax_S2.plot( SIDM.r,  SIDM.S2,'C5-.',lw=4)
      ax_S2.plot(   DM.r,    DM.S2,'C7-.',lw=4)
      #################################
      #################################
      ax_GX.plot( SIDM.r , SIDM.GX, 'C3', lw=4)
      ax_GX.plot(   DM.r ,   DM.GX,'k',   lw=4)
      #################################
      #################################
      ax_GT.plot( SIDM.r , SIDM.GT1h,'C3',lw=4)
      ax_GT.plot( SIDM.r , SIDM.GT1h_fit2,'C3--',alpha=0.5,lw=3)
      ax_GT.plot(   DM.r ,   DM.GT1h,'k' ,lw=4)
      ax_GT.plot(   DM.r ,   DM.GT1h_fit2,'k--',alpha=0.5,lw=3)
      #################################
      #################################
      ax_q2d.hist(DM.q2d ,np.linspace(0.35,1,20),histtype='step',density=True,lw=3)
      ax_q2d.hist(DM.q2dr,np.linspace(0.35,1,20),histtype='step',density=True,lw=3)
      ax_q2d.axvline(np.mean(DM.q2d ),color='C0',lw=3,ls='--')
      ax_q2d.axvline(np.mean(DM.q2dr),color='C1',lw=3,ls='--')
      ax_q2d.axvline(DM.q1h_2g,color='C3',lw=3)
      ax_q2d.axvline(DM.q1h_gt,color='C4',lw=3)
      ax_q2d.axvline(DM.q1h_gx,color='C5',lw=3)
      ax_q2d.axvline(DM.q2h_2g,color='C3',lw=3,ls='--')
      ax_q2d.axvline(DM.q2h_gt,color='C4',lw=3,ls='--')
      ax_q2d.axvline(DM.q2h_gx,color='C5',lw=3,ls='--')
      #################################
      #################################
      ax_q2d_it.hist(DM.q2d_it ,np.linspace(0.35,1,20),histtype='step',density=True,lw=3)
      ax_q2d_it.hist(DM.q2dr_it,np.linspace(0.35,1,20),histtype='step',density=True,lw=3)
      ax_q2d_it.axvline(np.mean(DM.q2d_it ),color='C0',lw=3,ls='--')
      ax_q2d_it.axvline(np.mean(DM.q2dr_it),color='C1',lw=3,ls='--')
      ax_q2d_it.axvline(DM.q1h_2g, color='C3',lw=3)
      ax_q2d_it.axvline(DM.q1h_gt,color='C4',lw=3)
      ax_q2d_it.axvline(DM.q1h_gx,color='C5',lw=3)
      ax_q2d_it.axvline(DM.q2h_2g, color='C3',lw=3,ls='--')
      ax_q2d_it.axvline(DM.q2h_gt,color='C4',lw=3,ls='--')
      ax_q2d_it.axvline(DM.q2h_gx,color='C5',lw=3,ls='--')
      #################################
      #################################
      ax_q2d1.hist(SIDM.q2d, np.linspace(0.35,1,20),histtype='step',density=True,lw=3)
      ax_q2d1.hist(SIDM.q2dr,np.linspace(0.35,1,20),histtype='step',density=True,lw=3)
      ax_q2d1.axvline(np.mean(SIDM.q2d), color='C0',lw=3,ls='--')
      ax_q2d1.axvline(np.mean(SIDM.q2dr),color='C1',lw=3,ls='--')
      ax_q2d1.axvline(SIDM.q1h_2g,color='C3',lw=3)
      ax_q2d1.axvline(SIDM.q1h_gt,color='C4',lw=3)
      ax_q2d1.axvline(SIDM.q1h_gx,color='C5',lw=3)
      ax_q2d1.axvline(SIDM.q2h_2g,color='C3',lw=3,ls='--')
      ax_q2d1.axvline(SIDM.q2h_gt,color='C4',lw=3,ls='--')
      ax_q2d1.axvline(SIDM.q2h_gx,color='C5',lw=3,ls='--')
      ################################# 
      #################################
      ax_q2d1_it.hist(SIDM.q2d_it, np.linspace(0.35,1,20),histtype='step',density=True,lw=3)
      ax_q2d1_it.hist(SIDM.q2dr_it,np.linspace(0.35,1,20),histtype='step',density=True,lw=3)
      ax_q2d1_it.axvline(np.mean(SIDM.q2d_it), color='C0',lw=3,ls='--')
      ax_q2d1_it.axvline(np.mean(SIDM.q2dr_it),color='C1',lw=3,ls='--')
      ax_q2d1_it.axvline(SIDM.q1h_2g,color='C3',lw=3)
      ax_q2d1_it.axvline(SIDM.q1h_gt,color='C4',lw=3)
      ax_q2d1_it.axvline(SIDM.q1h_gx,color='C5',lw=3)
      ax_q2d1_it.axvline(SIDM.q2h_2g,color='C3',lw=3,ls='--')
      ax_q2d1_it.axvline(SIDM.q2h_gt,color='C4',lw=3,ls='--')
      ax_q2d1_it.axvline(SIDM.q2h_gx,color='C5',lw=3,ls='--')
      ################################# 

    if col == 0:
      #################################
      ax_S.set_ylabel(r'$\Sigma [M_\odot/pc^2]$')
      #################################
      #################################
      ax_S2.set_ylabel(r'$\epsilon \times \Sigma_2 [M_\odot/pc^2]$')
      #################################
      #################################
      ax_GX.set_ylabel(r'$\epsilon \times \Gamma_X [M_\odot/pc^2]$')
      #################################
      #################################
      ax_GT.set_ylabel(r'$\epsilon \times \Gamma_T [M_\odot/pc^2]$')
      #################################
    else:
      #################################
      plt.setp(ax_S.get_yticklabels(), visible=False)
      #################################
      #################################
      plt.setp(ax_S2.get_yticklabels(), visible=False)
      #################################
      #################################
      plt.setp(ax_GX.get_yticklabels(), visible=False)
      #################################
      #################################
      plt.setp(ax_GT.get_yticklabels(), visible=False)
      #################################
      #################################
      plt.setp(ax_q2d.get_yticklabels(), visible=False)
      #################################
      #################################
      plt.setp(ax_q2d_it.get_yticklabels(), visible=False)
      #################################
      #################################
      plt.setp(ax_q2d1.get_yticklabels(), visible=False)
      #################################
      #################################
      plt.setp(ax_q2d1_it.get_yticklabels(), visible=False)
      #################################

    #################################
    ax_S.set_xscale('log')
    ax_S.set_yscale('log')
    ax_S.set_xlabel('$R [Mpc]$')
    ax_S.axis([0.09,ROUT*1.e-3,1,250])
    ax_S.legend(frameon=False,title=name_folder.replace("_"," ").title())
    #################################

    #################################
    ax_S2.set_xscale('log')
    ax_S2.set_yscale('log')
    ax_S2.set_xlabel('$R [Mpc]$')
    ax_S2.axis([0.09,ROUT*1.e-3,0.5,50])
    ax_S2.legend(frameon=False,title=name_folder.replace("_"," ").title())
    #################################

    #################################
    ax_GX.set_xscale('log')
    ax_GX.set_xlabel('$R [Mpc]$')
    ax_GX.axis([0.09,ROUT*1.e-3,-5,10])
    ax_GX.legend(frameon=False,title=name_folder.replace("_"," ").title())
    #################################

    #################################
    ax_GT.set_xscale('log')
    ax_GT.set_yscale('log')
    ax_GT.set_xlabel('$R [Mpc]$')
    ax_GT.axis([0.09,ROUT*1.e-3,5e-2,50])
    ax_GT.legend(frameon=False,title=name_folder.replace("_"," ").title())
    #################################

    #################################
    ax_q2d.legend(loc=2,frameon=False,title=name_folder.replace("_"," ").title())
    ax_q2d.axis([0.25,1.1,0.0,8])
    ax_q2d.set_xlabel('q2d')
    #################################
    
    #################################
    ax_q2d_it.legend(loc=2,frameon=False,title=name_folder.replace("_"," ").title())
    ax_q2d_it.axis([0.25,1.1,0.0,8])
    ax_q2d_it.set_xlabel('q2d')
    #################################

    #################################
    ax_q2d1.legend(loc=2,frameon=False,title=name_folder.replace("_"," ").title())
    ax_q2d1.axis([0.25,1.1,0.0,8])
    ax_q2d1.set_xlabel('q2d')
    #################################
    
    #################################
    ax_q2d1_it.legend(loc=2,frameon=False,title=name_folder.replace("_"," ").title())
    ax_q2d1_it.axis([0.25,1.1,0.0,8])
    ax_q2d1_it.set_xlabel('q2d')
    #################################

  fig_S.savefig(output_folder  + 'profile_S_' +name_tensor+'.png',bbox_inches='tight')
  fig_S2.savefig(output_folder + 'profile_S2_'+name_tensor+'.png',bbox_inches='tight')
  fig_GX.savefig(output_folder + 'profile_GX_'+name_tensor+'.png',bbox_inches='tight')
  fig_GT.savefig(output_folder + 'profile_GT_'+name_tensor+'.png',bbox_inches='tight')
  fig_q2d.savefig(output_folder + 'profile_q2d_cdm.png',bbox_inches='tight')
  fig_q2d_it.savefig(output_folder + 'profile_q2d_cdm_it.png',bbox_inches='tight')
  fig_q2d1.savefig(output_folder + 'profile_q2d_sidm.png',bbox_inches='tight')
  fig_q2d1_it.savefig(output_folder + 'profile_q2d_sidm_it.png',bbox_inches='tight')

  #plt.show()

tf = time.time()
print("Total Time %.2f\n" % (tf-ti))
