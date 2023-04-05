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
      
      self.q2d          = np.concatenate((self.main.b2D_xy/self.main.a2D_xy,self.main.b2D_zx/self.main.a2D_zx,self.main.b2D_yz/self.main.a2D_yz))
      self.q2dr         = np.concatenate((self.main.b2Dr_xy/self.main.a2Dr_xy,self.main.b2Dr_zx/self.main.a2Dr_zx,self.main.b2Dr_yz/self.main.a2Dr_yz))
      self.q2d_it       = np.concatenate((self.main.b2D_it_xy/self.main.a2D_it_xy,self.main.b2D_it_zx/self.main.a2D_it_zx,self.main.b2D_it_yz/self.main.a2D_it_yz))
      self.q2dr_it      = np.concatenate((self.main.b2Dr_it_xy/self.main.a2Dr_it_xy,self.main.b2Dr_it_zx/self.main.a2Dr_it_zx,self.main.b2Dr_it_yz/self.main.a2Dr_it_yz))

      f.close()

# Globales
typetensor = ['standard', 'reduced']
folders_list = ["total", "relajados", \
                "relajados_cerca", "relajados_lejos", \
                "relajados_masivos", "relajados_no_masivos"]
input_folder  = "./arreglos/"      

ti = time.time()

for idx, name_tensor in enumerate(typetensor):

  for jdx, name_folder in enumerate(folders_list):
      
    filename_DM = input_folder + "%s_DM_%s.npy" % (name_folder, name_tensor)
    DM = pack(filename_DM)

    filename_SIDM = input_folder + "%s_SIDM_%s.npy" % (name_folder, name_tensor)
    SIDM = pack(filename_SIDM)

tf = time.time()
print("Total Time %.2f\n" % (tf-ti))