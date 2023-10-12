import time
import numpy  as np
import pandas as pd
import pickle

class pack_core():

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
      self.DS1hc_fit        = np.load(f)
      self.DS2hc_fit        = np.load(f)
      self.DSc_fit          = np.load(f)
      self.lM200c_ds        = np.load(f)
      self.c200c_ds         = np.load(f)
      self.bm1_ds           = np.load(f)
      self.e_c200_ds        = np.load(f)
      self.e_lM200_ds       = np.load(f)
      self.e_bm1_ds         = np.load(f)
      self.mcmc_ds_lMc      = np.load(f)
      self.mcmc_ds_c200c    = np.load(f)
      self.mcmc_ds_bm1      = np.load(f)
      self.q1hc_2g          = np.load(f)
      self.q2hc_2g          = np.load(f)
      self.mcmc_q1hc_2g     = np.load(f)
      self.mcmc_q2hc_2g     = np.load(f)
      self.GT1hc_fit2       = np.load(f)
      self.GX1hc_fit2       = np.load(f)
      self.GT2hc_fit2       = np.load(f)
      self.GX2hc_fit2       = np.load(f)
      #self.q1h_gt          = np.load(f)
      #self.q2h_gt          = np.load(f)
      #self.mcmc_q1h_gt     = np.load(f)
      #self.mcmc_q2h_gt     = np.load(f)
      #self.GT1h            = np.load(f)
      #self.GT2h            = np.load(f)
      #self.q1h_gx          = np.load(f)
      #self.q2h_gx          = np.load(f)
      #self.mcmc_q1h_gx     = np.load(f)
      #self.mcmc_q2h_gx     = np.load(f)
      #self.GX1h            = np.load(f)
      #self.GX2h            = np.load(f)
      #self.a_2g            = np.load(f)
      #self.b_2g            = np.load(f)
      #self.q2hr_2g         = np.load(f)
      #self.mcmc_a_2g       = np.load(f)
      #self.mcmc_b_2g       = np.load(f)
      #self.mcmc_q2hr_2g    = np.load(f)
      #self.GT1hr_fit2      = np.load(f)
      #self.GX1hr_fit2      = np.load(f)
      #self.GT2hr_fit2      = np.load(f)
      #self.GX2hr_fit2      = np.load(f)
      #self.a_gx            = np.load(f)
      #self.b_gx            = np.load(f)
      #self.q2hr_gx         = np.load(f)
      #self.mcmc_a_gx       = np.load(f)
      #self.mcmc_b_gx       = np.load(f)
      #self.mcmc_q2hr_gx    = np.load(f)
      #self.GX1hr           = np.load(f)
      #self.GX2hr           = np.load(f)
      #self.a_gt            = np.load(f)
      #self.b_gt            = np.load(f)
      #self.q2hr_gt         = np.load(f)
      #self.mcmc_a_gt       = np.load(f)
      #self.mcmc_b_gt       = np.load(f)
      #self.mcmc_q2hr_gt    = np.load(f)
      #self.GT1hr           = np.load(f)
      #self.GT2hr           = np.load(f)
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
      self.fi              = np.load(f)
      self.err_fi          = np.load(f)
      self.qs_all          = np.load(f)
      self.err_qs_all      = np.load(f)
      self.fi_all          = np.load(f)
      self.err_fi_all      = np.load(f)
      self.a_2g_fb         = np.load(f)
      self.b_2g_fb         = np.load(f)
      self.q2hr_2g_fb      = np.load(f)
      self.mcmc_a_2g_fb    = np.load(f)
      self.mcmc_b_2g_fb    = np.load(f)
      self.mcmc_q2hr_2g_fb = np.load(f)
      self.main            = pickle.loads(np.load(f, allow_pickle=True).item())
      f.close()

      # guardo estos atributos extras para graficarlos
      self.q2d          = np.concatenate((self.main.b2D_xy/self.main.a2D_xy,self.main.b2D_zx/self.main.a2D_zx,self.main.b2D_yz/self.main.a2D_yz))
      self.q2dr         = np.concatenate((self.main.b2Dr_xy/self.main.a2Dr_xy,self.main.b2Dr_zx/self.main.a2Dr_zx,self.main.b2Dr_yz/self.main.a2Dr_yz))
      self.q2d_it       = np.concatenate((self.main.b2D_it_xy/self.main.a2D_it_xy,self.main.b2D_it_zx/self.main.a2D_it_zx,self.main.b2D_it_yz/self.main.a2D_it_yz))
      self.q2dr_it      = np.concatenate((self.main.b2Dr_it_xy/self.main.a2Dr_it_xy,self.main.b2Dr_it_zx/self.main.a2Dr_it_zx,self.main.b2Dr_it_yz/self.main.a2Dr_it_yz))

# main_file  = '/mnt/projects/lensing/SIDM_project/halo_props/projections/full_extend_prop_env_halo_propsv2_rock2_match_cdm_z0_main.csv.bz2'
# main_file1 = '/mnt/projects/lensing/SIDM_project/halo_props/projections/full_extend_prop_env_halo_propsv2_rock2_match_sidm1_z0_main.csv.bz2'
# main  = pd.read_csv(main_file)
# main1 = pd.read_csv(main_file1)

folders_list = ["relajados"]
typetensor = {'reduced': True}

for jdx, name_folder in enumerate(folders_list):

  for idx, fname in enumerate(typetensor.keys()):

    DM   = pack_core("./arreglos_core/%s_DM_%s.npy" % (name_folder, fname))
    SIDM = pack_core("./arreglos_core/%s_SIDM_%s.npy" % (name_folder, fname))

    main  = DM.main
    main1 = SIDM.main


