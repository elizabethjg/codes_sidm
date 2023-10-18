import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import time
import numpy  as np
import pandas as pd
import pickle
import corner
from stacked import quadrupoles_from_map_model
from models_profiles import *

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

    DMc   = pack_core("./arreglos_core/%s_DM_%s.npy" % (name_folder, fname))
    SIDMc = pack_core("./arreglos_core/%s_SIDM_%s.npy" % (name_folder, fname))



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


# La carpeta donde estan guarados los objetos
input_folder  = "./arreglos/"      
name_tensor = 'reduced'
name_folder = 'relajados'
sname = 'Relaxed'

filename_SIDM = input_folder + "%s_SIDM_%s.npy" % (name_folder, name_tensor)
filename_DM = input_folder + "%s_DM_%s.npy" % (name_folder, name_tensor)

DM   = pack(filename_DM)
SIDM = pack(filename_SIDM)

######### MCMC plot

mcmc_DM = np.array([DMc.mcmc_ds_lMc[:1500],DMc.mcmc_ds_c200c[:1500],DMc.mcmc_ds_bm1[:1500]]).T
mcmc_SIDM = np.array([SIDMc.mcmc_ds_lMc[:1500],SIDMc.mcmc_ds_c200c[:1500],SIDMc.mcmc_ds_bm1[:1500]]).T


f1 = corner.corner(mcmc_DM,labels=[r'$\log M_{200}$',r'$c_{200}$','$1/b$'],
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
f1.savefig('../final_plots/corner_core.pdf',bbox_inches='tight')

######### GAMMA components
fig, ax = plt.subplots(2,2, figsize=(8,8),sharex = True)    

fig.subplots_adjust(hspace=0,wspace=0)

ax[0,0].fill_between(DM.r,DM.GT+DM.e_GT,DM.GT-DM.e_GT,color='C7',alpha=0.4)
ax[0,0].plot(DM.r,DM.GT1h_fit2+DM.GT2h_fit2,'C4',label=r'NFW',lw=2)
ax[0,0].plot(DM.r,DMc.GT1hc_fit2+DMc.GT2hc_fit2,'C2',label=r'NFW with core',lw=2)

ax[0,1].fill_between(SIDM.r,SIDM.GT+SIDM.e_GT,SIDM.GT-SIDM.e_GT,color='C7',alpha=0.4)
ax[0,1].plot(SIDM.r,SIDM.GT1h_fit2+SIDM.GT2h_fit2,'C4',label=r'$q_0 = 0.6$, $\alpha = -0.1$',lw=2)
ax[0,1].plot(SIDM.r,SIDMc.GT1hc_fit2+SIDMc.GT2hc_fit2,'C2',label=r'$q_0 = 0.6$, $\alpha = -0.1$',lw=2)

ax[1,0].fill_between(DM.r,DM.GX+DM.e_GX,DM.GX-DM.e_GX,color='C7',alpha=0.4)
ax[1,0].plot(DM.r,DM.GX1h_fit2+DM.GX2h_fit2,'C4',label=r'$q_0 = 0.6$, $\alpha = -0.1$',lw=2)
ax[1,0].plot(DM.r,DMc.GX1hc_fit2+DMc.GX2hc_fit2,'C2',label=r'$q_0 = 0.6$, $\alpha = -0.1$',lw=2)
   
ax[1,1].fill_between(SIDM.r,SIDM.GX+SIDM.e_GX,SIDM.GX-SIDM.e_GX,color='C7',alpha=0.4)
ax[1,1].plot(SIDM.r,SIDM.GX1h_fit2+SIDM.GX2h_fit2,'C4',label=r'$q_0 = 0.6$, $\alpha = -0.1$',lw=2)
ax[1,1].plot(SIDM.r,SIDMc.GX1hc_fit2+SIDMc.GX2hc_fit2,'C2',label=r'$q_0 = 0.6$, $\alpha = -0.1$',lw=2)


ax[0,1].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,1].set_yscale('log')

ax[0,0].legend(frameon=False)

ax[1,0].set_xlabel('r [$h^{-1}$ Mpc]')
ax[1,1].set_xlabel('r [$h^{-1}$ Mpc]')
ax[1,0].set_ylabel(r'$\Gamma_\times [h M_\odot/pc^2]$')
ax[0,0].set_ylabel(r'$\Gamma_T [h M_\odot/pc^2]$')
ax[1,0].xaxis.set_ticks([0.1,1,3])


fig.savefig('../final_plots/comparison_core.pdf',bbox_inches='tight')


######### q1h, q2h, PD plots

plt.figure()
plt.hist(DMc.mcmc_q1hc_2g[:3000],100,histtype='step')
plt.hist(DM.mcmc_q1h_2g[:3000],100,histtype='step')
plt.hist(DMc.mcmc_q2hc_2g[:3000],100,histtype='step')
plt.hist(DM.mcmc_q2h_2g[:3000],100,histtype='step')
print('DM q1h',np.median(DMc.mcmc_q1hc_2g[:3000])/np.median(DM.mcmc_q1h_2g[:3000]))
print('DM q2h',np.median(DMc.mcmc_q2hc_2g[:3000])/np.median(DM.mcmc_q2h_2g[:3000]))

plt.figure()
plt.hist(SIDMc.mcmc_q1hc_2g[:3000],100,histtype='step')
plt.hist(SIDM.mcmc_q1h_2g[:3000],100,histtype='step')
plt.hist(SIDMc.mcmc_q2hc_2g[:3000],100,histtype='step')
plt.hist(SIDM.mcmc_q2h_2g[:3000],100,histtype='step')
print('SIDM q1h',np.median(SIDMc.mcmc_q1hc_2g[:3000])/np.median(SIDM.mcmc_q1h_2g[:3000]))
print('SIDM q2h',np.median(SIDMc.mcmc_q2hc_2g[:3000])/np.median(SIDM.mcmc_q2h_2g[:3000]))

######### GAMMA with radial variation

Gtermsc_dm = quadrupoles_from_map_model(M200=10**DMc.lM200c_ds,c200=DMc.c200c_ds,
                                          resolution=500,
                                          RIN=100.,ROUT=5000.,
                                          ndots=20,pname='NFW-core',b=1./DMc.bm1_ds)

G1hc_dm = Gtermsc_dm(DM.a_2g_fb,DM.b_2g_fb)

Gtermsc_sidm = quadrupoles_from_map_model(M200=10**SIDMc.lM200c_ds,c200=SIDMc.c200c_ds,
                                          resolution=500,
                                          RIN=100.,ROUT=5000.,
                                          ndots=20,pname='NFW-core',b=1./SIDMc.bm1_ds)

G1hc_sidm = Gtermsc_sidm(SIDM.a_2g_fb,SIDM.b_2g_fb)

Gterms_dm = quadrupoles_from_map_model(M200=10**DM.lM200_ds,c200=DM.c200_ds,
                                          resolution=500,
                                          RIN=100.,ROUT=5000.,
                                          ndots=20)

G1h_dm = Gterms_dm(DM.a_2g_fb,DM.b_2g_fb)

Gterms_sidm = quadrupoles_from_map_model(M200=10**SIDM.lM200_ds,c200=SIDM.c200_ds,
                                          resolution=500,
                                          RIN=100.,ROUT=5000.,
                                          ndots=20)

G1h_sidm = Gterms_sidm(SIDM.a_2g_fb,SIDM.b_2g_fb)


fig, ax = plt.subplots(2,2, figsize=(8,8),sharex = True)    

e2h_sidm   = (1.-SIDM.q2hr_2g_fb)/(1.+SIDM.q2hr_2g_fb)
e2h_dm   = (1.-SIDM.q2hr_2g_fb)/(1.+SIDM.q2hr_2g_fb)

fig.subplots_adjust(hspace=0,wspace=0)

ax[0,0].fill_between(DM.r,DM.GT+DM.e_GT,DM.GT-DM.e_GT,color='C7',alpha=0.4)
ax[0,0].plot(DM.r,G1h_dm['GT'] + e2h_dm*Gterms_dm.GT_2h,'C4',label=r'NFW',lw=2)
ax[0,0].plot(DM.r,G1hc_dm['GT'] + e2h_dm*Gtermsc_dm.GT_2h,'C2',label=r'NFW with core',lw=2)

ax[0,1].fill_between(SIDM.r,SIDM.GT+SIDM.e_GT,SIDM.GT-SIDM.e_GT,color='C7',alpha=0.4)
ax[0,1].plot(DM.r,G1h_sidm['GT'] + e2h_sidm*Gterms_sidm.GT_2h,'C4',label=r'NFW',lw=2)
ax[0,1].plot(DM.r,G1hc_sidm['GT'] + e2h_sidm*Gtermsc_sidm.GT_2h,'C2',label=r'NFW with core',lw=2)

ax[1,0].fill_between(DM.r,DM.GX+DM.e_GX,DM.GX-DM.e_GX,color='C7',alpha=0.4)
ax[1,0].plot(DM.r,G1h_dm['GX'] + e2h_dm*GXerms_dm.GX_2h,'C4',label=r'NFW',lw=2)
ax[1,0].plot(DM.r,G1hc_dm['GX'] + e2h_dm*GXermsc_dm.GX_2h,'C2',label=r'NFW with core',lw=2)
   
ax[1,1].fill_between(SIDM.r,SIDM.GX+SIDM.e_GX,SIDM.GX-SIDM.e_GX,color='C7',alpha=0.4)
ax[1,1].plot(DM.r,G1h_sidm['GX'] + e2h_sidm*GXerms_sidm.GX_2h,'C4',label=r'NFW',lw=2)
ax[1,1].plot(DM.r,G1hc_sidm['GX'] + e2h_sidm*GXermsc_sidm.GX_2h,'C2',label=r'NFW with core',lw=2)


ax[0,1].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,1].set_yscale('log')

ax[0,0].legend(frameon=False)

ax[1,0].set_xlabel('r [$h^{-1}$ Mpc]')
ax[1,1].set_xlabel('r [$h^{-1}$ Mpc]')
ax[1,0].set_ylabel(r'$\Gamma_\times [h M_\odot/pc^2]$')
ax[0,0].set_ylabel(r'$\Gamma_T [h M_\odot/pc^2]$')
ax[1,0].xaxis.set_ticks([0.1,1,3])


fig.savefig('../final_plots/comparison_core_radial.pdf',bbox_inches='tight')
