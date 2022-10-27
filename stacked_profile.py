import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import matplotlib.pyplot as plt
from stacked import stack_halos
from stacked import stack_profile
import pandas as pd
from fit_models_colossus import *
from models_profiles import *
from colossus.halo import concentration
from colossus.cosmology import cosmology  
from colossus.halo import mass_defs
params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
cosmology.addCosmology('MICE', params)
cosmo = cosmology.setCosmology('MICE')

zs = ['z0','z51','z96']


z = zs[0]

def mask_border(x0,y0,z0):
    mask = (x0 > 2.)*(x0 < 118.)*(y0 > 2.)*(y0 < 118.)*(z0 > 2.)*(z0 < 118.)
    return mask

rock       = pd.read_csv('../halo_props/halo_props_cdm_'+z+'_rock2.csv.bz2')
rock1      = pd.read_csv('../halo_props/halo_props_sidm1_'+z+'_rock2.csv.bz2')
mrock      = mask_border(rock.x0,rock.y0,rock.z0)
mrock1     = mask_border(rock1.x0,rock1.y0,rock1.z0)

path1 = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/rockstar/SIDM1/'
main_file1 = '/home/elizabeth/SIDM/halo_props/halo_props_rock2_sidm1_'+z+'_main.csv.bz2'
path = '/mnt/projects/lensing/SIDM_project/Lentes/Eli_Agus/snapshot_050/rockstar/CDM/'
main_file = '/home/elizabeth/SIDM/halo_props/halo_props_rock2_cdm_'+z+'_main.csv.bz2'

main  = pd.read_csv(main_file)
main1 = pd.read_csv(main_file1)

rc  = np.array(np.sqrt((main.xc - main.xc_rc)**2 + (main.yc - main.yc_rc)**2 + (main.zc - main.zc_rc)**2))
offset  = rc/main.r_max

rc1      = np.array(np.sqrt((main1.xc - main1.xc_rc)**2 + (main1.yc - main1.yc_rc)**2 + (main1.zc - main1.zc_rc)**2))
offset1  = rc1/main1.r_max


mhalos  = offset  < 0.1
mhalos1 = offset1 < 0.1

haloids  = np.array(main.column_halo_id)[mrock]
haloids1 = np.array(main1.column_halo_id)[mrock1]
    
x,y,z,x2d,y2d      = stack_halos(main_file,path,haloids,True)   
x1,y1,z1,x2d1,y2d1 = stack_halos(main_file1,path1,haloids1,True)   
    
m3d = (abs(x) < 4.)*(abs(y) < 4.)*(abs(z) < 4.)
m2d = (abs(x2d) < 4.)*(abs(y2d) < 4.)

m3d1 = (abs(x1) < 4.)*(abs(y1) < 4.)*(abs(z1) < 4.)
m2d1 = (abs(x2d1) < 4.)*(abs(y2d1) < 4.)

X,Y,Z = x[m3d]*1.e3,y[m3d]*1.e3,z[m3d]*1.e3
Xp,Yp = x2d[m2d]*1.e3,y2d[m2d]*1.e3

X1,Y1,Z1 = x1[m3d1]*1.e3,y1[m3d1]*1.e3,z1[m3d1]*1.e3
Xp1,Yp1  = x2d1[m2d1]*1.e3,y2d1[m2d1]*1.e3

theta  = np.arctan(main.a2Dy/main.a2Dx)
theta1 = np.arctan(main1.a2Dy/main1.a2Dx)

r,rho,S,DS,S_2    = stack_profile(X,Y,Z,Xp,Yp,100,0.,mhalos.sum())
r1,rho1,S1,DS1,S1_2 = stack_profile(X1,Y1,Z1,Xp1,Yp1,100,0.,mhalos.sum())

   
z = 0.


mr = r > 0.

# cvir = concentration.concentration(np.mean(rock.Mvir[mrock]), 'vir', z, model = 'diemer19')
# M200c,R,c200c = mass_defs.changeMassDefinition(np.mean(rock.Mvir[mrock]), cvir, z, 'vir', '200c')

M200c = 10**13.5
c200c = concentration.concentration(np.mean(rock.Mvir[mrock]), 'vir', z, model = 'diemer19')

s3d     = rho_NFW_2h(r[mr],z,M200 = M200c,c200=c200c,terms='1h')*(1.e6**3)
ds      = Delta_Sigma_NFW_2h(r[mr],z,M200 = M200c,c200=c200c,terms='1h')*(1.e6**2)
s       = Sigma_NFW_2h(r[mr],z,M200 = M200c,c200=c200c,terms='1h')*(1.e6**2)
s2      = S2_quadrupole(r[mr],z,M200 = M200c,c200=c200c,cosmo_params=params,terms='1h',pname='NFW')*(1.e6**2)

# rhof    = rho_fit(r[mr],rho[mr],1./r[mr]**3,z)
# Sf      = Sigma_fit(r[mr],S[mr]/mhalos.sum(),1./r[mr]**2,z)
# rhof_E    = rho_fit(r[mr],rho[mr]/mhalos.sum(),1./r[mr]**3,z,'Einasto',rhof.M200,rhof.c200)
# Sf_E      = Sigma_fit(r[mr],S[mr]/mhalos.sum(),1./r[mr]**2,z,'Einasto',rhof.M200,rhof.c200)

