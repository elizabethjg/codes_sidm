import numpy as np
import pandas as pd

rock     = pd.read_csv('../halo_props/halo_props_cdm_z0_rock.csv.bz2')
main     = pd.read_csv('../halo_props/halo_props_cdm_z0_main.csv.bz2')

rock1    = pd.read_csv('../halo_props/halo_props_sidm1_z0_rock.csv.bz2')
main1    = pd.read_csv('../halo_props/halo_props_sidm1_z0_main.csv.bz2')

# LOAD PARAMS

S_rock = rock['c to a']
Q_rock = rock['b to a']
S_rock1 = rock1['c to a']
Q_rock1 = rock1['b to a']

S = main.c3D/main.a3D
Q = main.c3D/main.a3D
S1 = main1.c3D/main1.a3D
Q1 = main1.c3D/main1.a3D

Eratio  = (2.*main.EKin/abs(main.EPot))
Eratio1 = (2.*main1.EKin/abs(main1.EPot))

lgM = main.lgM
lgM1 = main1.lgM

rc = np.array(np.sqrt((main.xc - main.xc_rc)**2 + (main.yc - main.yc_rc)**2 + (main.zc - main.zc_rc)**2))
rc1 = np.array(np.sqrt((main1.xc - main1.xc_rc)**2 + (main1.yc - main1.yc_rc)**2 + (main1.zc - main1.zc_rc)**2))
offset = rc/main.r_max
offset1 = rc1/main1.r_max



# COMPARISON rock vs new params
plt.figure()
plt.plot(S_rock,S,'.',label='c/a')
plt.plot(Q_rock,Q,'.',label='b/a')
plt.plot(S_rock1,S1,'x',label='c/a - SIDM')
plt.plot(Q_rock1,Q1,'x',label='b/a - SIDM')
plt.plot([0,1],[0,1],'C7--')
plt.xlabel('ROCKSTAR')
plt.ylabel('new params')
plt.legend()
