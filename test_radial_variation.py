import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import pylab
from models_profiles import *
import matplotlib.pyplot as plt
from make_grid import Grilla

z = 0.2
M200 = 1.e14
c200 = 4.

g = Grilla(rangex = [-1.5,1.5],nbins=60)

q = 0.5
e = (1. - q)/(1. + q)
qr = 0.4*g.r**-0.2
er = (1. - qr)/(1. + qr)

q = 0.8
e = (1. - q)/(1. + q)
qr = 0.8*g.r**-0.04
er = (1. - qr)/(1. + qr)

Sround = Sigma_NFW_2h(g.r,z,M200,c200)
S2 = S2_quadrupole(g.r,z,M200,c200 = c200)

R  = np.sqrt((g.r**2)*(q*(np.cos(g.t))**2 + (1./q)*(np.sin(g.t))**2))
Rr = np.sqrt((g.r**2)*(qr*(np.cos(g.t))**2 + (1./qr)*(np.sin(g.t))**2))

Sellip = Sigma_NFW_2h(R,z,M200,c200)
Sellip_model = Sround + e*S2*np.cos(2*g.t)

Sellipr = Sigma_NFW_2h(Rr,z,M200,c200)
Sellipr_model = Sround + er*S2*np.cos(2*g.t)

plt.figure()
plt.plot(g.r,qr)
plt.xlabel('r')
plt.ylabel('q')

plt.figure(figsize=(4,4))
plt.title('Sround')
plt.scatter(g.x,g.y,c=np.log10(Sround))

plt.figure(figsize=(4,4))
plt.title('Sellip')
plt.scatter(g.x,g.y,c=np.log10(Sellip))

plt.figure(figsize=(4,4))
plt.title('Sellip_model')
plt.scatter(g.x,g.y,c=np.log10(Sellip_model))

plt.figure(figsize=(4,5))
plt.title('(Sellip - Sellip_model)/Sellip')
plt.scatter(g.x,g.y,c=(Sellip-Sellip_model)/Sellip)
plt.colorbar()


plt.figure(figsize=(4,4))
plt.title('Sellipr')
plt.scatter(g.x,g.y,c=np.log10(Sellipr))

plt.figure(figsize=(4,4))
plt.title('Sellipr_model')
plt.scatter(g.x,g.y,c=np.log10(Sellipr_model))

plt.figure(figsize=(4,5))
plt.title('(Sellipr - Sellipr_model)/Sellipr')
plt.scatter(g.x,g.y,c=(Sellipr-Sellipr_model)/Sellipr)
plt.colorbar()
