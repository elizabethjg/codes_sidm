import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import pylab
from models_profiles import *
import matplotlib.pyplot as plt
from make_grid import Grilla
from lenspack.image.inversion import ks93inv
params = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.045, 'sigma8': 0.811, 'ns': 0.96}

mp = 1.
nhalos = 1.
z = 0.0
M200 = 1.e14
c200 = 4.
q = 0.6
e = (1. - q)/(1. + q)

class profile_from_map:

    def __init__(self,nhalos=1,RIN=100.,ROUT=1500.,ndots=20,resolution=1000):

        # MAKE KAPPA MAP
        xedges = np.linspace(-8,8,resolution)
        xedges = xedges[abs(xedges)>0.001]
        lsize  = np.diff(xedges)[0]
        xb, yb = np.meshgrid(xedges[:-1],xedges[:-1])+(lsize/2.)

        r = np.sqrt(xb**2+yb**2)
        theta  = np.arctan2(yb,xb)
        
        R  = np.sqrt((r**2)*(q*(np.cos(theta))**2 + (1./q)*(np.sin(theta))**2))
                
        Sellip = Sigma_NFW_2h(R,z,M200,c200,cosmo_params=params)
        
        kE = Sellip
        kB = np.zeros(kE.shape)

        e1, e2   = ks93inv(kE, kB)

        xb = xb.flatten()
        yb = yb.flatten()

        r = np.sqrt(xb**2+yb**2)
        theta  = np.arctan2(yb,xb)
        
        
        #get tangential ellipticities 
        et = (-e1.flatten()*np.cos(2*theta)-e2.flatten()*np.sin(2*theta))
        #get cross ellipticities
        ex = (-e1.flatten()*np.sin(2*theta)+e2.flatten()*np.cos(2*theta))
        # get kappa
        kE  = kE.flatten()

        bines = np.round(np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1),0)
        dig = np.digitize(r*1.e3,bines)
        R = (bines[:-1] + np.diff(bines)*0.5)*1.e-3
        
        SIGMA   = []
        DSIGMA_T = []
        DSIGMA_X = []
        SIGMAcos = []
        GAMMATcos = []
        GAMMAXsin = []
                                
                            
        for nbin in range(ndots):
            mbin = dig == nbin+1              
            
            SIGMA     = np.append(SIGMA,np.mean(kE[mbin]))
            DSIGMA_T  = np.append(DSIGMA_T,np.mean(et[mbin]))
            DSIGMA_X  = np.append(DSIGMA_X,np.mean(ex[mbin]))
            
            SIGMAcos  = np.append(SIGMAcos,np.sum(kE[mbin]*np.cos(2.*theta[mbin]))/np.sum(np.cos(2.*theta[mbin])**2))
            
            GAMMATcos = np.append(GAMMATcos,np.sum(et[mbin]*np.cos(2.*theta[mbin]))/np.sum(np.cos(2.*theta[mbin])**2))
            GAMMAXsin = np.append(GAMMAXsin,np.sum(ex[mbin]*np.sin(2.*theta[mbin]))/np.sum(np.sin(2.*theta[mbin])**2))

        GT_func,GX_func = GAMMA_components(R,z,ellip=e,M200 = M200,c200=c200,cosmo_params=params)        

        
        self.Sm    = Sigma_NFW_2h(R,z,M200,c200,cosmo_params=params)
        self.DSm    = Delta_Sigma_NFW_2h(R,z,M200,c200,cosmo_params=params)
        self.GTm   = GT_func
        self.GXm   = GX_func
        self.r     = R
        self.S     = SIGMA    
        self.DS_T  = DSIGMA_T 
        self.DS_X  = DSIGMA_X 
        self.S2    = SIGMAcos 
        self.GT    = GAMMATcos
        self.GX    = -1*GAMMAXsin

p = profile_from_map()

plt.figure()
plt.plot(p.r,p.DSm,'C3')
plt.plot(p.r,p.DS_T,'C0')
plt.ylabel('$\Delta \Sigma$')
plt.xlabel('R')

plt.figure()
plt.plot(p.r,p.Sm,'C3')
plt.plot(p.r,p.S,'C0')
plt.ylabel('$\Sigma$')
plt.xlabel('R')

plt.figure()
plt.plot(p.r,p.GTm,'C3')
plt.plot(p.r,p.GT,'C0')
plt.ylabel('$\Gamma_T$')
plt.xlabel('R')

plt.figure()
plt.plot(p.r,p.GXm,'C3')
plt.plot(p.r,p.GX,'C0')
plt.ylabel('$\Gamma_X$')
plt.xlabel('R')

g = Grilla(rangex = [-1.5,1.5],nbins=60)

# q = 0.5
# e = (1. - q)/(1. + q)
# qr = 0.4*g.r**-0.2
# er = (1. - qr)/(1. + qr)

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
