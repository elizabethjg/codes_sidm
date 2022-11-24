import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import pandas as pd
import h5py
from member_distribution import projected_coodinates
from lenspack.image.inversion import ks93inv

def stack_halos(main_file,path,haloids,reduced = False):

    main = pd.read_csv(main_file)
    
    x = np.array([])
    y = np.array([])
    z = np.array([])
    
    x2d = np.array([])
    y2d = np.array([])    
    
    for j in haloids:
        
        halo = h5py.File(path+'halo_'+str(j)+'.hdf5','r')       
        
        X = np.array(halo['X']) - main.xc_rc[j]/1.e3
        Y = np.array(halo['Y']) - main.yc_rc[j]/1.e3
        Z = np.array(halo['Z']) - main.zc_rc[j]/1.e3
        
        if reduced:
            xrot = (main.a3Drx[j]*X)+(main.a3Dry[j]*Y)+(main.a3Drz[j]*Z);
            yrot = (main.b3Drx[j]*X)+(main.b3Dry[j]*Y)+(main.b3Drz[j]*Z);
            zrot = (main.c3Drx[j]*X)+(main.c3Dry[j]*Y)+(main.c3Drz[j]*Z);
        else:
            xrot = (main.a3Dx[j]*X)+(main.a3Dy[j]*Y)+(main.a3Dz[j]*Z);
            yrot = (main.b3Dx[j]*X)+(main.b3Dy[j]*Y)+(main.b3Dz[j]*Z);
            zrot = (main.c3Dx[j]*X)+(main.c3Dy[j]*Y)+(main.c3Dz[j]*Z);
        
        x = np.append(x,xrot)
        y = np.append(y,yrot)
        z = np.append(z,zrot)
        
        X2d,Y2d = projected_coodinates(X,Y,Z,main.xc_rc[j],main.yc_rc[j],main.zc_rc[j])

        if reduced:
            x2drot = (main.a2Drx[j]*X2d)+(main.a2Dry[j]*Y2d)
            y2drot = (main.b2Drx[j]*X2d)+(main.b2Dry[j]*Y2d)
        else:
            x2drot = (main.a2Dx[j]*X2d)+(main.a2Dy[j]*Y2d)
            y2drot = (main.b2Dx[j]*X2d)+(main.b2Dy[j]*Y2d)

        x2d = np.append(x2d,x2drot)
        y2d = np.append(y2d,y2drot)
        
    return x,y,z,x2d,y2d


class stack_profile:

    def __init__(self,X,Y,Z,Xp,Yp,nrings,theta,nhalos):

        rin = 10.
        mp = 0.013398587e10
        step = (1000.-rin)/float(nrings)
        
        s = 1.
        q = 1.
        q2 = 1.
        
        rhop = np.zeros(nrings)
        
        Sp  = np.zeros(nrings)
        DSp  = np.zeros(nrings)
        DSp_cos  = np.zeros(nrings)
        DSp_sin  = np.zeros(nrings)
        Sp2 = np.zeros(nrings)
        rp  = np.zeros(nrings)
        mpV = np.zeros(nrings)
        mpA = np.zeros(nrings)
        
        
        Ntot = 0
        
        ring = 0
        
        rmax = 10000.
        
        while ring < nrings and (rin+step < rmax):
            
            r_in  = rin
            r_out = (rin+step)
            r_med = (rin + 0.5*step)
            rp[ring] = r_med/1.e3
            
            rpart_E_in = (X**2/r_in**2 + Y**2/r_in**2 + Z**2/r_in**2)
            rpart_E_out = (X**2/r_out**2 + Y**2/r_out**2 + Z**2/r_out**2)
            
            V    = (4./3.)*np.pi*(r_out**3 - r_in**3)
            
            
            mask = (rpart_E_in >= 1)*(rpart_E_out < 1)
            rhop[ring] = (mask.sum()*mp)/V
            mpV[ring] = mp/V
        
            # print(mask.sum())
        
            r_1 = (rin+0.4*step)
            r_2 = (rin+0.6*step)
            
            rpart_E_in  = (Xp**2/r_in**2 + Yp**2/r_in**2)
            rpart_E_out  = (Xp**2/r_out**2 + Yp**2/r_out**2)
            rpart_E_med  = (Xp**2/r_med**2 + Yp**2/r_med**2)
            
            rpart_E_1  = (Xp**2/r_1**2 + Yp**2/r_1**2)
            rpart_E_2 = (Xp**2/r_2**2 + Yp**2/r_2**2)
                
            A    = np.pi*(r_out**2 - r_in**2)
            A2   = np.pi*((rin+step*0.6)**2 - r_med**2)
            A1   = np.pi*(r_med**2 - (rin+step*0.4)**2)        
            Adisc = np.pi*(r_med**2)
            
            mask = (rpart_E_in >= 1)*(rpart_E_out < 1)
            mask1 = (rpart_E_1 >= 1)*(rpart_E_med < 1)
            mask2 = (rpart_E_med >= 1)*(rpart_E_2 < 1)
            mdisc = (rpart_E_med < 1)
            
            fi   = np.arctan2(Yp[mask],Xp[mask]) - theta
            rpar = np.sqrt(Yp**2+Xp**2)
            
            Sp[ring]  = (mask.sum()*mp)/A
            
            DSp[ring]  = (mdisc.sum()*mp)/Adisc - Sp[ring]
            
            psi2       = -1.*(mp*np.sum(rpar[mdisc]**2))/Adisc 
            S2         = -1.*r_med*mp*((mask2.sum()/A2 - mask1.sum()/A1)/(0.2*step))
            
            DSp_cos[ring]  = ((-6*psi2/r_med**2) - 2.*Sp[ring] - S2)
            DSp_sin[ring]  = (-6*psi2/r_med**2) - 4.*Sp[ring]
            Sp2[ring]      = (np.cos(2*fi).sum()*mp)/(np.pi*r_med*step)
            
            mpA[ring] = mp/A
            rin += step
            ring += 1
    
        self.r      = rp
        self.rho    = rhop/(nhalos*1.e3**3)
        self.S      = Sp/(nhalos*1.e3**2)
        self.DS     = DSp/(nhalos*1.e3**2)
        self.S2     = Sp2/(nhalos*1.e3**2)
        self.DS_cos = DSp_cos/(nhalos*1.e3**2)
        self.DS_sin = DSp_sin/(nhalos*1.e3**2)

class profile_from_map:

    def __init__(self,Xp,Yp,nhalos,RIN=100.,ROUT=1500.,ndots=20,resolution=500):

        # MAKE KAPPA MAP
        mp = 0.013398587e10
        xedges = np.linspace(-2,2,resolution)
        lsize  = np.diff(xedges)[0]
        xb, yb = np.meshgrid(xedges[:-1],xedges[:-1])+(lsize/2.)
        
        H, xedges, xedges = np.histogram2d(Xp*1.e-3, Yp*1.e-3, bins=(xedges,xedges))
        kE = (H*mp)/(nhalos*((lsize*1.e6)**2))
        kB = np.zeros(kE.shape)
        e1, e2 = ks93inv(kE, kB)
        
        xb = xb.flatten()
        yb = yb.flatten()
        
        r = np.sqrt(xb**2+yb**2)
        theta  = np.arctan2(yb,xb)
        
        #get tangential ellipticities 
        et = (-e1.flatten()*np.cos(2*theta)-e2.flatten()*np.sin(2*theta))
        #get cross ellipticities
        ex = (-e1.flatten()*np.sin(2*theta)+e2.flatten()*np.cos(2*theta))
        kE = kE.flatten()
        
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
        
        
        self.r     = R
        self.S     = SIGMA    
        self.DS_T  = DSIGMA_T 
        self.DS_X  = DSIGMA_X 
        self.S2    = -1.*SIGMAcos 
        self.GT    = -1.*GAMMATcos
        self.GX    = GAMMAXsin
        
        
