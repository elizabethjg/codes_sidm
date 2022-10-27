import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import pandas as pd
import h5py
from member_distribution import projected_coodinates

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


def stack_profile(X,Y,Z,Xp,Yp,nrings,theta):


    rin = 10.
    mp = 0.013398587e10
    step = (1000.-rin)/float(nrings)
    
    s = 1.
    q = 1.
    q2 = 1.
    
    rhop = np.zeros(nrings)
    
    Sp  = np.zeros(nrings)
    Sp2 = np.zeros(nrings)
    rp  = np.zeros(nrings)
    mpV = np.zeros(nrings)
    mpA = np.zeros(nrings)
    
    
    Ntot = 0
    
    ring = 0
    
    rmax = 10000.
    
    while ring < (nrings-1) and (rin+step < rmax):
        
        abin_in = rin/(q*s)**(1./3.)
        bbin_in = abin_in*q
        cbin_in = abin_in*s
    
        abin_out = (rin+step)/(q*s)**(1./3.)
        bbin_out = abin_out*q
        cbin_out = abin_out*s
        
        rp[ring] = (rin + 0.5*step)/1.e3
        
        rpart_E_in = (X**2/abin_in**2 + Y**2/bbin_in**2 + Z**2/cbin_in**2)
        rpart_E_out = (X**2/abin_out**2 + Y**2/bbin_out**2 + Z**2/cbin_out**2)
        
        V    = (4./3.)*np.pi*(((rin+step)/1.e3)**3 - (rin/1.e3)**3)
        mask = (rpart_E_in >= 1)*(rpart_E_out < 1)
        rhop[ring] = (mask.sum()*mp)/V
        mpV[ring] = mp/V
    
        # print(mask.sum())
    
        abin_in = rin/np.sqrt(q2) 
        bbin_in = abin_in*q2
    
        abin_out = (rin+step)/np.sqrt(q2) 
        bbin_out = abin_out*q2
    
        rpart_E_in = (Xp**2/abin_in**2 + Yp**2/bbin_in**2)
        rpart_E_out = (Xp**2/abin_out**2 + Yp**2/bbin_out**2)
            
        A    = np.pi*(((rin+step)/1.e3)**2 - (rin/1.e3)**2)
        mask = (rpart_E_in >= 1)*(rpart_E_out < 1)
    
        fi = np.arctan2(Yp,Xp) - theta
    
        Sp[ring]  = (mask.sum()*mp)/A
        Sp2[ring] = ((np.cos(2*fi[mask]).sum()*mp)/A)
        
        mpA[ring] = mp/A
        rin += step
        ring += 1
    
    return rp,rhop,Sp,Sp2
