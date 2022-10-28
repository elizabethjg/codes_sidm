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


def stack_profile(X,Y,Z,Xp,Yp,nrings,theta,nhalos):


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

        abin_1 = (rin+step*0.5-0.1*step)/np.sqrt(q2) 
        bbin_1 = abin_1*q2

        abin_2 = (rin+step*0.5+0.1*step)/np.sqrt(q2) 
        bbin_2 = abin_2*q2

        abin_med = (rp[ring]*1.e3)/np.sqrt(q2) 
        bbin_med = abin_med*q2
        
        rpart_E_in  = (Xp**2/abin_in**2 + Yp**2/bbin_in**2)
        rpart_E_out  = (Xp**2/abin_out**2 + Yp**2/bbin_out**2)
        rpart_E_med  = (Xp**2/abin_med**2 + Yp**2/bbin_med**2)
        
        rpart_E_1  = (Xp**2/abin_1**2 + Yp**2/bbin_1**2)
        rpart_E_2 = (Xp**2/abin_2**2 + Yp**2/bbin_2**2)
            
        A    = np.pi*(((rin+step)/1.e3)**2 - (rin/1.e3)**2)
        
        A2   = np.pi*(((rin+step*0.5+0.1*step)/1.e3)**2 - ((rin+step*0.5)/1.e3)**2)
        A1   = np.pi*(((rin+step*0.5)/1.e3)**2 - ((rin+step*0.5-0.1*step)/1.e3)**2)
        
        # A2   = np.pi*(((rin+2.*step)/1.e3)**2 - (rin/1.e3)**2)
        
        Aout = np.pi*(rp[ring]**2)
        
        mask = (rpart_E_in >= 1)*(rpart_E_out < 1)
        mask1 = (rpart_E_1 >= 1)*(rpart_E_med < 1)
        mask2 = (rpart_E_2 < 1)*(rpart_E_med >= 1)
        mout = (rpart_E_med < 1)
        
        fi   = np.arctan2(Yp[mask],Xp[mask]) - theta
        rpar = np.sqrt(Yp**2+Xp**2)/1.e3
        
        Sp[ring]  = (mask.sum()*mp)/A
        
        DSp[ring]  = (mout.sum()*mp)/Aout - Sp[ring]
        
        psi2       = -1.*(mp*np.sum(rpar[mout]**2))/Aout 
        S2         = -1.*rp[ring]*((mask2.sum()*mp)/A2 - (mask1.sum()*mp)/A1)/((0.2*step)/1.e3)
        
        # if ring > 0:
            # S2         = ((mask.sum()*mp)/A - Sp[ring])/(step/1.e3)
        # else:
            # S2         = Sp[ring]/(step/1.e3)
        
        
        DSp_cos[ring]  = ((-6*psi2/rp[ring]**2) - 2.*Sp[ring] - S2)
        
        DSp_sin[ring]  = (-6*psi2/rp[ring]**2) - 4.*Sp[ring]
        
        Sp2[ring] = ((np.cos(2*fi).sum()*mp))/(np.pi*rp[ring]*(step/1.e3))
        
        mpA[ring] = mp/A
        rin += step
        ring += 1
    
    return rp,rhop/nhalos,Sp/nhalos,DSp/nhalos,Sp2/nhalos,DSp_cos/nhalos,DSp_sin/nhalos


def stack_profile_forhalo(X,Y,Z,Xp,Yp,nrings,theta,nhalos=1):


    rin = 10.
    mp = 0.013398587e10
    step = (1000.-rin)/float(nrings)
    
    
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
        
        ##############
        # PERFIL 3D
        ##############
        
        r_in  = rin   
        r_out = (rin+step)
        r_med     = (rin + 0.5*step)
        
        rpart_E_in = (X**2/r_in**2 + Y**2/r_in**2 + Z**2/r_in**2)
        rpart_E_out = (X**2/r_out**2 + Y**2/r_out**2 + Z**2/r_out**2)
        
        V    = (4./3.)*np.pi*(r_out**3 - r_in**3)
        
        
        mask = (rpart_E_in >= 1)*(rpart_E_out < 1)
        rhop[ring] = (mask.sum()*mp)/V
        mpV[ring] = mp/V
    
        ##############
        # PERFILES 2D
        ##############
    

        r_1 = (rin+step*0.4)
        r_2 = (rin+step*0.6)
        
        rpart_E_in   = (Xp**2/r_in**2 + Yp**2/r_in**2)
        rpart_E_out  = (Xp**2/r_out**2 + Yp**2/r_out**2)
        rpart_E_med  = (Xp**2/r_med**2 + Yp**2/r_med**2)
        
        rpart_E_1    = (Xp**2/r_1**2 + Yp**2/r_1**2)
        rpart_E_2    = (Xp**2/r_2**2 + Yp**2/r_2**2)
                        
        A     = np.pi*(r_out**2 - r_in**2)
        A1    = np.pi*(r_med**2 - r_1**2)
        A2    = np.pi*(r_2**2 - r_med**2)
        
        Adisc = np.pi*(r_med**2)
        
        mask  = (rpart_E_in >= 1)*(rpart_E_out < 1)
        mask1 = (rpart_E_1 >= 1)*(rpart_E_med < 1)
        mask2 = (rpart_E_med >= 1)*(rpart_E_2 < 1)
        mdisc = (rpart_E_med < 1)
        
        fi   = np.arctan2(Yp[mask],Xp[mask]) - theta
        rpar = np.sqrt(Yp**2+Xp**2)
        
        Sp[ring]  = (mask.sum()*mp)/A
        
        DSp[ring]  = (mdisc.sum()*mp)/Adisc - Sp[ring]
        
        psi2       = -1.*(mp*np.sum(rpar[mdisc]**2))/Adisc
        
        S2         = -1.*r_med*mp*((mask2.sum()/A2 - mask1.sum()/A1)/(0.2*step))
    
        DSp_cos[ring]  = (-6*psi2/r_med**2) - 2.*Sp[ring] - S2
        
        DSp_sin[ring]  = (-6*psi2/r_med**2) - 4.*Sp[ring]
        
        Sp2[ring]      = (np.cos(2*fi).sum()*mp)/(np.pi*r_med*step)
        
        rp[ring] = r_med/1.e3
        
        mpA[ring] = mp/A
        
        rin += step
        ring += 1
    
    return rp,rhop/nhalos,Sp/nhalos,DSp/nhalos,Sp2/nhalos,DSp_cos/nhalos,DSp_sin/nhalos
