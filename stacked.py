import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import pandas as pd
import h5py
from member_distribution import projected_coodinates
from lenspack.image.inversion import ks93inv
from models_profiles import *
import emcee
from scipy.optimize import curve_fit
params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}

def fit_quadrupoles(R,gt,gx,egt,egx,GT,GX):
    
    
    def log_likelihood(q, R, profiles, eprofiles):
        
        gt, gx   = profiles
        egt, egx = eprofiles
        
        e   = (1.-q)/(1.+q)
        
        sigma2 = egt**2
        LGT = -0.5 * np.sum((e*GT - gt)**2 / sigma2 + np.log(2.*np.pi*sigma2))

        sigma2 = egx**2
        LGX = -0.5 * np.sum((e*GX - gx)**2 / sigma2 + np.log(2.*np.pi*sigma2))

        return LGT +  LGX
    
    
    def log_probability(q, R, profiles, eprofiles):
        
        
        if 0. < q < 1.:
            return log_likelihood(q, R, profiles, eprofiles)
            
        return -np.inf
    
    # initializing
    
    pos = np.array([np.random.uniform(0.6,0.9,15)]).T
    
    nwalkers, ndim = pos.shape
    
    #-------------------
    # running emcee
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(R,[gt,gx],[egt,egx]))
                                    # pool = pool)
                    
    sampler.run_mcmc(pos, 250, progress=True)
    
    mcmc_out = sampler.get_chain(flat=True)
    
    return np.median(mcmc_out[1500:]),mcmc_out

def fit_quadrupoles_2terms(R,gt,gx,egt,egx,GT,GX,GT_2h,GX_2h,fit_components):
    
    print('fitting components: ',fit_components)
    def log_likelihood(data_model, R, profiles, eprofiles,
                       fit_components = 'both'):
        
        q1h, q2h = data_model
        
        gt, gx   = profiles
        egt, egx = eprofiles
        
        e1h   = (1.-q1h)/(1.+q1h)
        e2h   = (1.-q2h)/(1.+q2h)
        
        sigma2 = egt**2
        mGT = e1h*GT + e2h*GT_2h
        LGT = -0.5 * np.sum((mGT - gt)**2 / sigma2 + np.log(2.*np.pi*sigma2))
        
        mGX = e1h*GX + e2h*GX_2h
        sigma2 = egx**2
        LGX = -0.5 * np.sum((mGX - gx)**2 / sigma2 + np.log(2.*np.pi*sigma2))
        
        if fit_components == 'both':
            L = LGT +  LGX
        if fit_components == 'tangential':
            L = LGT
        if fit_components == 'cross':
            L = LGX

        return L
    
    
    def log_probability(data_model, R, profiles, eprofiles):
        
        q1h, q2h = data_model
        
        if 0. < q1h < 1. and 0. < q2h < 1.:
            return log_likelihood(data_model, R, profiles, eprofiles)
            
        return -np.inf
    
    # initializing
    
    pos = np.array([np.random.uniform(0.6,0.9,15),
                    np.random.uniform(0.1,0.5,15)]).T
    
    nwalkers, ndim = pos.shape
    
    #-------------------
    # running emcee
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(R,[gt,gx],[egt,egx]))
                                    # pool = pool)
                    
    sampler.run_mcmc(pos, 250, progress=True)
    
    mcmc_out = sampler.get_chain(flat=True).T
    
    return np.median(mcmc_out[0][1500:]),np.median(mcmc_out[1][1500:]),mcmc_out[0],mcmc_out[1]

def fit_Delta_Sigma_2h(R,zmean,ds,eds,ncores):
    
    print('fitting Delta_Sigma')
    
    def log_likelihood_DS(data_model, R, ds, eds):
    
        lM200, c200 = data_model
        
        DS   = Delta_Sigma_NFW_2h_parallel(R,zmean,M200 = 10**lM200,c200=c200,cosmo_params=params,terms='1h+2h',ncores=ncores)
        
        sigma2 = eds**2
        
        L = -0.5 * np.sum((DS - ds)**2 / sigma2 + np.log(2.*np.pi*sigma2))

        return L
    
    def log_probability_DS(data_model, R, profiles, iCOV):
        
        lM200,c200 = data_model
        
        if 12.5 < lM200 < 16.0 and 1 < c200 < 7:
            return log_likelihood_DS(data_model, R, profiles, iCOV)
            
        return -np.inf
        
    # initializing

    t1 = time.time()
    pos = np.array([np.random.uniform(12.5,15.5,15),
                    np.random.uniform(1,5,15)]).T
    nwalkers, ndim = pos.shape
    
    sampler_DS = emcee.EnsembleSampler(nwalkers, ndim, log_probability_DS, 
                                    args=(R,ds,eds))
                                    
    sampler_DS.run_mcmc(pos, 250, progress=True)
    
    mcmc_out_DS = sampler_DS.get_chain(flat=True).T
    lM     = np.percentile(mcmc_out_DS[0][1500:], [16, 50, 84])
    c200   = np.percentile(mcmc_out_DS[1][1500:], [16, 50, 84])
    t2 = time.time()
    
    print('TIME DS')    
    print((t2-t1)/60.)
    
    return lM,c200,mcmc_out_DS[0],mcmc_out_DS[1]

def stack_halos(main_file,path,haloids,reduced = False, iterative = False):

    main = pd.read_csv(main_file)

    x = np.array([], dtype=np.float32)
    y = np.array([], dtype=np.float32)
    z = np.array([], dtype=np.float32)

    x2d = np.array([], dtype=np.float32)
    y2d = np.array([], dtype=np.float32)

    for j in haloids:
        
        halo = h5py.File(path+'halo_'+str(j)+'.hdf5','r')       

        
        X = np.array(halo['X']) - main.xc_rc[j]/1.e3
        Y = np.array(halo['Y']) - main.yc_rc[j]/1.e3
        Z = np.array(halo['Z']) - main.zc_rc[j]/1.e3
        
        ''' 3D
        if iterative:
            if reduced:
                xrot = (main.a3Drx_it[j]*X)+(main.a3Dry_it[j]*Y)+(main.a3Drz_it[j]*Z);
                yrot = (main.b3Drx_it[j]*X)+(main.b3Dry_it[j]*Y)+(main.b3Drz_it[j]*Z);
                zrot = (main.c3Drx_it[j]*X)+(main.c3Dry_it[j]*Y)+(main.c3Drz_it[j]*Z);
            else:
                xrot = (main.a3Dx_it[j]*X)+(main.a3Dy_it[j]*Y)+(main.a3Dz_it[j]*Z);
                yrot = (main.b3Dx_it[j]*X)+(main.b3Dy_it[j]*Y)+(main.b3Dz_it[j]*Z);
                zrot = (main.c3Dx_it[j]*X)+(main.c3Dy_it[j]*Y)+(main.c3Dz_it[j]*Z);
        else:
            if reduced:
                xrot = (main.a3Drx[j]*X)+(main.a3Dry[j]*Y)+(main.a3Drz[j]*Z);
                yrot = (main.b3Drx[j]*X)+(main.b3Dry[j]*Y)+(main.b3Drz[j]*Z);
                zrot = (main.c3Drx[j]*X)+(main.c3Dry[j]*Y)+(main.c3Drz[j]*Z);
            else:
                xrot = (main.a3Dx[j]*X)+(main.a3Dy[j]*Y)+(main.a3Dz[j]*Z);
                yrot = (main.b3Dx[j]*X)+(main.b3Dy[j]*Y)+(main.b3Dz[j]*Z);
                zrot = (main.c3Dx[j]*X)+(main.c3Dy[j]*Y)+(main.c3Dz[j]*Z);
        
        x = np.append(x,np.float32(xrot)) 
        y = np.append(y,np.float32(yrot))
        z = np.append(z,np.float32(zrot))
        '''
        
        X2d_xy,Y2d_xy = projected_coodinates(X,Y,Z,main.xc_rc[j],main.yc_rc[j],main.zc_rc[j])
        X2d_zx,Y2d_zx = projected_coodinates(Z,X,Y,main.xc_rc[j],main.yc_rc[j],main.zc_rc[j])
        X2d_yz,Y2d_yz = projected_coodinates(Y,Z,X,main.xc_rc[j],main.yc_rc[j],main.zc_rc[j])
        
        del(X,Y,Z)
        
        X2d = np.concatenate((X2d_xy,X2d_zx,X2d_yz))
        Y2d = np.concatenate((Y2d_xy,Y2d_zx,Y2d_yz))
            
        nparts = len(X2d_xy)

        if iterative:

            if reduced:
            
                a2Drx = np.concatenate((np.repeat(main.a2Drx_it_xy[j],nparts),
                                np.repeat(main.a2Drx_it_zx[j],nparts),
                                np.repeat(main.a2Drx_it_yz[j],nparts)))
                                
                b2Drx = np.concatenate((np.repeat(main.b2Drx_it_xy[j],nparts),
                                np.repeat(main.b2Drx_it_zx[j],nparts),
                                np.repeat(main.b2Drx_it_yz[j],nparts)))
        
                a2Dry = np.concatenate((np.repeat(main.a2Dry_it_xy[j],nparts),
                                np.repeat(main.a2Dry_it_zx[j],nparts),
                                np.repeat(main.a2Dry_it_yz[j],nparts)))
                
                b2Dry = np.concatenate((np.repeat(main.b2Dry_it_xy[j],nparts),
                                np.repeat(main.b2Dry_it_zx[j],nparts),
                                np.repeat(main.b2Dry_it_yz[j],nparts)))
    
                x2drot = (a2Drx*X2d)+(a2Dry*Y2d)
                y2drot = (b2Drx*X2d)+(b2Dry*Y2d)
    
            else:
            
                a2Dx  = np.concatenate((np.repeat(main.a2Dx_it_xy[j],nparts),
                                np.repeat(main.a2Dx_it_zx[j],nparts),
                                np.repeat(main.a2Dx_it_yz[j],nparts)))
                
                
                b2Dx  = np.concatenate((np.repeat(main.b2Dx_it_xy[j],nparts),
                                np.repeat(main.b2Dx_it_zx[j],nparts),
                                np.repeat(main.b2Dx_it_yz[j],nparts)))
        
                a2Dy  = np.concatenate((np.repeat(main.a2Dy_it_xy[j],nparts),
                                np.repeat(main.a2Dy_it_zx[j],nparts),
                                np.repeat(main.a2Dy_it_yz[j],nparts)))
                                
                b2Dy  = np.concatenate((np.repeat(main.b2Dy_it_xy[j],nparts),
                                np.repeat(main.b2Dy_it_zx[j],nparts),
                                np.repeat(main.b2Dy_it_yz[j],nparts)))
    
    
                x2drot = (a2Dx*X2d)+(a2Dy*Y2d)
                y2drot = (b2Dx*X2d)+(b2Dy*Y2d)
        
        else:
            if reduced:
            
                a2Drx = np.concatenate((np.repeat(main.a2Drx_xy[j],nparts),
                                np.repeat(main.a2Drx_zx[j],nparts),
                                np.repeat(main.a2Drx_yz[j],nparts),))
                                
                b2Drx = np.concatenate((np.repeat(main.b2Drx_xy[j],nparts),
                                np.repeat(main.b2Drx_zx[j],nparts),
                                np.repeat(main.b2Drx_yz[j],nparts)))
        
                a2Dry = np.concatenate((np.repeat(main.a2Dry_xy[j],nparts),
                                np.repeat(main.a2Dry_zx[j],nparts),
                                np.repeat(main.a2Dry_yz[j],nparts)))
                
                b2Dry = np.concatenate((np.repeat(main.b2Dry_xy[j],nparts),
                                np.repeat(main.b2Dry_zx[j],nparts),
                                np.repeat(main.b2Dry_yz[j],nparts)))
    
                x2drot = (a2Drx*X2d)+(a2Dry*Y2d)
                y2drot = (b2Drx*X2d)+(b2Dry*Y2d)
    
            else:
            
                a2Dx  = np.concatenate((np.repeat(main.a2Dx_xy[j],nparts),
                                np.repeat(main.a2Dx_zx[j],nparts),
                                np.repeat(main.a2Dx_yz[j],nparts)))
                
                
                b2Dx  = np.concatenate((np.repeat(main.b2Dx_xy[j],nparts),
                                np.repeat(main.b2Dx_zx[j],nparts),
                                np.repeat(main.b2Dx_yz[j],nparts)))
        
                a2Dy  = np.concatenate((np.repeat(main.a2Dy_xy[j],nparts),
                                np.repeat(main.a2Dy_zx[j],nparts),
                                np.repeat(main.a2Dy_yz[j],nparts)))
                                
                b2Dy  = np.concatenate((np.repeat(main.b2Dy_xy[j],nparts),
                                np.repeat(main.b2Dy_zx[j],nparts),
                                np.repeat(main.b2Dy_yz[j],nparts)))
    
    
                x2drot = (a2Dx*X2d)+(a2Dy*Y2d)
                y2drot = (b2Dx*X2d)+(b2Dy*Y2d)
                
        m2d = (np.abs(x2drot) < 10.) & (np.abs(y2drot) < 10.)
                
        x2d = np.append(x2d,np.float32(x2drot[m2d]))
        y2d = np.append(y2d,np.float32(y2drot[m2d]))
        
    return x,y,z,x2d,y2d

def stack_halos_unpack(minput):
	return stack_halos(*minput)

def stack_halos_parallel(main_file,path,haloids,
                         reduced = False,iterative = False,
                         ncores=10):
    

    if ncores > len(haloids):
        ncores = len(haloids)
    
    slicer = int(round(len(haloids)/float(ncores), 0))
    slices = ((np.arange(ncores-1)+1)*slicer).astype(int)
    slices = slices[(slices < len(haloids))]
    hids_splitted = np.split(haloids,slices)
    
    ncores = len(hids_splitted)
    
    mfile     = [main_file]*ncores
    path      = [path]*ncores
    reduced   = [reduced]*ncores
    iterative = [iterative]*ncores
            
    entrada = np.array([mfile,path,hids_splitted,reduced,iterative],dtype=object).T
    
    pool = Pool(processes=(ncores))
    salida = list(pool.map(stack_halos_unpack, entrada))
    pool.terminate()

    x   = np.array([], dtype=np.float32)
    y   = np.array([], dtype=np.float32)
    z   = np.array([], dtype=np.float32)
    x2d = np.array([], dtype=np.float32)
    y2d = np.array([], dtype=np.float32)
    
    while len(salida) > 0:
        X,Y,Z,X2d,Y2d = salida[0]

        x = np.append(x,np.float32(X))
        y = np.append(y,np.float32(Y))
        z = np.append(z,np.float32(Z))
        
        x2d = np.append(x2d,np.float32(X2d))
        y2d = np.append(y2d,np.float32(Y2d))

        salida.pop(0)
    
    return x,y,z,x2d,y2d
    

class stack_profile:

    def __init__(self,X,Y,Z,Xp,Yp,nhalos,nrings=100,theta=0):

        rin = 10.
        mp = 0.013398587e10
        step = (1000.-rin)/float(nrings)
        
        s = 1.
        q = 1.
        q2 = 1.
        
        rhop = np.zeros(nrings)
        
        Sp  = np.zeros(nrings)
        DSp = np.zeros(nrings)
        Sp2 = np.zeros(nrings)        
        
        DSp_cos  = np.zeros(nrings)
        DSp_sin  = np.zeros(nrings)

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
        self.erho   = mpV/(1.e3**3)
        self.S      = Sp/(nhalos*3*1.e3**2)
        self.eS     = mpA*nhalos/(1.e3**2)
        self.DS     = DSp/(nhalos*3*1.e3**2)
        self.S2     = Sp2/(nhalos*3*1.e3**2)

class profile_from_map:

    def __init__(self,Xp,Yp,nhalos,RIN=100.,ROUT=1500.,ndots=20,resolution=500):

        # MAKE KAPPA MAP
        mp = 0.013398587e10
        xedges = np.linspace(-8,8,resolution)
        lsize  = np.diff(xedges)[0]
        xb, yb = np.meshgrid(xedges[:-1],xedges[:-1])+(lsize/2.)

        H = np.zeros((resolution-1, resolution-1))
        Nelements = len(Xp)
        Nchunck = 10
        indices = np.array_split(np.arange(Nelements), Nchunck)

        idx = 0
        while len(indices) > 0:
          index = indices[0]
          print("Histogram.. %d/%d - %d/%d" % (idx,Nchunck,len(index),Nelements))
          tmp_H, _, _ = np.histogram2d(Xp[index]*1.e-3, Yp[index]*1.e-3, bins=(xedges,xedges))
          H += tmp_H
          idx += 1
          indices.pop(0)

        kE = (H*mp)/(nhalos*((lsize*1.e6)**2))
        kB = np.zeros(kE.shape)
        
        ekE = (np.ones(kE.shape)*mp)/(nhalos*((lsize*1.e6)**2))
        
        e1, e2   = ks93inv(kE, kB)
        ee1, ee2 = ks93inv(ekE, kB)
        
        xb = xb.flatten()
        yb = yb.flatten()
        
        r = np.sqrt(xb**2+yb**2)
        theta  = np.arctan2(yb,xb)
        
        #get tangential ellipticities 
        et = (-e1.flatten()*np.cos(2*theta)-e2.flatten()*np.sin(2*theta))
        #get cross ellipticities
        ex = (-e1.flatten()*np.sin(2*theta)+e2.flatten()*np.cos(2*theta))

        #get tangential ellipticities 
        eet = (-ee1.flatten()*np.cos(2*theta)-ee2.flatten()*np.sin(2*theta))
        #get cross ellipticities
        eex = (-ee1.flatten()*np.sin(2*theta)+ee2.flatten()*np.cos(2*theta))

        kE  = kE.flatten()
        ekE = ekE.flatten()
        
        bines = np.round(np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1),0)
        dig = np.digitize(r*1.e3,bines)
        R = (bines[:-1] + np.diff(bines)*0.5)*1.e-3
        
        SIGMA   = []
        DSIGMA_T = []
        DSIGMA_X = []
        SIGMAcos = []
        GAMMATcos = []
        GAMMAXsin = []
        
        eSIGMA   = []
        eDSIGMA_T = []
        eDSIGMA_X = []
        eSIGMAcos = []
        eGAMMATcos = []
        eGAMMAXsin = []
                        
                            
        for nbin in range(ndots):
            mbin = dig == nbin+1              
            
            SIGMA     = np.append(SIGMA,np.mean(kE[mbin]))
            DSIGMA_T  = np.append(DSIGMA_T,np.mean(et[mbin]))
            DSIGMA_X  = np.append(DSIGMA_X,np.mean(ex[mbin]))
            
            SIGMAcos  = np.append(SIGMAcos,np.sum(kE[mbin]*np.cos(2.*theta[mbin]))/np.sum(np.cos(2.*theta[mbin])**2))
            
            GAMMATcos = np.append(GAMMATcos,np.sum(et[mbin]*np.cos(2.*theta[mbin]))/np.sum(np.cos(2.*theta[mbin])**2))
            GAMMAXsin = np.append(GAMMAXsin,np.sum(ex[mbin]*np.sin(2.*theta[mbin]))/np.sum(np.sin(2.*theta[mbin])**2))

            eSIGMA     = np.append(eSIGMA,np.mean(ekE[mbin]))
            eDSIGMA_T  = np.append(eDSIGMA_T,np.mean(eet[mbin]))
            eDSIGMA_X  = np.append(eDSIGMA_X,np.mean(eex[mbin]))
            
            eSIGMAcos  = np.append(eSIGMAcos,np.sum(ekE[mbin]*np.cos(2.*theta[mbin]))/np.sum(np.cos(2.*theta[mbin])**2))
            
            eGAMMATcos = np.append(eGAMMATcos,np.sum(eet[mbin]*np.cos(2.*theta[mbin]))/np.sum(np.cos(2.*theta[mbin])**2))
            eGAMMAXsin = np.append(eGAMMAXsin,np.sum(eex[mbin]*np.sin(2.*theta[mbin]))/np.sum(np.sin(2.*theta[mbin])**2))
        
        
        self.r     = R
        self.S     = SIGMA    
        self.DS_T  = DSIGMA_T 
        self.DS_X  = DSIGMA_X 
        self.S2    = -1.*SIGMAcos 
        self.GT    = -1.*GAMMATcos
        self.GX    = GAMMAXsin
        self.eS    = eSIGMA    
        self.eDS_T = eDSIGMA_T 
        self.eDS_X = eDSIGMA_X 
        self.eS2   = -1.*eSIGMAcos 
        self.eGT   = -1.*eGAMMATcos
        self.eGX   = eGAMMAXsin
        

class map_and_fit_profiles(profile_from_map):

    def __init__(self,Xp,Yp,nhalos,
                 RIN=100.,ROUT=1000.,ndots=20,
                 resolution=500,params=params,z=0.,
                 twohalo = False,
                 ncores=36):
        
        
        # COMPUTE PROFILES
        
        profile_from_map.__init__(self, Xp,Yp,nhalos,RIN,ROUT,ndots,resolution)
        
        if not twohalo:
        
            # FIT KAPPA PROFILE

            def S(R,logM200,c200):
                return Sigma_NFW_2h(R,z,10**logM200,c200,cosmo_params=params)
    
            S_fit = curve_fit(S,self.r,self.S,sigma=np.ones(len(self.r)),absolute_sigma=True,bounds=([12,2],[15,10]))
            pcov    = S_fit[1]
            perr    = np.sqrt(np.diag(pcov))
            e_lM200 = perr[0]
            e_c200  = perr[1]
            logM200 = S_fit[0][0]
            c200    = S_fit[0][1]
            
            self.lM200_s = logM200
            self.c200_s  = c200
            self.S_fit   = S(self.r,logM200,c200)
    
            def S2(R,e):
                return e*S2_quadrupole(R,z,10**logM200,c200,cosmo_params=params)
                
            S2_fit = curve_fit(S2,self.r,self.S2,sigma=np.ones(len(self.r)),absolute_sigma=True,bounds=(0,1))
            e = S2_fit[0]
            
            self.q_s      = (1.-e)/(1.+e)
            self.S2_fit   = S2(self.r,e)

            # FIT SHEAR PROFILE
    
            def DS(R,logM200,c200):
                return Delta_Sigma_NFW_2h(R,z,10**logM200,c200,cosmo_params=params)
            
            mr = self.r < 1.
            
            DS_fit = curve_fit(DS,self.r[mr],self.DS_T[mr],sigma=self.eDS_T[mr],absolute_sigma=True,bounds=([12,2],[15,10]))
            pcov    = DS_fit[1]
            perr    = np.sqrt(np.diag(pcov))
            e_lM200 = perr[0]
            e_c200  = perr[1]
            logM200 = DS_fit[0][0]
            c200    = DS_fit[0][1]
            
            self.DS_fit   = DS(self.r,logM200,c200)
            self.lM200_ds = logM200
            self.c200_ds  = c200
        
            # FIT SHEAR QUADRUPOLE PROFILES
            # FIT THEM TOGETHER
            
            GT, GX = GAMMA_components(self.r,z,ellip=1.,M200 = 10**logM200,c200=c200,cosmo_params=params)
                    
            q_ds,mcmc_out = fit_quadrupoles(self.r,self.GT,self.GX,self.eGT,self.eGX,GT,GX)
            
            e = (1. - q_ds)/(1. + q_ds)
            
            self.q_2g     = q_ds
            self.mcmc_out = mcmc_out
            self.GT_fit2 = e*GT
            self.GX_fit2 = e*GX
            
            # FIT THEM SEPARATELY
            
            def GT(R,e):
                GT,GX = GAMMA_components(R,z,ellip=e,M200 = 10**logM200,c200=c200,cosmo_params=params)
                return GT
                
            GT_fit = curve_fit(GT,self.r,self.GT,sigma=np.ones(len(self.r)),absolute_sigma=True,bounds=(0,1))
            e = GT_fit[0]
            
            self.q_gt     = (1.-e)/(1.+e)
            self.GT_fit   = GT(self.r,e)
    
            def GX(R,e):
                GT,GX = GAMMA_components(R,z,ellip=e,M200 = 10**logM200,c200=c200,cosmo_params=params)
                return GX
                
            GX_fit = curve_fit(GX,self.r,self.GX,sigma=np.ones(len(self.r)),absolute_sigma=True,bounds=(0,1))
            e = GX_fit[0]
            
            self.q_gx     = (1.-e)/(1.+e)
            self.GX_fit   = GX(self.r,e)

        if twohalo:

            # FIT SHEAR PROFILE
    
            lM,cfit,mcmc_ds_lM,mcmc_ds_c200 = fit_Delta_Sigma_2h(self.r,z,DS_T,eDS_T,ncores)

            e_lM200 = np.diff(lM)
            e_c200  = np.diff(cfit)
            logM200 = lM[1]
            c200    = cfit[1]
            
            self.DS1h_fit   = Delta_Sigma_NFW_2h_parallel(self.r,z,M200 = 10**logM200,c200=c200,cosmo_params=params,terms='1h',ncores=ncores)
            self.DS2h_fit   = Delta_Sigma_NFW_2h_parallel(self.r,z,M200 = 10**logM200,c200=c200,cosmo_params=params,terms='2h',ncores=ncores)
            self.DS_fit     = self.DS1h_fit + self.DS2h_fit
            self.lM200_ds = logM200
            self.c200_ds  = c200
            self.e_c200_ds  = e_c200
            self.e_lM200_ds  = e_lM200
            self.mcmc_ds_lM  = mcmc_ds_lM
            self.mcmc_ds_c200  = mcmc_ds_c200
                    
            # FIT SHEAR QUADRUPOLE PROFILES
            
            GT,GX = GAMMA_components(self.r,z,ellip=1.,M200 = 10**logM200,c200=c200,cosmo_params=params,terms='1h')
            GT_2h,GX_2h = GAMMA_components(self.r,z,ellip=1.,M200 = 10**logM200,c200=c200,cosmo_params=params,terms='2h')            
            
            # FIT THEM TOGETHER
                    
            q1h,q2h,mcmc_q1h,mcmc_q2h = fit_quadrupoles_2terms(self.r,self.GT,self.GX,self.eGT,self.eGX,GT,GX,GT_2h,GX_2h,'both')
            
            e1h = (1. - q1h)/(1. + q1h)
            e2h = (1. - q2h)/(1. + q2h)
            
            self.q1h_2g      = q1h
            self.q2h_2g      = q2h
            self.mcmc_q1h_2g = mcmc_q1h
            self.mcmc_q2h_2g = mcmc_q2h
            self.GT1h_fit2   = e1h*GT
            self.GX1h_fit2   = e1h*GX
            self.GT2h_fit2   = e2h*GT_2h
            self.GX2h_fit2   = e2h*GX_2h
            
            # FIT THEM SEPARATELY
                    
            q1h,q2h,mcmc_q1h,mcmc_q2h = fit_quadrupoles_2terms(self.r,self.GT,self.GX,self.eGT,self.eGX,GT,GX,GT_2h,GX_2h,'tangential')
            
            e1h = (1. - q1h)/(1. + q1h)
            e2h = (1. - q2h)/(1. + q2h)
            
            self.q1h_gt      = q1h
            self.q2h_gt      = q2h
            self.mcmc_q1h_gt = mcmc_q1h
            self.mcmc_q2h_gt = mcmc_q2h
            self.GT1h        = e1h*GT
            self.GT2h        = e2h*GT_2h

            q1h,q2h,mcmc_q1h,mcmc_q2h = fit_quadrupoles_2terms(self.r,self.GT,self.GX,self.eGT,self.eGX,GT,GX,GT_2h,GX_2h,'cross')
            
            e1h = (1. - q1h)/(1. + q1h)
            e2h = (1. - q2h)/(1. + q2h)
            
            self.q1h_gx      = q1h
            self.q2h_gx      = q2h
            self.mcmc_q1h_gx = mcmc_q1h
            self.mcmc_q2h_gx = mcmc_q2h
            self.GX1h        = e1h*GX
            self.GX2h        = e2h*GX_2h


class fit_profiles():

    def __init__(self,z,r,S,eS,
                 S2,eS2,DS_T,eDS_T,
                 GT,eGT,GX,eGX,
                 twohalo = False,
                 ncores = 36):
        
        # COMPUTE PROFILES
        
        if not twohalo:
        
            # FIT KAPPA PROFILE

            def S_func(R,logM200,c200):
                return Sigma_NFW_2h(R,z,10**logM200,c200,cosmo_params=params)
    
            S_fit = curve_fit(S_func,r,S,sigma=eS,absolute_sigma=True,bounds=([12,2],[15,10]))
            pcov    = S_fit[1]
            perr    = np.sqrt(np.diag(pcov))
            e_lM200 = perr[0]
            e_c200  = perr[1]
            logM200 = S_fit[0][0]
            c200    = S_fit[0][1]
            
            self.lM200_s = logM200
            self.c200_s  = c200
            self.S_fit   = S_func(r,logM200,c200)
    
            def S2_func(R,e):
                return e*S2_quadrupole(R,z,10**logM200,c200,cosmo_params=params)
                
            S2_fit = curve_fit(S2_func,r,S2,sigma=eS2,absolute_sigma=True,bounds=(0,1))
            e = S2_fit[0]
            
            self.q_s      = (1.-e)/(1.+e)
            self.S2_fit   = S2_func(r,e)

            # FIT SHEAR PROFILE
    
            def DS_func(R,logM200,c200):
                return Delta_Sigma_NFW_2h(R,z,10**logM200,c200,cosmo_params=params)
            
            mr = r < 1.
            
            DS_fit = curve_fit(DS_func,r[mr],DS_T[mr],sigma=eDS_T[mr],absolute_sigma=True,bounds=([12,2],[15,10]))
            pcov    = DS_fit[1]
            perr    = np.sqrt(np.diag(pcov))
            e_lM200 = perr[0]
            e_c200  = perr[1]
            logM200 = DS_fit[0][0]
            c200    = DS_fit[0][1]
            
            self.DS_fit   = DS_func(r,logM200,c200)
            self.lM200_ds = logM200
            self.c200_ds  = c200
        
            # FIT SHEAR QUADRUPOLE PROFILES
            # FIT THEM TOGETHER
            
            GT_func,GX_func = GAMMA_components(r,z,ellip=1.,M200 = 10**logM200,c200=c200,cosmo_params=params)
                    
            q_ds,mcmc_out = fit_quadrupoles(r,GT,GX,eGT,eGX,GT_func,GX_func)
            
            e = (1. - q_ds)/(1. + q_ds)
            
            self.q_2g     = q_ds
            self.mcmc_out = mcmc_out
            self.GT_fit2 = e*GT_func
            self.GX_fit2 = e*GX_func
            
            # FIT THEM SEPARATELY
            
            def GT_func(R,e):
                GT,GX = GAMMA_components(R,z,ellip=e,M200 = 10**logM200,c200=c200,cosmo_params=params)
                return GT
                
            GT_fit = curve_fit(GT_func,r,GT,sigma=eGT,absolute_sigma=True,bounds=(0,1))
            e = GT_fit[0]
            
            self.q_gt     = (1.-e)/(1.+e)
            self.GT_fit   = GT_func(r,e)
    
            def GX_func(R,e):
                GT,GX = GAMMA_components(R,z,ellip=e,M200 = 10**logM200,c200=c200,cosmo_params=params)
                return GX
                
            GX_fit = curve_fit(GX_func,r,GX,sigma=eGX,absolute_sigma=True,bounds=(0,1))
            e = GX_fit[0]
            
            self.q_gx     = (1.-e)/(1.+e)
            self.GX_fit   = GX_func(r,e)

        if twohalo:

            # FIT SHEAR PROFILE
    
            lM,cfit,mcmc_ds_lM,mcmc_ds_c200 = fit_Delta_Sigma_2h(r,z,DS_T,eDS_T,ncores)

            e_lM200 = np.diff(lM)
            e_c200  = np.diff(cfit)
            logM200 = lM[1]
            c200    = cfit[1]

            self.DS1h_fit   = Delta_Sigma_NFW_2h_parallel(r,z,M200 = 10**logM200,c200=c200,cosmo_params=params,terms='1h',ncores=ncores)
            self.DS2h_fit   = Delta_Sigma_NFW_2h_parallel(r,z,M200 = 10**logM200,c200=c200,cosmo_params=params,terms='2h',ncores=ncores)
            self.DS_fit     = self.DS1h_fit + self.DS2h_fit
            self.lM200_ds = logM200
            self.c200_ds  = c200
            self.e_c200_ds  = e_c200
            self.e_lM200_ds  = e_lM200
            self.mcmc_ds_lM  = mcmc_ds_lM
            self.mcmc_ds_c200  = mcmc_ds_c200
                    
            # FIT SHEAR QUADRUPOLE PROFILES
            
            GT_func,GX_func = GAMMA_components(r,z,ellip=1.,M200 = 10**logM200,c200=c200,cosmo_params=params,terms='1h')
            GT_2h_func,GX_2h_func = GAMMA_components(r,z,ellip=1.,M200 = 10**logM200,c200=c200,cosmo_params=params,terms='2h')            
            
            # FIT THEM TOGETHER
                    
            q1h,q2h,mcmc_q1h,mcmc_q2h = fit_quadrupoles_2terms(r,GT,GX,eGT,eGX,GT_func,GX_func,GT_2h_func,GX_2h_func,'both')
            
            e1h = (1. - q1h)/(1. + q1h)
            e2h = (1. - q2h)/(1. + q2h)
            
            self.q1h_2g      = q1h
            self.q2h_2g      = q2h
            self.mcmc_q1h_2g = mcmc_q1h
            self.mcmc_q2h_2g = mcmc_q2h
            self.GT1h_fit2   = e1h*GT_func
            self.GX1h_fit2   = e1h*GX_func
            self.GT2h_fit2   = e2h*GT_2h_func
            self.GX2h_fit2   = e2h*GX_2h_func
            
            # FIT THEM SEPARATELY
                    
            q1h,q2h,mcmc_q1h,mcmc_q2h = fit_quadrupoles_2terms(r,GT,GX,eGT,eGX,GT_func,GX_func,GT_2h_func,GX_2h_func,'tangential')
            
            e1h = (1. - q1h)/(1. + q1h)
            e2h = (1. - q2h)/(1. + q2h)
            
            self.q1h_gt      = q1h
            self.q2h_gt      = q2h
            self.mcmc_q1h_gt = mcmc_q1h
            self.mcmc_q2h_gt = mcmc_q2h
            self.GT1h        = e1h*GT_func
            self.GT2h        = e2h*GT_2h_func

            q1h,q2h,mcmc_q1h,mcmc_q2h = fit_quadrupoles_2terms(r,GT,GX,eGT,eGX,GT_func,GX_func,GT_2h_func,GX_2h_func,'cross')
            
            e1h = (1. - q1h)/(1. + q1h)
            e2h = (1. - q2h)/(1. + q2h)
            
            self.q1h_gx      = q1h
            self.q2h_gx      = q2h
            self.mcmc_q1h_gx = mcmc_q1h
            self.mcmc_q2h_gx = mcmc_q2h
            self.GX1h        = e1h*GX_func
            self.GX2h        = e2h*GX_2h_func
