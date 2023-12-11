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
params = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.045, 'sigma8': 0.811, 'ns': 0.96}

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
                    
    sampler.run_mcmc(pos, 1000, progress=True)
    
    mcmc_out = sampler.get_chain(flat=True)
    
    return np.median(mcmc_out[3000:]),mcmc_out

def fit_quadrupoles_2terms(R,gt,gx,egt,egx,GT,GX,GT_2h,GX_2h,fit_components):
    
    print('fitting components: ',fit_components)
    def log_likelihood(data_model, R, profiles, eprofiles):
        
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
                    
    sampler.run_mcmc(pos, 1000, progress=True)
    
    mcmc_out = sampler.get_chain(flat=True).T
    
    return np.median(mcmc_out[0][3000:]),np.median(mcmc_out[1][3000:]),mcmc_out[0],mcmc_out[1]

def fit_quadrupoles_2terms_qrfunc(R,gt,gx,egt,egx,GT,GX,GT_2h,GX_2h,fit_components):
    
    print('fitting components: ',fit_components)
    def log_likelihood(data_model, R, profiles, eprofiles):
        
        a, b, q2h = data_model
        q1h = b*R**a
        
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
        
        a, b, q2h = data_model
        
        if -0.5 < a < 0. and 0. < b < 1. and 0. < q2h < 1.:
            return log_likelihood(data_model, R, profiles, eprofiles)
            
        return -np.inf
    
    # initializing
    
    pos = np.array([np.random.uniform(-0.1,0.,15),
                    np.random.uniform(0.6,0.9,15),
                    np.random.uniform(0.1,0.5,15)]).T
    
    nwalkers, ndim = pos.shape
    
    #-------------------
    # running emcee
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(R,[gt,gx],[egt,egx]))
                                    # pool = pool)
                    
    sampler.run_mcmc(pos, 1000, progress=True)
    
    mcmc_out = sampler.get_chain(flat=True).T
    
    return np.median(mcmc_out[0][3000:]),np.median(mcmc_out[1][3000:]),np.median(mcmc_out[2][3000:]),mcmc_out[0],mcmc_out[1],mcmc_out[2]


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
        
        if 12.5 < lM200 < 16.0 and 1 < c200 < 10:
            return log_likelihood_DS(data_model, R, profiles, iCOV)
            
        return -np.inf
        
    # initializing

    t1 = time.time()
    pos = np.array([np.random.uniform(12.5,15.5,15),
                    np.random.uniform(4,8,15)]).T
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

def fit_Delta_Sigma_core_2h(R,zmean,ds,eds,ncores):
    
    print('fitting Delta_Sigma')
    
    def log_likelihood_DS(data_model, R, ds, eds):
    
        lM200, c200, bm1 = data_model
        
        
        DS_1h = Delta_Sigma_NFW_cored_parallel(R,zmean,M200 = 10**lM200,b=1./bm1,c200=c200,ncores=ncores)
        DS_2h = Delta_Sigma_NFW_2h_parallel(R,zmean,M200 = 10**lM200,c200=5,cosmo_params=params,terms='2h',ncores=ncores,limint=100e3)
        
        DS = DS_1h + DS_2h
        
        sigma2 = eds**2
        
        L = -0.5 * np.sum((DS - ds)**2 / sigma2 + np.log(2.*np.pi*sigma2))

        return L
    
    def log_probability_DS(data_model, R, profiles, iCOV):
        
        lM200, c200, bm1 = data_model
        
        if 12.5 < lM200 < 16.0 and 1 < c200 < 10 and 0 < bm1 < 3:
            return log_likelihood_DS(data_model, R, profiles, iCOV)
            
        return -np.inf
        
    # initializing

    t1 = time.time()
    pos = np.array([np.random.uniform(13.0,14.0,15),
                    np.random.uniform(4,8,15),
                    np.random.uniform(0,0.5,15)]).T
    nwalkers, ndim = pos.shape
    
    sampler_DS = emcee.EnsembleSampler(nwalkers, ndim, log_probability_DS, 
                                    args=(R,ds,eds))

    state = sampler_DS.run_mcmc(pos, 100, progress=True)
    sampler_DS.reset()
    sampler_DS.run_mcmc(state, 500, progress=True)

    mcmc_out_DS = sampler_DS.get_chain(flat=True).T
    lM     = np.percentile(mcmc_out_DS[0][1500:], [16, 50, 84])
    c200   = np.percentile(mcmc_out_DS[1][1500:], [16, 50, 84])
    bm1    = np.percentile(mcmc_out_DS[2][1500:], [16, 50, 84])
    t2 = time.time()
    
    print('TIME DS')    
    print((t2-t1)/60.)
    
    return lM,c200,bm1,mcmc_out_DS[0],mcmc_out_DS[1],mcmc_out_DS[2]


def rotate_for_halo(j,path,main,reduced=False,iterative=False):
    
        halo = h5py.File(path+'halo_'+str(j)+'.hdf5','r')       
        
        X = np.array(halo['X']) - main.xc_rc[j]/1.e3
        Y = np.array(halo['Y']) - main.yc_rc[j]/1.e3
        Z = np.array(halo['Z']) - main.zc_rc[j]/1.e3
        
        Xp_xy,Yp_xy = projected_coodinates(X,Y,Z,main.xc_rc[j],main.yc_rc[j],main.zc_rc[j])
        Xp_zx,Yp_zx = projected_coodinates(Z,X,Y,main.xc_rc[j],main.yc_rc[j],main.zc_rc[j])
        Xp_yz,Yp_yz = projected_coodinates(Y,Z,X,main.xc_rc[j],main.yc_rc[j],main.zc_rc[j])
        
        del(X,Y,Z)
        
        Xp = np.concatenate((Xp_xy,Xp_zx,Xp_yz))
        Yp = np.concatenate((Yp_xy,Yp_zx,Yp_yz))
        
        nparts = len(Xp_xy)

        del(Xp_xy,Yp_xy,Xp_zx,Yp_zx,Xp_yz,Yp_yz)

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
    
                x2drot = (a2Drx*Xp)+(a2Dry*Yp)
                y2drot = (b2Drx*Xp)+(b2Dry*Yp)
    
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
    
    
                x2drot = (a2Dx*Xp)+(a2Dy*Yp)
                y2drot = (b2Dx*Xp)+(b2Dy*Yp)
        
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
    
                x2drot = (a2Drx*Xp)+(a2Dry*Yp)
                y2drot = (b2Drx*Xp)+(b2Dry*Yp)
    
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
    
    
                x2drot = (a2Dx*Xp)+(a2Dy*Yp)
                y2drot = (b2Dx*Xp)+(b2Dy*Yp)
                
        m2d = (np.abs(x2drot) < 10.) & (np.abs(y2drot) < 10.)

        Xp, Yp   = x2drot[m2d],   y2drot[m2d] # 2D coordinates in kpc

        del(x2drot, y2drot)
        
        return Xp, Yp    

def make_shape_profile(main_file,path,haloids,rlims,reduced,iterative):

    nbins = len(rlims)
    count = np.zeros(nbins)
    T2D   = np.zeros((nbins,2,2))
    main  = pd.read_csv(main_file)

    for j in haloids:
        
      Xp, Yp = rotate_for_halo(j,path,main,reduced,iterative)

      r = np.sqrt(Xp**2+Yp**2)
      
      if reduced:
          Wp = (1./r**2)
      else:
          Wp = np.ones(len(r))
 
      for i in range(nbins):

        m = (r <= rlims[i])
        xp, yp, wp = Xp[m], Yp[m], Wp[m]
        
        T2D[i,0,0] += np.sum(wp*xp**2)
        T2D[i,0,1] += np.sum(wp*xp*yp)
        T2D[i,1,0] += np.sum(wp*xp*yp)
        T2D[i,1,1] += np.sum(wp*yp**2)
        count[i]   += float(m.sum())
       
    return count, T2D        
 
def make_shape_profile_unpack(minput):
	return make_shape_profile(*minput)

def make_shape_profile_parallel(main_file,path,haloids,reduced = False, iterative = False, rmin=0.2, rmax=2.0, nbins=10, ncores=10):

    if ncores > len(haloids):
        ncores = len(haloids)
    
    hids_splitted = np.array_split(haloids,ncores)
    ncores = len(hids_splitted)
    
    rlims = np.linspace(rmin,rmax,nbins)
    b_profile  = np.zeros(nbins)
    a_profile  = np.zeros(nbins)
    fi_profile = np.zeros(nbins)
    count = np.zeros(nbins)
    T2D   = np.zeros((nbins,2,2))

    list_mfile      = [main_file]*ncores
    list_path       = [path]*ncores
    list_rlims      = [rlims]*ncores
    list_reduced    = [reduced]*ncores
    list_iterative  = [iterative]*ncores
    
    entrada = np.array([list_mfile,list_path,hids_splitted,list_rlims,list_reduced,list_iterative],dtype=object).T
    
    pool = Pool(processes=(ncores))
    salida = list(pool.map(make_shape_profile_unpack, entrada))
    pool.terminate()
     
    while len(salida) > 0:
      tmp_count, tmp_T2D = salida[0]
      count += tmp_count
      T2D   += tmp_T2D
      salida.pop(0)
    
    for i in range(nbins):

      if count[i] == 0:
        continue

      T2D[i] /= count[i]
      w2d, v2d = np.linalg.eig(T2D[i])
       
      j = np.flip(np.argsort(w2d))
      a_profile[i]  = np.sqrt(w2d[j][0])
      b_profile[i]  = np.sqrt(w2d[j][1])
      fi_profile[i] = np.arctan2(v2d[j[0]][1], v2d[j[0]][0])
    
    return rlims, a_profile, b_profile, fi_profile

def stack_halos_2DH(main_file,path,haloids,reduced = False, iterative = False, resolution=1000):

    main = pd.read_csv(main_file)

    H = np.zeros((resolution-1, resolution-1))
    
    xedges = np.linspace(-8,8,resolution)
    xedges = xedges[abs(xedges)>0.001]
    lsize  = np.diff(xedges)[0]
    xb, yb = np.meshgrid(xedges[:-1],xedges[:-1])+(lsize/2.)
 
    for j in haloids:
        
        Xp, Yp = rotate_for_halo(j,path,main,reduced,iterative)
        
        tmp_H, _, _ = np.histogram2d(Xp, Yp, bins=(xedges,xedges))
        H += tmp_H

        # MAKE KAPPA MAP
       
        ###Nchunck = 10
        ###Nelements = len(Xp)
        ###indices = np.array_split(np.arange(Nelements), Nchunck)

        ###idx = 0
        ###while len(indices) > 0:
        ###  index = indices[0]
        ###  #print("Histogram.. %d/%d - %d/%d" % (idx,Nchunck,len(index),Nelements))
        ###  tmp_H, _, _ = np.histogram2d(Xp[index], Yp[index], bins=(xedges,xedges))
        ###  H += tmp_H
        ###  idx += 1
        ###  indices.pop(0)
    
    return H 

def stack_halos_2DH_unpack(minput):
	return stack_halos_2DH(*minput)

def stack_halos_parallel(main_file,path,haloids,
                         reduced = False,iterative = False,
                         ncores=10, resolution=1000):
    

    if ncores > len(haloids):
        ncores = len(haloids)
    
    slicer = int(round(len(haloids)/float(ncores), 0))
    slices = ((np.arange(ncores-1)+1)*slicer).astype(int)
    slices = slices[(slices < len(haloids))]
    hids_splitted = np.split(haloids,slices)
    
    ncores = len(hids_splitted)
    
    list_mfile      = [main_file]*ncores
    list_path       = [path]*ncores
    list_reduced    = [reduced]*ncores
    list_iterative  = [iterative]*ncores
    list_resolution = [resolution]*ncores
            
    entrada = np.array([list_mfile,list_path,hids_splitted,list_reduced,list_iterative,list_resolution],dtype=object).T
    
    pool = Pool(processes=(ncores))
    salida = list(pool.map(stack_halos_2DH_unpack, entrada))
    pool.terminate()

    H = np.zeros((resolution-1, resolution-1))

    while len(salida) > 0:
        tmp_H = salida[0]
        H += tmp_H
        salida.pop(0)
    
    return H 

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

    def __init__(self,H,nhalos,RIN=100.,ROUT=5000.,ndots=20,resolution=1000):

        # MAKE KAPPA MAP
        mp = 0.013398587e10
        xedges = np.linspace(-8,8,resolution)
        xedges = xedges[abs(xedges)>0.001]
        lsize  = np.diff(xedges)[0]
        xb, yb = np.meshgrid(xedges[:-1],xedges[:-1])+(lsize/2.)

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
        
'''
class map_and_fit_profiles(profile_from_map):

    def __init__(self,H,nhalos,
                 RIN=100.,ROUT=1000.,ndots=20,
                 resolution=1000,params=params,z=0.,
                 twohalo = False,
                 ncores=36):
        
        
        # COMPUTE PROFILES
        
        profile_from_map.__init__(self,H,nhalos,RIN,ROUT,ndots,resolution)
        
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
'''

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
            
            ##################
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
            
            
            ##################        
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


            ##################        
            # FIT SHEAR QUADRUPOLE PROFILES WITH RADIAL VARIATION
                        
            # FIT THEM TOGETHER
                    
            a,b,q2hr,mcmc_a,mcmc_b,mcmc_q2hr = fit_quadrupoles_2terms_qrfunc(r,GT,GX,eGT,eGX,GT_func,GX_func,GT_2h_func,GX_2h_func,'both')
            
            q1h = b*r**a
            e1h = (1. - q1h)/(1. + q1h)
            e2h = (1. - q2hr)/(1. + q2hr)
            
            self.a_2g         = a
            self.b_2g         = b
            self.q2hr_2g      = q2hr
            
            self.mcmc_a_2g    = mcmc_a
            self.mcmc_b_2g    = mcmc_b
            self.mcmc_q2hr_2g = mcmc_q2hr
            
            self.GT1hr_fit2   = e1h*GT_func
            self.GX1hr_fit2   = e1h*GX_func
            self.GT2hr_fit2   = e2h*GT_2h_func
            self.GX2hr_fit2   = e2h*GX_2h_func
            
            # FIT THEM SEPARATELY
            a,b,q2hr,mcmc_a,mcmc_b,mcmc_q2hr = fit_quadrupoles_2terms_qrfunc(r,GT,GX,eGT,eGX,GT_func,GX_func,GT_2h_func,GX_2h_func,'cross')
            
            q1h = b*r**a
            e1h = (1. - q1h)/(1. + q1h)
            e2h = (1. - q2hr)/(1. + q2hr)
            
            self.a_gx         = a
            self.b_gx         = b
            self.q2hr_gx      = q2hr
            
            self.mcmc_a_gx    = mcmc_a
            self.mcmc_b_gx    = mcmc_b
            self.mcmc_q2hr_gx = mcmc_q2hr
            
            self.GX1hr   = e1h*GX_func
            self.GX2hr   = e2h*GX_2h_func

            a,b,q2hr,mcmc_a,mcmc_b,mcmc_q2hr = fit_quadrupoles_2terms_qrfunc(r,GT,GX,eGT,eGX,GT_func,GX_func,GT_2h_func,GX_2h_func,'tangential')
            
            q1h = b*r**a
            e1h = (1. - q1h)/(1. + q1h)
            e2h = (1. - q2hr)/(1. + q2hr)
            
            self.a_gt         = a
            self.b_gt         = b
            self.q2hr_gt      = q2hr
            
            self.mcmc_a_gt    = mcmc_a
            self.mcmc_b_gt    = mcmc_b
            self.mcmc_q2hr_gt = mcmc_q2hr
            
            self.GT1hr   = e1h*GT_func
            self.GT2hr   = e2h*GT_2h_func

class fit_profiles_with_core():

    def __init__(self,z,r,S,eS,
                 S2,eS2,DS_T,eDS_T,
                 GT,eGT,GX,eGX,
                 twohalo = True,
                 ncores = 36):
        
        # COMPUTE PROFILES
        
            
        ##################
        # FIT SHEAR PROFILE
    
        lM,cfit,bm1_fit,mcmc_ds_lM,mcmc_ds_c200,mcmc_ds_bm1 = fit_Delta_Sigma_core_2h(r,z,DS_T,eDS_T,ncores)

        e_lM200 = np.diff(lM)
        e_c200  = np.diff(cfit)
        e_bm1  = np.diff(bm1_fit)
        logM200 = lM[1]
        c200    = cfit[1]
        bm1     = bm1_fit[1]

        self.DS1hc_fit   = Delta_Sigma_NFW_cored_parallel(r,z,M200 = 10**logM200,b=1./bm1,c200=c200,ncores=ncores)
        self.DS2hc_fit   = Delta_Sigma_NFW_2h_parallel(r,z,M200 = 10**logM200,c200=c200,cosmo_params=params,terms='2h',ncores=ncores)
        
        self.DSc_fit     = self.DS1hc_fit + self.DS2hc_fit
        self.lM200c_ds = logM200
        self.c200c_ds  = c200
        self.bm1_ds   = bm1
        self.e_c200c_ds  = e_c200
        self.e_lM200c_ds  = e_lM200
        self.e_bm1_ds  = e_bm1
        self.mcmc_ds_lMc  = mcmc_ds_lM
        self.mcmc_ds_c200c  = mcmc_ds_c200
        self.mcmc_ds_bm1  = mcmc_ds_bm1
        
        ##################        
        # FIT SHEAR QUADRUPOLE PROFILES
        GTc_func,GXc_func = GAMMA_components(r,z,ellip=1.,M200 = 10**logM200,c200=c200,b=1./bm1,terms='1h',pname='NFW-core')        
        GTc_2h_func,GXc_2h_func = GAMMA_components(r,z,ellip=1.,M200 = 10**logM200,c200=c200,cosmo_params=params,terms='2h')            
        
        # FIT THEM TOGETHER
                
        q1h,q2h,mcmc_q1h,mcmc_q2h = fit_quadrupoles_2terms(r,GT,GX,eGT,eGX,GTc_func,GXc_func,GTc_2h_func,GXc_2h_func,'both')
        
        e1h = (1. - q1h)/(1. + q1h)
        e2h = (1. - q2h)/(1. + q2h)
        
        self.q1hc_2g      = q1h
        self.q2hc_2g      = q2h
        self.mcmc_q1hc_2g = mcmc_q1h
        self.mcmc_q2hc_2g = mcmc_q2h
        self.GT1hc_fit2   = e1h*GTc_func
        self.GX1hc_fit2   = e1h*GXc_func
        self.GT2hc_fit2   = e2h*GTc_2h_func
        self.GX2hc_fit2   = e2h*GXc_2h_func
        


class quadrupoles_from_map_model:
    
    def __init__(self,M200,c200,
                RIN,ROUT,ndots,
                z=0,resolution=2000,
                cosmo_params=params,
                pname='NFW',
                b=1.e3):
        
        
        # MAKE KAPPA MAP
        xedges = np.linspace(-8,8,resolution)
        xedges = xedges[abs(xedges)>0.001]
        lsize  = np.diff(xedges)[0]
        xb, yb = np.meshgrid(xedges[:-1],xedges[:-1])+(lsize/2.)

        r = np.sqrt(xb**2+yb**2)
        theta  = np.arctan2(yb,xb)
        
        self.r     = r
        self.theta = theta
        self.M200  = M200
        self.c200  = c200
        self.z     = z
        self.b     = b
        self.params = params
        self.ndots  = ndots
        self.pname  = pname

        xb = xb.flatten()
        yb = yb.flatten()

        self.rp     = np.sqrt(xb**2+yb**2)
        self.thetap = np.arctan2(yb,xb)

        bines = np.round(np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1),0)
        self.dig = np.digitize(self.rp*1.e3,bines)
        self.R = (bines[:-1] + np.diff(bines)*0.5)*1.e-3
        
        GT_2h,GX_2h = GAMMA_components(self.R,self.z,ellip=1.,
                                        M200 = self.M200,c200=self.c200,
                                        cosmo_params=self.params,terms='2h')
                                        
        self.GT_2h = GT_2h
        self.GX_2h = GX_2h

    
    def __call__(self,a,b):

        qr = b*self.r**a
        Rellip = np.sqrt((self.r**2)*(qr*(np.cos(self.theta))**2 + (1./qr)*(np.sin(self.theta))**2))                
        if self.pname == 'NFW':
            Sellip = Sigma_NFW_2h(Rellip,self.z,self.M200,self.c200,cosmo_params=self.params)
        elif self.pname == 'NFW-core':
            Sellip = np.reshape(Sigma_NFW_cored_parallel(Rellip.flatten(),self.z,self.M200,self.b,self.c200),Rellip.shape)

        

        kE = Sellip 
        kB = np.zeros(kE.shape)

        e1, e2   = ks93inv(kE, kB)

        #get tangential ellipticities 
        et = (-e1.flatten()*np.cos(2*self.thetap)-e2.flatten()*np.sin(2*self.thetap))
        #get cross ellipticities
        ex = (-e1.flatten()*np.sin(2*self.thetap)+e2.flatten()*np.cos(2*self.thetap))
        # get kappa

        
        
        GAMMATcos = []
        GAMMAXsin = []
                                
        for nbin in range(self.ndots):
            mbin = self.dig == nbin+1              
                        
            GAMMATcos = np.append(GAMMATcos,np.sum(et[mbin]*np.cos(2.*self.thetap[mbin]))/np.sum(np.cos(2.*self.thetap[mbin])**2))
            GAMMAXsin = np.append(GAMMAXsin,np.sum(ex[mbin]*np.sin(2.*self.thetap[mbin]))/np.sum(np.sin(2.*self.thetap[mbin])**2))
       
        return {'GT' : GAMMATcos,
                'GX' : -1*GAMMAXsin,
                }
        
def fit_quadrupoles_2terms_qrfunc_from_map(R,gt,gx,egt,egx,
                                           lM200,c200,z=0,
                                           RIN=100.,ROUT=5000.,ndots=20,
                                           resolution=2000,
                                           cosmo_params=params):
    M200 = 10**lM200
    
    Gterms = quadrupoles_from_map_model(M200=M200,c200=c200,
                                        resolution=resolution,
                                        RIN=RIN,ROUT=ROUT,
                                        ndots=ndots)
    
    def log_likelihood(data_model, R, profiles, eprofiles):
        
        a, b, q2h = data_model
        q1h = b*R**a
        
        gt, gx   = profiles
        egt, egx = eprofiles
        
        G1h = Gterms(a,b)
        
        e2h   = (1.-q2h)/(1.+q2h)
        
        sigma2 = egt**2
        mGT = G1h['GT'] + e2h*Gterms.GT_2h
        LGT = -0.5 * np.sum((mGT - gt)**2 / sigma2 + np.log(2.*np.pi*sigma2))
        
        mGX = G1h['GX'] + e2h*Gterms.GX_2h
        sigma2 = egx**2
        LGX = -0.5 * np.sum((mGX - gx)**2 / sigma2 + np.log(2.*np.pi*sigma2))
        
        L = LGT +  LGX

        return L
    
    
    def log_probability(data_model, R, profiles, eprofiles):
        
        a, b, q2h = data_model
        
        if -0.5 < a < 0.5 and 0. < b < 1. and 0. < q2h < 1.:
            return log_likelihood(data_model, R, profiles, eprofiles)
            
        return -np.inf
    
    # initializing
    
    pos = np.array([np.random.uniform(-0.15,0.15,15),
                    np.random.uniform(0.4,0.8,15),
                    np.random.uniform(0.3,0.7,15)]).T
    
    nwalkers, ndim = pos.shape
    
    #-------------------
    # running emcee
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(R,[gt,gx],[egt,egx]))
                                    # pool = pool)
                    
    sampler.run_mcmc(pos, 1000, progress=True)
    
    mcmc_out = sampler.get_chain(flat=True).T
    
    # a_2g, b_2g, q2hr_2g, mcmc_a_2g, mcmc_b_2g, mcmc_q2hr_2g

    return np.median(mcmc_out[0][3000:]),np.median(mcmc_out[1][3000:]),np.median(mcmc_out[2][3000:]),mcmc_out[0],mcmc_out[1],mcmc_out[2]


#fit_quadrupoles_2terms_qrfunc_from_map(DM.r,DM.GT,DM.GX,DM.e_GT,DM.e_GX,DM.lM200_ds,DM.c200_ds)
