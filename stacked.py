import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
import pandas as pd
import h5py
from member_distribution import projected_coodinates

def stack_halos(main_file,path,haloids):

    main = pd.read_csv(main_file)
    
    x = np.array([])
    y = np.array([])
    z = np.array([])
    
    x2d = np.array([])
    y2d = np.array([])

    xr = np.array([])
    yr = np.array([])
    zr = np.array([])
    
    x2dr = np.array([])
    y2dr = np.array([])
    
    
    for j in haloids:
        
        halo = h5py.File(path+'halo_'+str(j)+'.hdf5','r')       
        
        X = np.array(halo['X']) - main.xc_rc[j]/1.e3
        Y = np.array(halo['Y']) - main.yc_rc[j]/1.e3
        Z = np.array(halo['Z']) - main.zc_rc[j]/1.e3
        
        xrot = (main.a3Dx[j]*X)+(main.a3Dy[j]*Y)+(main.a3Dz[j]*Z);
        yrot = (main.b3Dx[j]*X)+(main.b3Dy[j]*Y)+(main.b3Dz[j]*Z);
        zrot = (main.c3Dx[j]*X)+(main.c3Dy[j]*Y)+(main.c3Dz[j]*Z);

        xrot_r = (main.a3Drx[j]*X)+(main.a3Dry[j]*Y)+(main.a3Drz[j]*Z);
        yrot_r = (main.b3Drx[j]*X)+(main.b3Dry[j]*Y)+(main.b3Drz[j]*Z);
        zrot_r = (main.c3Drx[j]*X)+(main.c3Dry[j]*Y)+(main.c3Drz[j]*Z);
        
        x = np.append(x,xrot)
        y = np.append(y,yrot)
        z = np.append(z,zrot)

        xr = np.append(xr,xrot_r)
        yr = np.append(yr,yrot_r)
        zr = np.append(zr,zrot_r)
        
        X2d,Y2d = projected_coodinates(X,Y,Z,main.xc_rc[j],main.yc_rc[j],main.zc_rc[j])
        
        x2drot = (main.a2Dx[j]*X2d)+(main.a2Dy[j]*Y2d)
        y2drot = (main.b2Dx[j]*X2d)+(main.b2Dy[j]*Y2d)

        x2drot_r = (main.a2Drx[j]*X2d)+(main.a2Dry[j]*Y2d)
        y2drot_r = (main.b2Drx[j]*X2d)+(main.b2Dry[j]*Y2d)
    
        x2d = np.append(x2d,x2drot)
        y2d = np.append(y2d,y2drot)

        x2dr = np.append(x2dr,x2drot_r)
        y2dr = np.append(y2dr,y2drot_r)
        
    return x,y,z,x2d,y2d,xr,yr,zr,x2dr,y2dr
