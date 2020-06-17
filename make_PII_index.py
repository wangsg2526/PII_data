#!/sw1/wangs/anaconda3.6_new/bin//python

""" by S. Wang """

import sys
import os.path
import scipy.signal
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
print (sys.version)
print(sys.executable)

import lanczos
import xarray as xr
import scipy.io
from eofs.standard import Eof

from scipy import signal
from numba import jit

import matplotlib;
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------

def filtering_zonal_r_wavenumber_domain_nomean(precip_eq, keep = 'W'):
    n2 = int(precip_eq.shape[0]/2); nx2 = int(precip_eq.shape[-1]/2)
    #print(n2, nx2)
    precip_fil = np.zeros(precip_eq.shape)
    for i in range(precip_eq.shape[1]):
        precip_f = np.fft.rfft2(precip_eq[:,i,:].squeeze().T)
        if i < 0:
            print(precip_f.shape)
        precip_f[0,:] = 0
        precip_f[:,0] = 0
        if keep == 'E':        
            precip_f[1:nx2,:] = 0;   # k*f <0
        elif keep == 'W':
            precip_f[nx2:,:] = 0;   # filter out westward waves, k*f ?
        precip_fil[:,i,:] = np.real(np.fft.irfft2(precip_f)).T
    return precip_fil

#------------------------------------------------------------------------------------------------

def calc_mjo_rt(xx, precip_filt_east, dtime):
        o_eof  = xx.eof.values.reshape([365,2,-1])
        tin = dtime
        precip_in = precip_filt_east
        mjo_recon = np.zeros(precip_in.shape)
        mjo_rt = np.zeros((precip_in.shape[0], 2))
        
        for it in np.arange(precip_in.shape[0]):
            a1 = precip_in[it,:,:].ravel()
            iday = tin[it].timetuple().tm_yday - 1 # start from 0
    
            if (tin[it].year % 4 == 0) and iday == 365 :
                a2 = 0.5*(o_eof[iday-1,0,:]+o_eof[0,0,:]).T
                mjo_rt[it, 0] = np.sum(a1*a2 )/np.sum(a2*a2)
    
                mjo_recon[it,:,:] = mjo_rt[it,0]*a2.reshape((17,144))
                
                a2 = 0.5*(o_eof[iday-1,1,:]+o_eof[0,1,:]).T
                mjo_rt[it, 1] = np.sum(a1*a2 )/np.sum(a2*a2)
    
                mjo_recon[it,:,:] += mjo_rt[it,1]*a2.reshape((17,144))
                
            else:
                a2 = o_eof[iday,0,:].T
                mjo_rt[it, 0] = np.sum(a1*a2 )/np.sum(a2*a2)
    
                a2 = o_eof[iday, 1,:].T
                mjo_rt[it, 1] = np.sum(a1*a2 )/np.sum(a2*a2)
                
                mjo_recon[it,:,:] = (mjo_rt[it,0]*o_eof[iday,0,:] + mjo_rt[it,1]*o_eof[iday,1,:]).reshape((17,144))
            
        mjo_recon = mjo_recon.reshape((-1,17,144))
        return mjo_rt, mjo_recon

#------------------------------------------------------------------------------------------------
import copy
import copy
def rotate_eof_pair(xxx0, latmax = 90, pair=0):
   
    xxx = copy.deepcopy(xxx0)
    ilat = np.argmin(np.abs(np.arange(0,360,2.5) - latmax))
                  
    nxy = 17*144
    i0 = pair*2
    i1 = pair*2+1
    eof1 = xxx.eof.values[:,i0,0:nxy].reshape(365,17,144)
    eof2 = xxx.eof.values[:,i1,0:nxy].reshape(365,17,144)
    for i in range(365):
        eof1[i,:,:] /= np.sqrt(np.sum(eof1[i,:,:]**2))
        eof2[i,:,:] /= np.sqrt(np.sum(eof2[i,:,:]**2))
    theta_range = np.arange(-1,1.01, 0.01)*np.pi
    theta_range = np.arange(-180,180)/180*np.pi

    eof1_n = np.zeros(eof1.shape)
    eof2_n = np.zeros(eof1.shape)

    for i in range(365):

        Ntest = len(theta_range)
        eof1_n_test = np.zeros((Ntest,17,144))
        eof2_n_test = np.zeros((Ntest,17,144))

        for j,theta in enumerate(theta_range):
            #theta = 0.5*np.pi    
            eof1_n_test[j,:,:] = eof1[i,:,:]*np.cos(theta) - eof2[i,:,:]*np.sin(theta)
            eof2_n_test[j,:,:] = eof1[i,:,:]*np.sin(theta) + eof2[i,:,:]*np.cos(theta)
        #isel = np.argmin(eof1_n_test.mean(1)[:,40:45].mean(-1))
        isel = np.argmax(eof2_n_test.mean(1)[:,ilat]) # 36 -> 90E
        eof1_n[i,:,:] = eof1_n_test[isel,:,:]
        eof2_n[i,:,:] = eof2_n_test[isel,:,:]

    xxx.eof.values[:,i0:i1+1,:] = np.hstack((eof1_n[:,None,...], eof2_n[:,None,...])).reshape(365,2,17*144)

    return xxx

#------------------------------------------------------------------------------------------------
def xcorr(x,y,maxlags):
    #x= precip_scs
    #y = precip[:,10,60]
     
    N = maxlags
    cor = np.zeros((2*N+1))
    cor[N] = np.corrcoef(x, y)[0,1]
    pval = np.zeros((2*N+1))
    
    for i in np.arange(1,N+1):
        #print(x[:-i].shape, y[i:].shape)
        cor[N+i] = np.corrcoef(x[:-i], y[i:])[0,1]
        cor[N-i] = np.corrcoef(x[i:], y[:-i])[0,1]
    return cor

def calc_corr_mjo_rt(mjo_rt, dtime) :
    cor_ = xcorr(mjo_rt[:,0], mjo_rt[:,1], 35)
    dtime_m = np.array([d.month for d in dtime])
    mjo_rt_sum = mjo_rt[np.where((dtime_m>=5)&(dtime_m<=10))[0],:]
    cor_sum_ = xcorr(mjo_rt_sum[:,0], mjo_rt_sum[:,1], 35)
    mjo_rt_win = mjo_rt[np.where((dtime_m<=3)|(dtime_m>=12))[0],:]
    cor_win_ = xcorr(mjo_rt_win[:,0], mjo_rt_win[:,1], 35)
    return cor_, cor_sum_, cor_win_
#------------------------------------------------------------------------------------------------

if __name__ == '__main__':


    from trmm_util import load_trmm2deg_and_filter
    # tfile = '/sw21/wangs/trmmdata/2013later/all/trmm_1degree/trmm2p5_/trmm2p5.mat'
    tfile = './trmm2p5.mat'
    precip_filt, precip_noac, precip_clim, precip_obs, lons, lats, dtime = load_trmm2deg_and_filter(ofile=tfile, lon_bnd=[0, 361], lat_bnd=[-20,20], bpfilt=[20,96])
    
    #xxx = xr.open_dataset('./tttt_trmm2p5_eof_normalized_crc_sign.nc', autoclose=True)
    xxx = xr.open_dataset('./PII_EOFs.nc', autoclose=True)
    print(xxx)

    nx = 144
    ny = 17
    nxy = nx*ny
    print(nx,ny, nxy)
    
    llon = np.arange(0,359.5,2.5)
    llat = np.arange(-20,21,2.5)
    
    if np.mod(precip_filt.shape[0], 2) == 1: 
        precip_filt_east = np.zeros(precip_filt.shape)
        precip_filt_east[:-1,:,:] = filtering_zonal_r_wavenumber_domain_nomean(precip_filt[:-1,:,:], keep='E')
    else:
        precip_filt_east = filtering_zonal_r_wavenumber_domain_nomean(precip_filt, keep='E')
 

    pii_pc, mjo_recon = calc_mjo_rt(xxx, precip_filt_east, dtime)
    pii_ind, mjo_recon = calc_mjo_rt(xxx, precip_filt, dtime)
    
    if 1 == 1:
        #Plot the DYNAMO case

        plt.figure(figsize=(6.5,9))
        
        fac = np.std(pii_ind[:,0]), np.std(pii_ind[:,1])
        
        idate = [datetime(2011,10,1,0,0,0), datetime(2012,1,1,0,0,0)]
        
        for i in range(2):
            plt.subplot(3,2,i+1)
            i0 = np.where(dtime == np.datetime64(idate[i]))[0][0]
        
            nn = 31
            plt.plot(pii_ind[i0:i0+nn+1,0]/fac[0], pii_ind[i0:i0+nn+1,1]/fac[1], '-')
            plt.plot(pii_ind[i0+nn:i0+2*nn+1,0]/fac[0], pii_ind[i0+nn:i0+2*nn+1,1]/fac[1], '-')
            plt.plot(pii_ind[i0+2*nn:i0+3*nn,0]/fac[0], pii_ind[i0+2*nn:i0+3*nn,1]/fac[1], '-')
            plt.plot(pii_ind[i0:i0+90:15,0]/fac[0], pii_ind[i0:i0+90:15,1]/fac[1], 's')
        
            plt.axis(np.array([-3,3,-3,3]))
            plt.plot(np.cos(np.linspace(0,2*np.pi,100)), np.sin(np.linspace(0,2*np.pi,100)), '--', color='k', linewidth=0.2, dashes=(5, 1))
            
        cor_1, cor_sum_1, cor_win_1 = calc_corr_mjo_rt(pii_ind, dtime)
        print('pii_ind=', cor_1[35], cor_sum_1[35], cor_win_1[35]) 

        plt.subplot(3,2,3)
        plt.plot(np.arange(-35,36), cor_1, 'k',label='All year')
        plt.plot(np.arange(-35,36), cor_win_1, label='Win(12-3)')
        plt.plot(np.arange(-35,36), cor_sum_1, '--', label='Sum(5-10)')
        plt.xlim([-15,15])
        plt.grid()
        plt.legend()
        
        plt.tight_layout()
        
        plt.savefig('PII_DYNAMO_test.jpg', dpi=300)
