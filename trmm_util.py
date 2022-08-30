#!/sw1/wangs/anaconda3.6_new/bin//python

##!/sw1/wangs/anaconda3.6_new/bin/python
import sys
import os.path
import scipy.signal
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
print (sys.version)
import xarray as xr
import time
from eofs.standard import Eof
import lanczos
import scipy.io

# ----------------------------------------------------------------------------------------------------------------------------------------------

# a function to remove the climatology
def cal_anamoly(var_in, dtime):
    #var_in = cpc_sca_rain_all
    #dtime = cpc_time_all
    #nt, nlon = var_in.shape
    nt = var_in.shape[0]
    print(var_in.shape)
    
    if type(dtime[0]).__name__ == 'datetime64' :
        ns = 1e-9 # number of seconds in a nanosecond
        dtime = [datetime.utcfromtimestamp(t.astype(int) * ns) for t in dtime]
    
    dayofyear = np.zeros(nt)
    #var_clim = np.zeros((366, var_in_shape[1], ))
    if var_in.ndim > 1:
        var_clim = np.zeros(np.hstack( (366, np.array(var_in.shape[1:])) ) )
    else:
        var_clim = np.zeros((366,))
    ic_c = np.zeros((366))
    
    for i in range(nt):
        tmon = dtime[i].month
        tday = dtime[i].day
        if not (tmon==2 and tday==29):
            dayofyear[i] = (datetime(1981,tmon,tday) - datetime(1981,1,1)).days + 1
        
        j = int(dayofyear[i])
        #print(j)
        if j > 0:
            var_clim[j,...] = var_clim[j,...] + var_in[i,...]
            ic_c[j] = ic_c[j] + 1
    
    for i in range(1,366):
        var_clim[i,...] = var_clim[i,...]/ic_c[i]
    
    var_ana = np.zeros(var_in.shape)
    
    for i in range(nt):
        tmon = dtime[i].month
        tday = dtime[i].day
        if not (tmon==2 and tday==29):      
            dayofyear[i] = (datetime(1981,tmon,tday) - datetime(1981,1,1)).days+1
        j = int(dayofyear[i])
        
        if j > 0:
            var_ana[i,...] = var_in[i,...] - var_clim[j,...]
        else:  # for Feb 29 in leap years
            j1 = int(dayofyear[i-1])
            j2 = int(dayofyear[i+1])
            var_ana[i,...] = var_in[i,...] - (var_clim[j1,...]+var_clim[j2,...])*0.5
            
    # substract previous 120 d mean
    var_ana_window = np.zeros(var_ana.shape)
    for it in np.arange(var_ana.shape[0]-1,120,-1):
        var_ana_window[it,...] = var_ana[it,...] - var_ana[it-120:it,...].mean(0)
                    
    return var_ana, var_clim,  var_ana_window                                 

# ----------------------------------------------------------------------------------------------------------------------------------------------



#ns = 1e-9 # number of seconds in a nanosecond
#dtime_m = np.array([t.month for t in dtime])
#it_sel = np.where((dtime_m >=5)&(dtime_m<=10))[0]


from scipy import signal
b, a = signal.butter(31, [ 1/90*2, 1/20*2], btype='bandpass')
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
    
def bandpass_kaiser(ntaps, lowcut, highcut, fs, width):
    nyq = 0.5 * fs
    atten = kaiser_atten(ntaps, width / nyq)
    beta = kaiser_beta(atten)
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=('kaiser', beta), scale=False)
    return taps



# ----------------------------------------------------------------------------------------------------------------------------------------------

from numba import jit
@jit(nogil=True)
def runningMeanFast_conv(x, N):
    N2 = np.int_((N-1)/2)
    out = np.convolve(x, np.ones((N,))/N,mode='valid')
    padbeg = np.zeros((N2,))
    padend = np.zeros((N2,))

    #print(out.shape)
    padbeg[0] = x[0]
    for i in np.arange(1,N2):
        padbeg[i] = x[0:2*i+1].mean()

    xrev = x[::-1]
    padend[0] = xrev[0]
    for i in np.arange(1,N2):
        padend[i] = xrev[:2*i+1].mean()
    padend = padend[::-1]

    #out = np.concatenate(  ( np.zeros((N+1,)), out, np.zeros((N-3,)) )  )
    out = np.concatenate(  ( padbeg, out, padend )  )

    #print(out.shape)
    return out

import copy
def cal_window_and_smooth(precip_noac, N=120, Ns = 9):
    precip_noac_window = np.zeros(precip_noac.shape)
    #for it in np.arange(precip_noac.shape[0]-1,120,-1):
    for it in np.arange(N, precip_noac.shape[0]):
            precip_noac_window[it,...] = precip_noac[it,...] - precip_noac[it-N:it+1,...].mean(0)
            
    precip_noac_window_avg = np.zeros(precip_noac.shape)
    if Ns >= 3:
        for ii in np.arange(precip_noac_window.shape[1]):
            for jj in np.arange(precip_noac_window.shape[2]):
                precip_noac_window_avg[:, ii, jj]  = runningMeanFast_conv(precip_noac_window[:, ii, jj], Ns)
    else:
        precip_noac_window_avg  = copy.deepcopy(precip_noac_window)

            
    return precip_noac_window_avg

def filtering_zonal_r_wavenumber_domain(precip_eq, keep = 'W', keepMean=True):
    n2 = int(precip_eq.shape[0]/2); nx2 = int(precip_eq.shape[-1]/2)
    #print(n2, nx2)
    precip_fil = np.zeros(precip_eq.shape)
    for i in range(precip_eq.shape[1]):
        precip_f = np.fft.rfft2(precip_eq[:,i,:].squeeze().T)
        if i < 0:
            print(precip_f.shape)
        if not keepMean:
            precip_f[0,:] = 0
            precip_f[:,0] = 0
        if keep == 'E':        
            precip_f[1:nx2,:] = 0;   # k*f <0
        elif keep == 'W':
            precip_f[nx2:,:] = 0;   # filter out westward waves, k*f ?
        precip_fil[:,i,:] = np.real(np.fft.irfft2(precip_f)).T
    return precip_fil

def filtering_zonal(precip_eq, keep = 'W'):
    n2 = int(precip_eq.shape[0]/2); nx2 = int(precip_eq.shape[-1]/2)
    print(n2, nx2)
    precip_fil = np.zeros(precip_eq.shape)
    for i in range(precip_eq.shape[1]):
        precip_f = np.fft.fft2(precip_eq[:,i,:].squeeze())
        #precip_f = scipy.fftpack.fft2(precip_eq_ana[:,i,:].squeeze())
        precip_f[0,:] = 0
        precip_f[:,0] = 0
        if keep == 'W':        
            precip_f[:n2,nx2:] = 0; precip_f[n2:,:nx2] = 0  # k*f <0
        elif keep == 'E':
            precip_f[:n2,:nx2] = 0; precip_f[n2:,nx2:] = 0 # filter out westward waves, k*f ?
        #precip_f[:n2,] = 0
        precip_fil[:,i,:] = np.real(np.fft.ifft2(precip_f))
        #precip_eq2[:,i,:] = np.real(scipy.fftpack.fft2(precip_f))
    return precip_fil

# ----------------------------------------------------------------------------------------------------------------------------------------------

def load_trmm2deg_and_filter(ofile='/home/wangs/s2s/subx/gpcp1d_2p5.nc', lat_bnd=[-30,30], lon_bnd=[0,361], bpfilt=[30,96]):

    tmat = scipy.io.loadmat(ofile)
    lats = tmat['lat'][0,:]
    lons = tmat['lon'][0,:]
    tord = tmat['time'][0,:]-366
    otime = [datetime.fromordinal(to) for to in tord]
    #[datetime.fromordinal(tmat['time'][0,i]) for i in range(tmat['time'].shape[1])]
    precip = tmat['rain2p5']
    if 1 == 11:
        fout = Dataset(ofile,'r',mmap=False)
        lats = fout.variables['lat'][:].squeeze()
        lons = fout.variables['lon'][:].squeeze()
        time = fout.variables['time'][:].squeeze()
        precip = fout.variables[var][:].squeeze()
        fout.close()

    print(otime[0], otime[-1])
    ilat_sel = (lats>=lat_bnd[0]) & (lats<=lat_bnd[1]) 
    lats = lats[ilat_sel]
    precip = precip[:,ilat_sel,:]

    ilon_sel = (lons>=lon_bnd[0]) & (lons<=lon_bnd[1]) 
    lons = lons[ilon_sel]
    precip = precip[:,:, ilon_sel]
    
    nt, nlat, nlon = precip.shape
    print(nt, nlat, nlon)
    
    otime_ord = np.array([tt.toordinal() for tt in otime])
    tsel = np.where( (otime_ord >= datetime(1996,1,1).toordinal()) & (otime_ord <= datetime(2018, 12,31).toordinal()))[0]
    precip = precip[tsel,:,:]
    otime_ord = otime_ord[tsel]
    precip.shape, otime_ord.shape

    otime = [otime[ii] for ii in tsel]
    otime[0], otime[-1]

    # remove annual cycles
    precip_clim = np.zeros((365, nlat, nlon))
    precip_clim_leap = np.zeros((366, nlat, nlon))

    ic = 0
    for iy in np.arange(1998, 2018):
        cc = np.where( (otime_ord >= datetime(iy,1,1).toordinal()) &  (otime_ord <= datetime(iy,12,31).toordinal()) ) [0]
        if len(cc) == 365:
            precip_clim = precip_clim + precip[cc,:,:]
            ic += 1
        elif len(cc) == 366:
            ci = np.where(otime_ord[cc] != datetime(iy,2,29).toordinal())[:]
            cc = cc[ci]
            precip_clim = precip_clim + precip[cc,:,:]
            ic += 1
    precip_clim = precip_clim/ic           
    precip_clim.shape

    precip_clim_filt  = np.zeros(precip_clim.shape)
    ii = 75; jj = 13
    for ii in np.arange(nlon):
        for jj in np.arange(nlat):
            otmp = precip_clim[:,jj,ii]
            fft_coef = np.fft.fft(otmp)
            fft_coef[4:365-3] = 0.0 # remove mean and first 3 harmonic components
            otmp = np.real(np.fft.ifft(fft_coef))
            precip_clim_filt[:,jj,ii] = otmp

    precip_clim_leap[:60,:,:] = precip_clim_filt[:60,:,:]
    precip_clim_leap[61:,:,:] = precip_clim_filt[60:,:,:]
    precip_clim_leap[60,:,:] = (precip_clim_filt[59,:,:] + precip_clim_filt[60,:,:])*0.5

    precip_noac = np.zeros(precip.shape)
    for iy in np.arange(1998, 2019):
        cc = np.where( (otime_ord >= datetime(iy,1,1).toordinal()) &  (otime_ord <= datetime(iy,12,31).toordinal()) ) [0]
        cc1 = otime_ord[cc] - datetime(iy,1,1).toordinal()
        if np.mod(iy,4) != 0:
        #if len(cc) <= 365 : 
            precip_noac[cc,:,:] = precip[cc,:,:] - precip_clim_filt[cc1,:,:]
        else :
        #elif len(cc) == 366 :
            precip_noac[cc,:,:] = precip[cc,:,:] - precip_clim_leap[cc1,:,:]
    print('precip_noac.shape=', precip_noac.shape)
    
    # Now we do filtering not
    fs=1
    taps_kaiser = bandpass_kaiser(301, 1.0/bpfilt[1], 1.0/bpfilt[0], fs=1, width=0.02);
    #taps_kaiser = bandpass_kaiser(151, 1.0/bpfilt[1], 1.0/bpfilt[0], fs=1, width=0.02);
    #taps_kaiser = bandpass_kaiser(301, 1.0/70, 1.0/20.0, fs, width=0.02);
    if 1 == 11:
        w, h = signal.freqz(taps_kaiser, 1)
        #plt.plot(w, 20 * np.log10(abs(h)))    
        plt.plot((fs / np.pi) * w, abs(h))
        plt.grid(True)
        plt.xlim([0,0.5])


    lanczos139 = lanczos.Lanczos('bp', nwts=139, pca=bpfilt[0], pcb=bpfilt[1], delta_t=1)

    print('precip:', precip.shape)
    precip_obs_filt  = np.zeros(precip.shape)
    #ii = 75; jj = 13
    for ii in np.arange(nlon):
        for jj in np.arange(nlat):
            #otmp = signal.filtfilt(lanczos139.wgt,1, precip_noac[:, jj, ii])    
            #otmp = signal.filtfilt(taps_kaiser,1, precip_noac[:, jj, ii])    
            #otmp = signal.filtfilt(taps_kaiser,1, precip_noac[:, jj, ii])            
            otmp = signal.filtfilt(lanczos139.wgt,1, precip_noac[:, jj, ii])            
            precip_obs_filt[:,jj,ii] = otmp

    #scipy.io.savemat('precip_filt.mat', {'time':otime_ord, 'lat':lats, 'lon':lons, 'precip':np.float32(precip_obs_filt)})   
    
    return precip_obs_filt, precip_noac, precip_clim, precip, lons, lats, otime


# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    print(sys.argv)
    if len(sys.argv)>1:
         winsize = float(sys.argv[1])
    else:
         winsize = 60
    print('winsize=', winsize, type(winsize))



    tfile = '/sw21/wangs/trmmdata/2013later/all/trmm_1degree/trmm2p5_/trmm2p5.mat'
    tfile = '/gpfs/fs1/work/shuguang/eofs/trmm2p5_hq.mat'
    tfile = '/sw21/wangs/trmmdata/2013later/all/trmm_1degree/trmm2p5_/HQ/trmm2p5_hq.mat'

    precip_filt, precip_noac, precip_clim, precip_obs, lons, lats, dtime = load_trmm2deg_and_filter(ofile=tfile,
                                                    lon_bnd=[0, 361], lat_bnd=[-20,20], bpfilt=[20,90])
    
    dtime_m = np.array([t.month for t in dtime])
    #it_sel = np.where((dtime_m >=5)&(dtime_m<=10))[0]
    
    nt,nlat,nlon = precip_filt.shape
    print(nt,nlat,nlon)
    
    #fac8 = np.std(precip_filt, axis=0).mean()
    
    if 1 == 1:
        if np.mod(precip_filt.shape[0],2) == 0:
           precip_filt_east = filtering_zonal_r_wavenumber_domain(precip_filt, keep='E',  keepMean=False)
        else:
           precip_filt_east = np.zeros(precip_filt.shape)
           precip_filt_east[:-1,:,:] = filtering_zonal_r_wavenumber_domain(precip_filt[:-1,:,:], keep='E',  keepMean=False)
    else:
        precip_filt_east = precip_filt
    
    print('precip_filt_east:', precip_filt_east.shape)
    
    #ds = xr.DataArray(precip_filt_east, dims=['time', 'lat', 'lon'], coords={'time':dtime, 'lon':np.arange(0,360,2), 'lat':np.arange(-20,20.1,2)}, \
    ds = xr.DataArray(precip_filt_east, dims=['time', 'lat', 'lon'], coords={'time':dtime, 'lon':np.arange(0,359,2.5), 'lat':np.arange(-20,20.1,2.5)}, \
                       name='rain_filt').to_dataset()
    #ds.to_netcdf('gpcp_2p5_filt_2096.nc')
    
    precip_filt_allyear_east = precip_filt_east.reshape(-1,nlat*nlon)
    otime_64 = np.array([np.datetime64(t) for t in np.array(dtime)[:-1]])
    
    
    print( nlon, nlat, nlon*nlat, precip_filt_allyear_east.shape )
    
    iyear = 1980
    idoy = 10
    
    nt = precip_filt_allyear_east.shape[0]
    nxy = nlon*nlat
    eofvar_all = np.zeros((365,20))
    eoferr_all = np.zeros((365,20))
    eof_all = np.zeros((365,6, nxy))
    pc_all = np.zeros((nt,6))
    xx = np.arange(nt)
   
    ifirst_flag = True 
    for iidoy in np.int_(np.hstack((np.arange(300,365), np.arange(0,300)))):
        idoy = int(iidoy)
    #for idoy in [0,180,364]:
        t1 = time.time()
        print(idoy)
        ds_list = []
        #ctmp = np.array([], dtype=np.int64).reshape(0,4896)
        ctmp = np.array([], dtype=np.int64).reshape(0,nxy)
        ctmp1 = np.zeros(precip_filt_allyear_east.shape)
        wgt = np.zeros((precip_filt_allyear_east.shape[0],))
        iid_set = np.array([], dtype=np.int64).reshape(0,)
        i0_set = []; #np.array([], dtype=np.int64).reshape(0,)
        dt0_set = [] #np.array([], dtype=np.int64).reshape(0,)
     
        for iyear in range(1999,2015):
            #if iyear == yskip:
            #    continue

            dt0 = datetime(iyear,1,1) + timedelta(idoy)
            dt0 = datetime(1981,1,1) + timedelta(idoy)
            dt0 = datetime(iyear, dt0.month, dt0.day)
          
            if not dt0 in otime_64 :
                print('Warning: ', dt0, 'NOT found')
                continue
            #dt1, dt2 = dt0-timedelta(30), dt0+timedelta(30)
            dt1, dt2 = dt0-timedelta(60), dt0+timedelta(60)
            #ds_list.append(ds.precip_filt.sel(time=slice(dt1, dt2)))
            iid = np.where((otime_64>=dt1)&(otime_64<=dt2))[0]
            iid_set = np.hstack((iid_set,iid))
            iid1 = np.where((otime_64<dt1)|(otime_64>dt2))[0]
            #print(dt0, dt1, dt2, )
            #ds_list.append(precip_filt_allyear_east[iid,...])    
            ctmp = np.vstack((ctmp, precip_filt_allyear_east[iid,...]))
            ctmp1[iid,...] = precip_filt_allyear_east[iid,...] 
            i0 = np.where(otime_64==dt0)[0][0]
            dt0_set.append(dt0)
            i0_set.append(i0)
            #winsize = 70 
            #if idoy <= 180:
            #    winsize = 70 + 50*np.exp(-((idoy+365-340)/20.0)**2)
            #else:
            #    winsize = 70 + 50*np.exp(-((idoy-340)/20.0)**2)
            if iyear == 2000:
                print('winsize=', winsize)
            wgt += np.exp(-((xx-i0)/winsize)**4)
    
        if 1 == 1:
            #dss = xr.merge(ds_list)        
            #solver = Eof(dss.precip_filt.values)
            isel_nonzero = np.where(wgt>=1e-3)[0]
            otmp = wgt[:,np.newaxis]*precip_filt_allyear_east
            print('svd ...', otmp[isel_nonzero, ...].shape)
            solver = Eof(otmp[isel_nonzero, ...])
            #solver = Eof(ctmp1)
            errors = solver.northTest(neigs=20, vfscaled=True)
            eof1 = solver.eofsAsCovariance(neofs=10)
            eofvar = solver.varianceFraction()
            print('eofvar:', eofvar[:4])
     
            eofvar_all[idoy, :] = eofvar[:20]
            eoferr_all[idoy, :] = errors[:20]
            eof_all[idoy, :,:] = eof1[:6,:]
    
            if ifirst_flag:
                 sign = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                 ifirst_flag = False
            else: #if idoy > 300:
                for jj in range(6):
                    id1 = np.mod(idoy-1, 365)
                    rr = np.corrcoef(eof_all[idoy, jj,:].ravel(), eof_all[id1, jj,:].ravel())[0,1]
                    sign[jj] = np.sign(rr)
                    eof_all[idoy, jj,:] *= sign[jj]
                print( sign)
    
            #pc = solver.pcs(npcs=10)
            pc = np.zeros((nt,6))
            pc[isel_nonzero,...] = solver.pcs(npcs=6)
            ima = np.in1d(otime_64, dt0_set)
            isel = np.nonzero(ima)[0]
            t_set = otime_64[np.array(iid_set)]
            ima = np.in1d(t_set, dt0_set)
            #isel1 = np.nonzero(ima)[0]
            for jj in range(6):
                pc_all[isel,jj] = sign[jj]*pc[isel,jj]
    
        if np.mod(idoy, 30) == 0:
            da_pc = xr.DataArray(pc_all, dims=('time', 'num'), name='pc', coords={'time':dtime})
            da_eof = xr.DataArray(eof_all, dims=('doy', 'num', 'xy'), name='eof')
            da_eofvar = xr.DataArray(eofvar_all, dims=('doy', 'npc'), name='eofvar')
            da_eoferr = xr.DataArray(eoferr_all, dims=('doy', 'npc'), name='error')
            dsout = xr.merge((da_eof , da_eofvar, da_pc, da_eoferr), compat='no_conflicts',)
            dsout.to_netcdf('test_trmm2p5_eof.nc')
        t2 = time.time()
        print('Elapsed time: ', t2-t1)
    
    eofvar.shape, eof1.shape, ctmp.shape
    
    
    da_pc = xr.DataArray(pc_all, dims=('time', 'num'), name='pc', coords={'time':dtime})
    da_eof = xr.DataArray(eof_all, dims=('doy', 'num', 'xy'), name='eof')
    da_eofvar = xr.DataArray(eofvar_all, dims=('doy', 'npc'), name='eofvar')
    da_eoferr = xr.DataArray(eoferr_all, dims=('doy', 'npc'), name='error')
    dsout = xr.merge((da_eof , da_eofvar, da_pc, da_eoferr), compat='no_conflicts',)
    dsout.to_netcdf('test_trmm2p5_eof.nc', mode='a')
    dsout
    
