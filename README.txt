This repository contains data and python code used to produce PII: precipitation index for tropical intraseasonal index (MJO/BSISO). 

ncdump -h PII_EOFs.nc:

netcdf PII_EOFs {
dimensions:
        lat = 17 ;
        lon = 144 ;
        doy = 365 ;
        num = 2 ;
        npc = 20 ;
        time = 7548 ;
variables:
        double lat(lat) ;
                lat:_FillValue = NaN ;
        double lon(lon) ;
                lon:_FillValue = NaN ;
        double eof(doy, num, lat, lon) ;
                eof:_FillValue = NaN ;
        double eofvar(doy, npc) ;
                eofvar:_FillValue = NaN ;
        int64 time(time) ;
                time:units = "days since 1998-01-01 00:00:00" ;
                time:calendar = "proleptic_gregorian" ;
        double pc(time, num) ;
                pc:_FillValue = NaN ;
        double mjo_ind(time, num) ;
                mjo_ind:_FillValue = NaN ;
        double mjo_rt(time, num) ;
                mjo_rt:_FillValue = NaN ;
}

Variables: 

lon: longitude, 0, 2.5, 5.0, 7.5, ..., 357.0
lat: latitude, -20, -17.5, ..., -2.5, 0, 2.5, 17.5, 20.0
eof: The EOF spatial patterns (365). For the leap year, need interpolation. 
mjo_ind: PII index from the TRMM Multi-satellite Preciptiation Analysis. 
mjo_rt: Realtime PII Index

