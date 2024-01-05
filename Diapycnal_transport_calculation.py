#!/usr/bin/env python
# coding: utf-8

"""Calculation of diapycnal mixing in MOM6"""


import xarray as xr
import numpy as np
import cosima_cookbook as cc
from dask.distributed import Client
import cftime
from datetime import timedelta
import sys
import matplotlib.path as mpath

if __name__ == '__main__':
    def transport_across_isopycnals_12months(expt, U, V, dvol):
        # this should be used for 1/10th
        resolution = expt.split('-')[1]
        if resolution == '01':
            U = U.isel(yh=slice(None, -1))
            dvol = dvol.isel(yh=slice(None, -1))
            if str(U.time[0].values)[:7] == '2003-01':
                U = U[1:, :]
                V = V[1:, :]
    
        D = 0*dvol 
        k = len(dvol.rho2_l)-1
        D[:, k, :] = (U.isel(xq=slice(None, -1), rho2_l=k).values -
                      U.isel(xq=slice(1, None), rho2_l=k).values +
                      V.isel(yq=slice(None, -1), rho2_l=k).values -
                      V.isel(yq=slice(1, None), rho2_l=k).values -
                      dvol.isel(rho2_l=k))
        for k in range(len(dvol.rho2_l)-2, -1, -1):
            D[:, k, :] = (
                U.isel(xq=slice(None, -1), rho2_l=k).values -
                U.isel(xq=slice(1, None), rho2_l=k).values +
                V.isel(yq=slice(None, -1), rho2_l=k).values -
                V.isel(yq=slice(1, None), rho2_l=k).values -
                dvol.isel(rho2_l=k) + D[:, k+1, :])
        D['time'] = U.time
        return D

    def transport_across_isopycnals_1month(expt, U, V, dvol):
        # this should be used for 1/20th, calculating all 
        # 12 months isn't feasable due to huge memory used
        resolution = expt.split('-')[1]

        D = 0*dvol 
        k = len(dvol.rho2_l)-1
        D[k, :] = (U.isel(xq=slice(None, -1), rho2_l=k).values -
                   U.isel(xq=slice(1, None), rho2_l=k).values +
                   V.isel(yq=slice(None, -1), rho2_l=k).values -
                   V.isel(yq=slice(1, None), rho2_l=k).values -
                   dvol.isel(rho2_l=k))
        for k in range(len(dvol.rho2_l)-2, -1, -1):
            D[k, :] = (
                U.isel(xq=slice(None, -1), rho2_l=k).values -
                U.isel(xq=slice(1, None), rho2_l=k).values +
                V.isel(yq=slice(None, -1), rho2_l=k).values -
                V.isel(yq=slice(1, None), rho2_l=k).values -
                dvol.isel(rho2_l=k) + D[k+1, :])
        D['time'] = U.time
        return D

    client = Client()
    client

    year = int(sys.argv[1])
    month = int(sys.argv[2])
    expt = sys.argv[3]
    expt_name = sys.argv[4]
    
    session = cc.database.create_session()
    frequency = '1 monthly'
    path_output = '/g/data/e14/cs6673/mom6_comparison/data_DSW/'
    resolution = expt_name.split('_')[1][:-3]

    DSW_region = {
        'name': ['Weddell', 'Prydz', 'Adelie', 'Ross'],
        'lon_min_area': [-58, 47, 90-360, 166-360],
        'lon_max_area': [-30, 72, 147-360, -170],
        'lat_min_area': [-75, -68, -67.5, -76.5],
        'lat_max_area': [-59, -64, -61.9, -65.8]}

    if resolution == '01':
        start_time = str(year) + '-01-01'
        end_time = str(year+1) + '-01-02'
    else:
        if month == 12:
            start_time = str(year) + '-' +  str(month).zfill(2) + '-01'
            end_time = str(year+1) + '-01-02'
        else:
            start_time = str(year) + '-' +  str(month).zfill(2) + '-01'
            end_time = str(year) + '-' +  str(month+1).zfill(2) + '-02'

    if resolution != '0025':
        # UMO and VMO
        U = cc.querying.getvar(
            expt, 'umo', session, frequency='1 monthly',
            start_time=start_time, end_time=end_time,
            chunks={'rho2_l': '200MB'}).sel(
            time=slice(start_time, end_time), yh=slice(None, -55)).squeeze()
        V = cc.querying.getvar(
            expt, 'vmo', session, frequency='1 monthly',
            start_time=start_time, end_time=end_time,
            chunks={'rho2_l': '200MB'}).sel(
            time=slice(start_time, end_time), yq=slice(None, -55)).squeeze()
        
        vol = cc.querying.getvar(
            expt, 'volcello', session,
            attrs={'cell_methods': 'area:sum rho2_l:sum yh:sum xh:sum time: point'},
            start_time=start_time, end_time=end_time,
            chunks={'rho2_l': '200MB'}).sel(
            time=slice(start_time, end_time), yh=slice(None, -55))
        # change in volume per second between monthly snapshots * density
        dvol =  vol.diff('time', label='lower')/(
            vol.time.diff('time', label='lower').astype('int')/1e9)*vol.rho2_l
        dvol = dvol.squeeze()
    
        if resolution == '01':
            D = transport_across_isopycnals_12months(expt, U, V, dvol)
        elif resolution == '005':
            D = transport_across_isopycnals_1month(expt, U, V, dvol)
    
        # save data
        D.name = 'diapycnal_transport'
        if resolution == '01':
            time_str = str(year)
            enc = {'diapycnal_transport':
                   {'chunksizes': (1, 50, 292, 1200),
                    'zlib': True, 'complevel': 5, 'shuffle': True}}
        elif resolution == '005':
            time_str = str(D.time.values)[:7]
            enc = {'diapycnal_transport':
                   {'chunksizes': (50, 292, 1200),
                    'zlib': True, 'complevel': 5, 'shuffle': True}}
        
        D.to_netcdf(path_output + 'Diapycnal_transport_at_upper_interface_' +
                    expt_name + '_' + frequency[:3:2] + '_' +
                    time_str + '.nc', encoding=enc)

    else:
        # for 1/40th cut out DSW regions, otherwise it takes forever
        for a, area_text in enumerate(DSW_region['name']):
            # UMO and VMO
            U = cc.querying.getvar(
                expt, 'umo', session, frequency='1 monthly',
                start_time=start_time, end_time=end_time,
                chunks={'rho2_l': '200MB'}).sel(
                time=slice(start_time, end_time), 
                xq=slice(DSW_region['lon_min_area'][a],
                         DSW_region['lon_max_area'][a]),
                yh=slice(DSW_region['lat_min_area'][a],
                         DSW_region['lat_max_area'][a])).squeeze()
            V = cc.querying.getvar(
                expt, 'vmo', session, frequency='1 monthly',
                start_time=start_time, end_time=end_time,
                chunks={'rho2_l': '200MB'}).sel(
                time=slice(start_time, end_time), 
                xh=slice(DSW_region['lon_min_area'][a],
                         DSW_region['lon_max_area'][a]),
                yq=slice(DSW_region['lat_min_area'][a],
                         DSW_region['lat_max_area'][a])).squeeze()
            
            vol = cc.querying.getvar(
                expt, 'volcello', session,
                attrs={'cell_methods': 'area:sum rho2_l:sum yh:sum xh:sum time: point'},
                start_time=start_time, end_time=end_time,
                chunks={'rho2_l': '200MB'}).sel(
                time=slice(start_time, end_time), 
                xh=slice(DSW_region['lon_min_area'][a],
                         DSW_region['lon_max_area'][a]),
                yh=slice(DSW_region['lat_min_area'][a],
                         DSW_region['lat_max_area'][a])).squeeze()
            # change in volume per second between monthly snapshots * density
            dvol =  vol.diff('time', label='lower')/(
                vol.time.diff('time', label='lower').astype('int')/1e9)*vol.rho2_l
            dvol = dvol.squeeze()
    
            # ensure correct length of dimensions
            if U.xq[0] > V.xh[0]:
                V = V.isel(xh=slice(1, None))
                dvol = dvol.isel(xh=slice(1, None))
            if U.xq[-1] < V.xh[-1]:
                V = V.isel(xh=slice(0, -1))
                dvol = dvol.isel(xh=slice(0, -1))
            if V.yq[0] > U.yh[0]:
                U = U.isel(yh=slice(1, None))
                dvol = dvol.isel(yh=slice(1, None))
            if V.yq[-1] < U.yh[-1]:
                U = U.isel(yh=slice(0, -1))
                dvol = dvol.isel(yh=slice(0, -1))
            assert len(U.xq) == (len(V.xh) + 1), 'longitude has wrong dimensions'
            assert len(U.xq) == (len(dvol.xh) + 1), 'longitude of volume has wrong dimensions'
            assert len(V.yq) == (len(U.yh) + 1), 'latitude has wrong dimensions'
            assert len(V.yq) == (len(dvol.yh) + 1), 'latitude of volume has wrong dimensions'
    
            D = transport_across_isopycnals_1month(expt, U, V, dvol)
            
            D.name = 'diapycnal_transport'
            time_str = str(D.time.values)[:7]
            enc = {'diapycnal_transport':
                   {'chunksizes': (99, 90, 200),
                    'zlib': True, 'complevel': 5, 'shuffle': True}}
            
            D.to_netcdf(path_output + 'Diapycnal_transport_at_upper_interface_in_' +
                        area_text + '_' + expt_name + '_' + frequency[:3:2] + '_' +
                        time_str + '.nc', encoding=enc)
