#!/usr/bin/env python
# coding: utf-8

"""Calculation of daily (monthly for density) time series of surface and
    bottom properties integrated over the continental shelf in MOM6"""


import xarray as xr
import numpy as np
import cosima_cookbook as cc
from dask.distributed import Client
import cftime
from datetime import timedelta
import sys
import matplotlib.path as mpath

if __name__ == '__main__':
    def shelf_mask_isobath(var, expt, output_mask=False):
        '''
        Mask varibales from MOM6 at 1/10th by the region polewards
        of the 1000m isobath
        '''
        
        if expt == 'panan_005deg_jra55_ryf':
            contour_file = xr.open_dataset(
                '/home/142/cs6673/work/mom6_comparison/Python_scripts/' +
                'Antarctic_slope_contour_1000m_MOM6_005deg.nc')
            shelf_mask = contour_file['contour_masked_above'].data
            yh = contour_file['yh'].data
            xh = contour_file['xh'].data
        else:
            isobath_depth = 1000
            outfile = ('/g/data/v45/akm157/model_data/access-om2/Antarctic_slope_contour_' +
                       str(isobath_depth) + 'm.npz')
            contour_file = np.load(outfile)

            shelf_mask = contour_file['contour_masked_above']
            yh = contour_file['yt_ocean']
            xh = contour_file['xt_ocean']

        # in this file the points along the isobath are given a positive value,
        # the points outside (northwards) of the isobath are given a value of
        # -100 and all the points on the continental shelf have a value of 0 
        # so we mask for the 0 values 
        shelf_mask[np.where(shelf_mask!=0)] = np.nan
        shelf_mask = shelf_mask+1
        shelf_map = np.nan_to_num(shelf_mask)
        shelf_mask = xr.DataArray(
            shelf_mask, coords = [('yh', yh), ('xh', xh)])
        shelf_map = xr.DataArray(
            shelf_map, coords = [('yh', yh), ('xh', xh)])

        # then we want to multiply the variable with the mask so we need to account
        # for the shape of the mask. The mask uses a northern cutoff of 59S.
        masked_var = var.sel(yh = slice(-90, -59.03)) * shelf_mask

        if output_mask == True:
            return masked_var, shelf_map
        else:
            return masked_var

    def mask_from_polygon(lon, lat, xh, yh):
        polygon = [(lon[0], lat[0])]
        for l in range(1, len(lon)):
            polygon += [(lon[l], lat[l])]
        poly_path = mpath.Path(polygon)

        x, y = xr.broadcast(xh, yh)
        coors = np.hstack((x.values.reshape(-1, 1), y.values.reshape(-1, 1)))

        mask = poly_path.contains_points(coors)
        mask = mask.reshape(xh.size, yh.size).transpose()
        mask = xr.DataArray(
            mask, dims=['yh', 'xh'], coords={'xh': xh, 'yh': yh})
        return mask
    
    def select_bottom_values(var, land_mask):
        depth_array = var*0 + var.z_l
        max_depth = depth_array.max(dim='z_l', skipna=True)

        var_b = var.where(depth_array.z_l >= max_depth)
        var_b = var_b.sum(dim='z_l').compute()
        var_b = var_b.where(land_mask == 0)
        return var_b

    client = Client()
    client

    year = int(sys.argv[1])
    expt = sys.argv[2]
    
    start_time = str(year) + '-01-01'
    if expt == 'panan_005deg_jra55_ryf':
        end_time = str(year) + '-12-31'
    else:
        end_time = str(year+9) + '-12-31'
    
    if expt == 'panan_005deg_jra55_ryf':
        session = cc.database.create_session(
            '/g/data/e14/cs6673/mom6_comparison/Python_scripts/panan_005deg_jra55_ryf.db')
    else:
        session = cc.database.create_session()
    path_output_SWMT = '/g/data/e14/cs6673/mom6_comparison/data_DSW/'
    
    """Calculate timeseries"""
    depth = cc.querying.getvar(
        expt, 'deptho', session, n=1,
        chunks={'xh': '200MB', 'yh': '200MB'})
    depth = depth.sel(yh=slice(None, -59))
    land_mask = (depth*0).fillna(1)
    
    area = cc.querying.getvar(
        expt=expt, variable='areacello', session=session,
        frequency='static', n=1, chunks={'xh': '200MB', 'yh': '200MB'})
    area = area.sel(yh=slice(None, -59))
    area_shelf = shelf_mask_isobath(area, expt)
    # area has values one land which need to be removed
    area_shelf = area_shelf.where(land_mask == 0)
    area_shelf = area_shelf/area_shelf.sum(['xh', 'yh'])
    
    # area for Weddell Sea
    lon = [-63, -63, -24, -24, -63]
    lat = [-79, -71, -71, -79, -79]
    mask_W = mask_from_polygon(lon, lat, area.xh, area.yh)
    mask_W = mask_W.where(mask_W == True, 0)
    mask_W = mask_W.where(mask_W != 0)
    # area has values one land which need to be removed
    # before multiplying with the mask
    area_W = area_shelf.where(land_mask == 0) * mask_W
    area_W = area_W/area_W.sum(['xh', 'yh'])
    
    # area for Ross Sea
    lon = [-200, -200, -150, -150, -200]
    lat = [-79, -71, -71, -79, -79]
    mask_R = mask_from_polygon(lon, lat, area.xh, area.yh)
    mask_R = mask_R.where(mask_R == True, 0)
    mask_R = mask_R.where(mask_R != 0)
    # area has values one land which need to be removed
    # before multiplying with the mask
    area_R = area_shelf.where(land_mask == 0) * mask_R
    area_R = area_R/area_R.sum(['xh', 'yh'])

    SST = cc.querying.getvar(
        expt, 'tos', session, frequency='1 daily',
        start_time=start_time, end_time=end_time,
        chunks={'xh': '200MB', 'yh': '200MB'})
    SST = SST.sel(yh=slice(None, -59), time=slice(start_time, end_time))
    SST_shelf = shelf_mask_isobath(SST, expt)
    # SST_shelf_mean = (SST_shelf*area_shelf).sum(['xh', 'yh']).compute()
    SST_W_mean = (SST*area_W).sum(['xh', 'yh']).compute()
    SST_R_mean = (SST*area_R).sum(['xh', 'yh']).compute()
    
    T_bot = cc.querying.getvar(
        expt, 'tob', session, frequency='1 daily',
        start_time=start_time, end_time=end_time,
        chunks={'xh': '200MB', 'yh': '200MB'})
    T_bot = T_bot.sel(yh=slice(None, -59), time=slice(start_time, end_time))
    T_bot_shelf = shelf_mask_isobath(T_bot, expt)
    # T_bot_shelf_mean = (T_bot_shelf*area_shelf).sum(['xh', 'yh']).compute()
    T_bot_W_mean = (T_bot*area_W).sum(['xh', 'yh']).compute()
    T_bot_R_mean = (T_bot*area_R).sum(['xh', 'yh']).compute()
    print('calculated T')

    SSS = cc.querying.getvar(
        expt, 'sos', session, frequency='1 daily',
        start_time=start_time, end_time=end_time,
        chunks={'xh': '200MB', 'yh': '200MB'})
    SSS = SSS.sel(yh=slice(None, -59), time=slice(start_time, end_time))
    SSS_shelf = shelf_mask_isobath(SSS, expt)
    # SSS_shelf_mean = (SSS_shelf*area_shelf).sum(['xh', 'yh']).compute()
    SSS_W_mean = (SSS*area_W).sum(['xh', 'yh']).compute()
    SSS_R_mean = (SSS*area_R).sum(['xh', 'yh']).compute()
    
    S_bot = cc.querying.getvar(
        expt, 'sob', session, frequency='1 daily',
        start_time=start_time, end_time=end_time,
        chunks={'xh': '200MB', 'yh': '200MB'})
    S_bot = S_bot.sel(yh=slice(None, -59), time=slice(start_time, end_time))
    S_bot_shelf = shelf_mask_isobath(S_bot, expt)
    # S_bot_shelf_mean = (S_bot_shelf*area_shelf).sum(['xh', 'yh']).compute()
    S_bot_W_mean = (S_bot*area_W).sum(['xh', 'yh']).compute()
    S_bot_R_mean = (S_bot*area_R).sum(['xh', 'yh']).compute()
    print('calculated S')
    
    # rho2 is only at monthly resolution
    rho2 = cc.querying.getvar(
        expt, 'rhopot2', session, frequency='1 monthly',
        start_time=start_time, end_time=end_time,
        chunks={'xh': '200MB', 'yh': '200MB'})
    rho2 = rho2.sel(yh=slice(None, -59), time=slice(start_time, end_time))
    rho2_shelf = shelf_mask_isobath(rho2, expt)
    
    # surface
    # rho2_surf_shelf_mean = (rho2_shelf.isel(z_l=0)*area_shelf).sum(
    #     ['xh', 'yh']).compute()
    rho2_surf_W_mean = (rho2_shelf.isel(z_l=0)*area_W).sum(
        ['xh', 'yh']).compute()
    rho2_surf_R_mean = (rho2_shelf.isel(z_l=0)*area_R).sum(
        ['xh', 'yh']).compute()
    # select bottom value of potrho2
    rho2_bot_shelf = select_bottom_values(rho2_shelf, land_mask)
    # rho2_bot_shelf_mean = (rho2_bot_shelf*area_shelf).sum(
        # ['xh', 'yh']).compute()
    rho2_bot_W_mean = (rho2_bot_shelf*area_W).sum(
        ['xh', 'yh']).compute()
    rho2_bot_R_mean = (rho2_bot_shelf*area_R).sum(
        ['xh', 'yh']).compute()
    print('calculated rho')
    
    MLD = cc.querying.getvar(
        expt, 'mlotst', session, frequency='1 daily',
        start_time=start_time, end_time=end_time,
        chunks={'xh': '200MB', 'yh': '200MB'})
    MLD = MLD.sel(yh=slice(None, -59), time=slice(start_time, end_time))
    MLD_shelf = shelf_mask_isobath(MLD, expt)
    # MLD_shelf_mean = (MLD_shelf*area_shelf).sum(['xh', 'yh']).compute()
    MLD_W_mean = (MLD*area_W).sum(['xh', 'yh']).compute()
    MLD_R_mean = (MLD*area_R).sum(['xh', 'yh']).compute()
    print('calculated MLD')
    
    ice_conc = cc.querying.getvar(
        expt, 'siconc', session, frequency='1 daily',
        start_time=start_time, end_time=end_time,
        chunks={'xT': '200 MB', 'yT': '200 MB'}).rename({'yT':'yh', 'xT':'xh'})
    ice_conc = ice_conc.sel(yh=slice(None, -59), time=slice(start_time, end_time))
    ice_conc_shelf = shelf_mask_isobath(ice_conc, expt)
    # ice_conc_shelf_mean = (ice_conc_shelf*area_shelf).sum(['xh', 'yh']).compute()
    ice_conc_W_mean = (ice_conc*area_W).sum(['xh', 'yh']).compute()
    ice_conc_R_mean = (ice_conc*area_R).sum(['xh', 'yh']).compute()
    print('calculated ice concentration')
    
    ice_thick = cc.querying.getvar(
        expt, 'sithick', session, frequency='1 daily',
        start_time=start_time, end_time=end_time,
        chunks={'xT': '200 MB', 'yT': '200 MB'}).rename({'yT':'yh', 'xT':'xh'})
    ice_thick = ice_thick.sel(yh=slice(None, -59), time=slice(start_time, end_time))
    ice_thick_shelf = shelf_mask_isobath(ice_thick, expt)
    # ice_thick_shelf_mean = (ice_thick_shelf*area_shelf).sum(['xh', 'yh']).compute()
    ice_thick_W_mean = (ice_thick*area_W).sum(['xh', 'yh']).compute()
    ice_thick_R_mean = (ice_thick*area_R).sum(['xh', 'yh']).compute()
    print('calculated ice thickness')

#     SST_shelf_mean.name = 'tos'
#     ds = SST_shelf_mean.to_dataset()
#     ds['tob'] = T_bot_shelf_mean
#     ds['sos'] = SSS_shelf_mean
#     ds['sob'] = S_bot_shelf_mean
#     ds['mlotst'] = MLD_shelf_mean
#     ds['siconc'] = ice_conc_shelf_mean
#     ds['sithick'] = ice_thick_shelf_mean
#     ds.attrs = {'mean': 'area weighted mean integrated over the continental ' +
#                 'shelf south of the 1000 m isobath'}
#     comp = dict(zlib=True, complevel=5, shuffle=True)
#     enc = {var: comp for var in ds.data_vars}
#     ds.to_netcdf(
#         path_output_SWMT + 'Timeseries_shelf_properties_daily_' +
#         expt + '_' + str(SST_shelf_mean.time.min().dt.year.values) +
#         '-' + str(SST_shelf_mean.time.max().dt.year.values) + '.nc',
#         encoding=enc)
    
#     rho2_surf_shelf_mean.name = 'rhopot2_surface'
#     ds_m  = rho2_surf_shelf_mean.to_dataset()
#     ds_m['rhopot2_bottom'] = rho2_bot_shelf_mean
#     ds_m.attrs = {'mean': 'area weighted mean integrated over the continental ' +
#                   'shelf south of the 1000 m isobath'}
#     comp = dict(zlib=True, complevel=5, shuffle=True)
#     enc = {var: comp for var in ds_m.data_vars}
#     ds_m.to_netcdf(
#         path_output_SWMT + 'Timeseries_shelf_properties_monthly_' +
#         expt + '_' + str(SST_shelf_mean.time.min().dt.year.values) +
#         '-' + str(SST_shelf_mean.time.max().dt.year.values) + '.nc',
#         encoding=enc)
    
    SST_W_mean.name = 'tos'
    ds = SST_W_mean.to_dataset()
    ds['tob'] = T_bot_W_mean
    ds['sos'] = SSS_W_mean
    ds['sob'] = S_bot_W_mean
    ds['mlotst'] = MLD_W_mean
    ds['siconc'] = ice_conc_W_mean
    ds['sithick'] = ice_thick_W_mean
    ds.attrs = {'mean': 'area weighted mean integrated over the continental ' +
                'shelf south of the 1000 m isobath in the Weddell Sea'}
    comp = dict(zlib=True, complevel=5, shuffle=True)
    enc = {var: comp for var in ds.data_vars}
    ds.to_netcdf(
        path_output_SWMT + 'Timeseries_Weddell_properties_daily_' +
        expt + '_' + str(SST_W_mean.time.min().dt.year.values) +
        '-' + str(SST_W_mean.time.max().dt.year.values) + '.nc',
        encoding=enc)
    
    rho2_surf_W_mean.name = 'rhopot2_surface'
    ds_m  = rho2_surf_W_mean.to_dataset()
    ds_m['rhopot2_bottom'] = rho2_bot_W_mean
    ds_m.attrs = {'mean': 'area weighted mean integrated over the continental ' +
                  'shelf south of the 1000 m isobath in the Weddell Sea'}
    comp = dict(zlib=True, complevel=5, shuffle=True)
    enc = {var: comp for var in ds_m.data_vars}
    ds_m.to_netcdf(
        path_output_SWMT + 'Timeseries_Weddell_properties_monthly_' +
        expt + '_' + str(SST_W_mean.time.min().dt.year.values) +
        '-' + str(SST_W_mean.time.max().dt.year.values) + '.nc',
        encoding=enc)
    
    SST_R_mean.name = 'tos'
    ds = SST_R_mean.to_dataset()
    ds['tob'] = T_bot_R_mean
    ds['sos'] = SSS_R_mean
    ds['sob'] = S_bot_R_mean
    ds['mlotst'] = MLD_R_mean
    ds['siconc'] = ice_conc_R_mean
    ds['sithick'] = ice_thick_R_mean
    ds.attrs = {'mean': 'area weighted mean integrated over the continental ' +
                'shelf south of the 1000 m isobath in the Ross Sea'}
    comp = dict(zlib=True, complevel=5, shuffle=True)
    enc = {var: comp for var in ds.data_vars}
    ds.to_netcdf(
        path_output_SWMT + 'Timeseries_Ross_properties_daily_' +
        expt + '_' + str(SST_R_mean.time.min().dt.year.values) +
        '-' + str(SST_R_mean.time.max().dt.year.values) + '.nc',
        encoding=enc)
    
    rho2_surf_R_mean.name = 'rhopot2_surface'
    ds_m  = rho2_surf_R_mean.to_dataset()
    ds_m['rhopot2_bottom'] = rho2_bot_W_mean
    ds_m.attrs = {'mean': 'area weighted mean integrated over the continental ' +
                  'shelf south of the 1000 m isobath in the Ross Sea'}
    comp = dict(zlib=True, complevel=5, shuffle=True)
    enc = {var: comp for var in ds_m.data_vars}
    ds_m.to_netcdf(
        path_output_SWMT + 'Timeseries_Ross_properties_monthly_' +
        expt + '_' + str(SST_R_mean.time.min().dt.year.values) +
        '-' + str(SST_R_mean.time.max().dt.year.values) + '.nc',
        encoding=enc)