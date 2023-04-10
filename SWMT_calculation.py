#!/usr/bin/env python
# coding: utf-8

"""Calculation of Surface Water Mass Transformation in MOM6"""


import xarray as xr
import numpy as np
import cosima_cookbook as cc
from dask.distributed import Client
import cftime
from datetime import timedelta
from gsw import alpha, SA_from_SP, p_from_z, CT_from_pt, beta, sigma1
import sys
import matplotlib.path as mpath

if __name__ == '__main__':
    def calculate_SWMT(expt, session, year, frequency,
                       path_output, lat_north=-59):

        '''
        Computes southern ocean surface water mass transformation rates
        (partitioned into transformation from heat and freshwater) referenced to
        1000 db from monthly ACCESS-OM2 output.
        Suitable for analysis of high-resolution (0.1 degree) output
        (the scattered .load()'s allowed this)

        expt - text string indicating the name of the experiment
        session - a database session created by cc.database.create_session()
        start_time - text string designating the start date ('YYYY-MM-DD')
        end_time - text string indicating the end date ('YYYY-MM-DD')
        path_output - text string indicating directory where output databases are
            to be saved
        filename - text string of the name of the saved file
        lat_north - function computes processes between lat = -90 and
            lat = lat_north

        NOTE: assumes SST is potential temperature (°C) and
              SSS is practical salinity (psu)

        required modules:
        xarray as xr
        numpy as np
        cosima_cookbook as cc
        from gsw import alpha, SA_from_SP, p_from_z, CT_from_pt, beta, sigma1
        '''

        # load variables
        # potential temperature
        SST = cc.querying.getvar(expt, 'tos', session, frequency='1 daily') 
        # practical salinity
        SSS_PSU =  cc.querying.getvar(expt, 'sos', session, frequency='1 daily')

        net_surface_heating = cc.querying.getvar(
            expt, 'hfds', session, frequency=frequency)
        # mass flux of precip - evap + river ((kg of water)/m^2/s )
        pme_river = cc.querying.getvar(
            expt, 'wfo', session, frequency=frequency)

        # slice for time and latitudinal constraints
        start_time=cftime.datetime(
            year, 1, 1, 0, 0, 0, 0, calendar='noleap', has_year_zero=True)
        end_time=cftime.datetime(
            year, 12, 31, 0, 0, 0, 0, calendar='noleap', has_year_zero=True)
        time_slice = slice(start_time, end_time)
        lat_slice = slice(-90, lat_north)
        SST = SST.sel(time=time_slice, yh=lat_slice)
        SSS_PSU = SSS_PSU.sel(time=time_slice, yh=lat_slice)
        net_surface_heating = net_surface_heating.sel(time=time_slice, yh=lat_slice)
        pme_river = pme_river.sel(time=time_slice, yh=lat_slice)

        # convert to monthly
        time_monthly = pme_river.time.values
        SST = SST.resample(
            time="1M", label='left', loffset=timedelta(days=16)).mean("time")
        SST['time'] = time_monthly
        SSS_PSU = SSS_PSU.resample(
            time="1M", label='left', loffset=timedelta(days=16)).mean("time")
        SSS_PSU['time'] = time_monthly

        # extract coordinate arrays
        yh = SST.yh.values
        xh = SST.xh.values

        months_standard_noleap = np.array(
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        days_per_month = xr.DataArray(
            months_standard_noleap,
            coords=[time_monthly], dims=['time'], name='days per month')

        depth = -0.541281 # st_ocean value of the uppermost cell

        # 2D arrays of depths, longitude, latitude and pressure
        depth_2D = xr.DataArray(
            data = np.full((len(yh), len(xh)), depth),
            dims = ["yh","xh"], coords = {'yh': yh, 'xh': xh})
        xh_2D = np.tile(xh,(len(yh),1))
        yh_2D = np.tile(yh,(len(xh),1)).transpose()

        pressure = xr.DataArray(
            p_from_z(depth_2D, yh_2D), coords=[yh, xh], dims=['yh', 'xh'],
            name='pressure', attrs={'units': 'dbar'})

        # convert units to absolute salinity
        SSS = xr.DataArray(
            SA_from_SP(SSS_PSU, pressure, xh_2D, yh_2D), coords=[time_monthly, yh, xh],
            dims = ['time', 'yh', 'xh'], name='sea surface salinity',
            attrs={'units': 'Absolute Salinity (g/kg)'})
        ## convert to conservative temperature but in Celsius not Kelvin
        SST_Conservative = xr.DataArray(
            CT_from_pt(SSS, SST), coords=[time_monthly, yh, xh], dims=['time', 'yh', 'xh'],
            name='sea surface temperature', attrs = {'units': 'Conservative Temperature (C)'})

        # compute potential density referenced to 1000dbar
        # (or referenced otherwise, depending on your purpose)
        pot_rho_1 = xr.DataArray(
            sigma1(SSS, SST_Conservative), coords=[time_monthly, yh, xh],
            dims = ['time', 'yh', 'xh'], name='potential density ref 1000dbar',
            attrs = {'units': 'kg/m^3 (-1000 kg/m^3)'})
        pot_rho_1 = pot_rho_1.load()

        # Compute salt transformation (no density binning)
        haline_contraction = xr.DataArray(
            beta(SSS, SST_Conservative, pressure), coords=[time_monthly, yh, xh],
            dims=['time', 'yh', 'xh'],
            name='saline contraction coefficient (constant conservative temp)',
            attrs = {'units': 'kg/g'})
        salt_transformation = haline_contraction*SSS*pme_river*days_per_month
        salt_transformation = salt_transformation.load()

        # Compute heat transformation (no density binning)
        thermal_expansion = xr.DataArray(
            alpha(SSS, SST_Conservative, pressure), coords=[time_monthly, yh, xh],
            dims=['time', 'yh', 'xh'],
            name='thermal expansion coefficient (constant conservative temp)',
            attrs = {'units':'1/K'})
        heat_transformation =  thermal_expansion*net_surface_heating*days_per_month
        heat_transformation = heat_transformation.load()

        # Record the time bounds before summing through time 
        # (just to make sure it's consistent with requested years)
        time_bounds =  (
            str(salt_transformation.coords['time.year'][0].values) + '_' +
            str(salt_transformation.coords['time.month'][0].values) + '-' +
            str(salt_transformation.coords['time.year'][-1].values) + '_' +
            str(salt_transformation.coords['time.month'][-1].values))

        # Isopycnal binning in several steps:
        # cycle through isopycnal bins, determine which cells are within the
        # given bin for each time step, find the transformation values for
        # those cells for each time step, sum these through time.
        # -> array of shape (isopyncal bins * lats * lons) where the array
        # associated with a given isopycnal bin is NaN everywhere except where
        # pot_rho_0 was within the bin, there it has a time summed
        # transformation value

        # alter if this density range doesn't capture surface processes in your
        # study region, or if a different density field (not sigma1) is used
        isopycnal_bins = np.arange(31, 33.5, 0.02)

        bin_bottoms = isopycnal_bins[:-1]
        binned_salt_transformation = xr.DataArray(
            np.zeros((len(bin_bottoms), len(yh), len(xh))),
            coords=[bin_bottoms, yh, xh], dims=['isopycnal_bins', 'yh', 'xh'],
            name='salt transformation in isopycnal bins summed over time')
        binned_salt_transformation.chunk({'isopycnal_bins': 1})
        for i in range(len(isopycnal_bins)-1):
            bin_mask = pot_rho_1.where(pot_rho_1 <= isopycnal_bins[i+1]).where(
                pot_rho_1 > isopycnal_bins[i]) * 0 + 1
            masked_transform = (salt_transformation * bin_mask).sum(dim='time') 
            masked_transform = masked_transform.where(masked_transform != 0) 
            masked_transform = masked_transform.load()
            binned_salt_transformation[i, :, :] = masked_transform
        print('salt_transformation binning done')

        binned_heat_transformation = xr.DataArray(
            np.zeros((len(bin_bottoms), len(yh), len(xh))),
            coords=[bin_bottoms, yh, xh], dims=['isopycnal_bins', 'yh', 'xh'],
            name='heat transformation in isopycnal bins summed over time')
        binned_heat_transformation.chunk({'isopycnal_bins': 1})

        # This for loop is likely the least efficient part of the code.
        # The problem can be vectorised for improved speed
        for i in range(len(isopycnal_bins)-1):
            bin_mask = pot_rho_1.where(pot_rho_1 <= isopycnal_bins[i+1]).where(
                pot_rho_1 > isopycnal_bins[i]) * 0 + 1
            masked_transform = (heat_transformation * bin_mask).sum(dim='time') 
            masked_transform = masked_transform.where(masked_transform != 0)
            masked_transform = masked_transform.load()
            binned_heat_transformation[i, :, :] = masked_transform
        print('heat_transformation binning done')

        ndays = days_per_month.sum().values
        salt_transformation = binned_salt_transformation/ndays
        c_p = 3992.1
        heat_transformation = binned_heat_transformation/c_p/ndays

        isopycnal_bin_diff = np.diff(isopycnal_bins)
        salt_transformation = salt_transformation/isopycnal_bin_diff[
            :, np.newaxis, np.newaxis]
        heat_transformation = heat_transformation/isopycnal_bin_diff[
            :, np.newaxis, np.newaxis]
        isopycnal_bin_mid = (isopycnal_bins[1:] + isopycnal_bins[:-1])/2

        # this procedure defines fluxes from lighter to denser classes
        # as negative, I want the opposite
        salt_transformation = salt_transformation *-1
        heat_transformation = heat_transformation *-1

        # Save to file
        salt_transformation = salt_transformation.expand_dims(time=[year])
        heat_transformation = heat_transformation.expand_dims(time=[year])
        ds = xr.Dataset(
            {'binned_salt_transformation': salt_transformation,
             'binned_heat_transformation': heat_transformation,
             'time_bounds': time_bounds})
        ds.coords['isopycnal_bins'] = isopycnal_bin_mid  # isopycnal bin midpoints
        ds.attrs = {'units': 'm/s'}
        comp = dict(chunksizes=(1, 42, 255, 188),
                    zlib=True, complevel=5, shuffle=True)
        enc = {var: comp for var in ds.data_vars}
        ds.to_netcdf(
            path_output + 'SWMT_' + expt + '_mean_' + time_bounds + '.nc',
            encoding=enc)
        print('SWMT_' + expt + '_mean_' + time_bounds + '.nc' +
              ' saved to ' + path_output)
    
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
    
    def shelf_mask_isobath(var, output_mask=False):
        '''
        Mask varibales from MOM6 at 1/10th by the region polewards
        of the 1000m isobath
        '''
        contour_file = np.load(
            '/g/data/ik11/grids/Antarctic_slope_contour_1000m.npz')

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
    

    client = Client()
    client

    year = int(sys.argv[1])
    expt = sys.argv[2]
    
    session = cc.database.create_session()
    frequency = '1 monthly'
    path_output = '/g/data/e14/cs6673/mom6_comparison/data_DSW/'
    
    """Calculate SWMT"""

    calculate_SWMT(expt, session, year, frequency, path_output, lat_north=-59)
    
    
    """Spatial sum of SWMT in DSW regions"""

    DSW_region = {
    'name': ['Weddell', 'Prydz', 'Adelie', 'Ross'],
    'name_long': ['Weddell Sea', 'Prydz Bay', 'Adélie Coast', 'Ross Sea'],
    'lon': [[-60, -35, -48, -62, -60],
            [48, 73, 74, 48, 48],
            [128-360, 152-360, 152-360, 128-360, 128-360],
            [185-360, 160-360, 164-360, 172-360, 185-360]],
    'lat': [[-71, -75, -78, -75, -71],
            [-65, -66.5, -69, -68, -65],
            [-64.5, -66, -69, -67.5, -64.5],
            [-78, -78, -73, -71.5, -78]]}

    area = cc.querying.getvar(
        expt=expt, variable='areacello', session=session, frequency='static', n=1)
    xh = area.xh
    yh = area.yh

    for a, area_text in enumerate(DSW_region['name']):
        mask = mask_from_polygon(DSW_region['lon'][a], DSW_region['lat'][a],
                                 xh, yh)
        mask = mask.where(mask == True, 0)
        if a == 0:
            mask_DSW = mask.expand_dims(area=[area_text])
        else:
            mask_DSW = xr.concat((mask_DSW, mask.expand_dims(
                area=[area_text])), dim='area')
    mask_DSW = mask_DSW.where(mask_DSW != 0)

    ds_SWMT = xr.open_mfdataset(
        path_output + 'SWMT_' + expt + '_mean_' + str(year) + '*.nc')
    swmt_heat = ds_SWMT.binned_heat_transformation
    swmt_salt = ds_SWMT.binned_salt_transformation

    swmt_heat_shelf = shelf_mask_isobath(swmt_heat)
    swmt_salt_shelf = shelf_mask_isobath(swmt_salt)
    area_shelf = shelf_mask_isobath(area)

    area_DSW = (mask_DSW * area.sel(yh=slice(-90, -59)))
    swmt_heat_DSW_regions = (swmt_heat_shelf * area_DSW/1e6).sum(
        ['xh', 'yh']).compute()
    swmt_salt_DSW_regions = (swmt_salt_shelf * area_DSW/1e6).sum(
        ['xh', 'yh']).compute()

    swmt_heat_DSW_regions.name = 'binned_heat_transformation_in_DSW_region'
    ds = swmt_heat_DSW_regions.to_dataset()
    ds['binned_salt_transformation_in_DSW_region'] = swmt_salt_DSW_regions
    ds.attrs = {'units': 'Sv'}
    comp = dict(zlib=True, complevel=5, shuffle=True)
    enc = {var: comp for var in ds.data_vars}
    ds.to_netcdf(
        path_output + 'SWMT_in_DSW_region_' + expt + '_' + str(year) + '.nc',
        encoding=enc)

