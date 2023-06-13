#!/usr/bin/env python
# coding: utf-8

"""Export across the 1000-m isobath in MOM6"""

import xarray as xr
import numpy as np
import cosima_cookbook as cc
from dask.distributed import Client
import sys

if __name__ == '__main__':
    
    client = Client()
    client

    year = int(sys.argv[1])
    year = str(year)
    expt = sys.argv[2]
    expt_name = sys.argv[3]
    contour_depth = sys.argv[4]
    
    db = expt_name + '.db'
    session = cc.database.create_session(db)
    frequency = '1 monthly'
    path_output = '/g/data/e14/cs6673/mom6_comparison/data_DSW/'
    resolution = expt_name.split('_')[1][:-3]
    
    start_time= year + '-01-01'
    end_time= year + '-12-31'

    # reference density in MOM6 
    rho_0 = 1035.0
    # Note: change this range, so it matches the size of contour arrays
    lat_range = slice(-79, -55)

    '''Open contour data'''
    ds_contour = xr.open_dataset(
        '/home/142/cs6673/work/mom6_comparison/Antarctic_slope_contours/' +
        'Antarctic_slope_contour_' + str(contour_depth) + 'm_MOM6_' + resolution +
        'deg.nc')

    # load data and rename coordinates to general x/y to be able to multiply them
    mask_y_transport = ds_contour.mask_y_transport.rename(
        {'yq': 'y', 'xh': 'x'})
    mask_x_transport = ds_contour.mask_x_transport.rename(
        {'yh': 'y', 'xq': 'x'})
    mask_y_transport_numbered = ds_contour.mask_y_transport_numbered.rename(
        {'yq': 'y', 'xh': 'x'})
    mask_x_transport_numbered = ds_contour.mask_x_transport_numbered.rename(
        {'yh': 'y', 'xq': 'x'})

    # number of points along contour:
    num_points = int(np.maximum(
        np.max(mask_y_transport_numbered),np.max(mask_x_transport_numbered)))
    
    '''Stack contour data into 1D and extract lat/lon on contour'''
    # Create the contour order data-array. Note that in this procedure the
    # x-grid counts have x-grid dimensions and the y-grid counts have y-grid
    # dimensions, but these are implicit, the dimension *names* are kept
    # general across the counts, the generic y/x, so that concatening
    # works but we dont double up with numerous counts for one lat/lon
    # point.
    
    # stack contour data into 1d:
    mask_x_numbered_1d = mask_x_transport_numbered.stack(contour_index = ['y', 'x'])
    mask_x_numbered_1d = mask_x_numbered_1d.where(mask_x_numbered_1d > 0, drop = True)
    mask_y_numbered_1d = mask_y_transport_numbered.stack(contour_index = ['y', 'x'])
    mask_y_numbered_1d = mask_y_numbered_1d.where(mask_y_numbered_1d > 0, drop = True)
    contour_ordering = xr.concat((mask_x_numbered_1d,mask_y_numbered_1d), dim = 'contour_index')
    contour_ordering = contour_ordering.sortby(contour_ordering)

    # get lat and lon along contour, useful for plotting later:
    lat_along_contour = contour_ordering.y
    lon_along_contour = contour_ordering.x
    contour_index_array = np.arange(1,len(contour_ordering)+1)
    # don't need the multi-index anymore, replace with contour count and save
    lat_along_contour = lat_along_contour.drop_vars({'x', 'y', 'contour_index'})
    lat_along_contour.coords['contour_index'] = contour_index_array
    lon_along_contour = lon_along_contour.drop_vars({'x', 'y', 'contour_index'})
    lon_along_contour.coords['contour_index'] = contour_index_array
    
    '''Load mass transport umo and vmo'''
    vmo = cc.querying.getvar(
        expt, 'vmo', session, frequency=frequency,
        start_time=start_time, end_time=end_time,
        chunks={'xh': '200MB', 'yq': '200MB'})
    umo = cc.querying.getvar(
        expt, 'umo', session, frequency=frequency,
        start_time=start_time, end_time=end_time,
        chunks={'xq': '200MB', 'yh': '200MB'})

    # select latitude range and this year:
    vmo = vmo.sel(yq=lat_range).sel(time=slice(start_time,end_time))
    vmo = vmo.isel(yq=slice(1, None))
    umo = umo.sel(yh=lat_range).sel(time=slice(start_time,end_time))
    umo = umo.isel(xq=slice(1, None))

    # Note that vmo is Ocean Mass Y Transport (kg s-1) and defined as the transport across
    # the northern edge of a tracer cell so its coordinates should be (yq, xh).
    # umo is Ocean Mass X Transport (kg s-1) and defined as the transport across
    # the eastern edge of a tracer cell so its coordinates should be (yh, xq).
    # However we will keep the actual name as simply y/x irrespective of the variable
    # to make concatenation and sorting possible.
    vmo = vmo.rename({'yq':'y', 'xh':'x'})
    umo = umo.rename({'yh':'y', 'xq':'x'})

    # convert kg/s to Sv and multiply by contour masks
    vmo = vmo/(1e6*rho_0)*mask_y_transport
    umo = umo/(1e6*rho_0)*mask_x_transport
    
    '''Extract transport values along contour'''
    umo_i = umo.compute()
    vmo_i = vmo.compute()

    # stack transports into 1d and drop any points not on contour:
    x_transport_1d_i = umo_i.stack(contour_index=['y', 'x'])
    x_transport_1d_i = x_transport_1d_i.where(mask_x_numbered_1d>0, drop=True)
    y_transport_1d_i = vmo_i.stack(contour_index=['y', 'x'])
    y_transport_1d_i = y_transport_1d_i.where(mask_y_numbered_1d>0, drop=True)

    # combine all points on contour:
    vol_trans_across_contour = xr.concat(
        (x_transport_1d_i, y_transport_1d_i), dim='contour_index')
    vol_trans_across_contour = vol_trans_across_contour.sortby(contour_ordering)
    vol_trans_across_contour = vol_trans_across_contour.drop_vars(
        {'x', 'contour_index', 'y'})
    vol_trans_across_contour.coords['contour_index'] = contour_index_array
    vol_trans_across_contour = vol_trans_across_contour.compute()
    
    '''Save data'''
    vol_trans_across_contour.name = 'vol_trans_across_contour'
    vol_trans_across_contour.attrs = {
        'long_name': 'Volume transport across 1000-m isobath',
        'units': 'Sv'}
    ds = vol_trans_across_contour.to_dataset()
    ds['lat'] = lat_along_contour
    ds['lon'] = lon_along_contour
    if len(vol_trans_across_contour.contour_index) < 15000:
        chunk_ind = len(vol_trans_across_contour.contour_index)
    else:
        chunk_ind = 10000
    enc = {'vol_trans_across_contour':
           {'chunksizes': (12, 99, chunk_ind),
            'zlib': True, 'complevel': 5, 'shuffle': True}}
    ds.to_netcdf(
        path_output + 'vol_transp_across_' + str(contour_depth) +
        'm_isobath_' + expt_name + '_' + frequency[:3:2] + '_' +
        year + '.nc', encoding=enc)