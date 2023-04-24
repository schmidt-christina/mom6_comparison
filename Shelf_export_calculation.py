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
    
    session = cc.database.create_session()
    frequency = '1 monthly'
    path_output = '/g/data/e14/cs6673/mom6_comparison/data_DSW/'
    
    start_time= year + '-01-01'
    end_time= year + '-12-31'

    # reference density in MOM6 
    rho_0 = 1035.0
    # Note: change this range, so it matches the size of contour arrays
    lat_range = slice(-90,-59)

    
    '''Open grid cell width data for domain'''
    # some grid data is required, a little complicated because these
    # variables don't behave well with some 
    dyt = cc.querying.getvar(expt, 'dyt',session, n=1)
    dxu = cc.querying.getvar(expt, 'dxCu',session, n=1)
    dxv = cc.querying.getvar(expt, 'dxCv',session, n=1)

    # select latitude range:
    dyt = dyt.sel(yh=lat_range)
    dxu = dxu.sel(yh=lat_range)
    dxv = dxv.sel(yq=lat_range)

    '''Open contour data'''
    isobath_depth = 1000
    outfile = ('/g/data/v45/akm157/model_data/access-om2/Antarctic_slope_contour_' +
               str(isobath_depth) + 'm.npz')
    data = np.load(outfile)
    mask_y_transport = data['mask_y_transport']
    mask_x_transport = data['mask_x_transport']
    mask_y_transport_numbered = data['mask_y_transport_numbered']
    mask_x_transport_numbered = data['mask_x_transport_numbered']

    yh = cc.querying.getvar(expt, 'yh', session, n=1)
    yh = yh.sel(yh=lat_range)
    yq = cc.querying.getvar(expt, 'yq', session, n=1)
    yq = yq.sel(yq=lat_range)
    yq = yq[1:]
    xh = cc.querying.getvar(expt, 'xh', session, n=1)
    xq = cc.querying.getvar(expt, 'xq', session, n=1)[1:]

    # Convert contour masks to data arrays, so we can multiply them later.
    # We need to ensure the lat lon coordinates correspond to the actual data location:
    #       The y masks are used for vmo, so like vmo this should have dimensions (yq, xh).
    #       The x masks are used for umo, so like umo this should have dimensions (yh, xq).
    #       However the actual name will always be simply y or x irrespective of the variable
    #       to make concatenation of transports in both direction and sorting possible.

    mask_x_transport = xr.DataArray(
        mask_x_transport, coords=[('y', yh.data), ('x', xq.data)])
    mask_y_transport = xr.DataArray(
        mask_y_transport, coords=[('y', yq.data), ('x', xh.data)])
    mask_x_transport_numbered = xr.DataArray(
        mask_x_transport_numbered, coords=[('y', yh.data), ('x', xq.data)])
    mask_y_transport_numbered = xr.DataArray(
        mask_y_transport_numbered, coords=[('y', yq.data), ('x', xh.data)])

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
    enc = {'vol_trans_across_contour':
       {'chunksizes': (12, 79, 6002),
        'zlib': True, 'complevel': 5, 'shuffle': True}}
    ds.to_netcdf(
        path_output + 'vol_transp_across_contour_' + expt + '_' +
        frequency[:3:2] + '_' + year + '.nc', encoding=enc)