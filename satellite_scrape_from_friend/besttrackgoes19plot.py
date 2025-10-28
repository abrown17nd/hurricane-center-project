import xarray as xr 
import matplotlib.pyplot as plt
import cartopy, cartopy.crs as ccrs 
import satcmaps as cmaps
import image_retrieval as ir
import numpy as np 
from datetime import datetime, timedelta
from pyproj import Proj 
import scipy 
import os

from ibtracs_reader_new import getIBTRACS, getTCPointsOnly
import getintervals as intervals

#Couple of filepaths, edit as need be


ibtracs_filename = '/Users/elliott/Documents/Grad Research/dataset_readers/dataset_files/ibtracs.since1980.list.v04r01.csv'
ibtracs_data = getIBTRACS(ibtracs_filename)

base_folder_path = r"/Users/elliott/Documents/Grad Research/"
image_folder_path = r"/Users/elliott/Documents/Grad Research/ImeldaFozIR2025/"
image_prefix = "ImeldaGOES"


#Import IBTRACS data.
#Data can be accessed at:
#https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/
#For the purposes of this code, the 1980 onwards IBTRACS CSV file is used.
    
IBTRACSID = ibtracs_data[0]
year = ibtracs_data[1]
number = ibtracs_data[2]
basin = ibtracs_data[3]
stormname = ibtracs_data[4]
timestamp = ibtracs_data[5]
stormDT = ibtracs_data[6]
stormNature = ibtracs_data[7]
lat = ibtracs_data[8]
lon = ibtracs_data[9]
vmax = ibtracs_data[10]
pressure = ibtracs_data[11]
disttoland = ibtracs_data[12]
disttolandfall = ibtracs_data[13]
ATCFID = ibtracs_data[14]
SSHWS = ibtracs_data[15]
rmw = ibtracs_data[16]
alluniqueATCFID = ibtracs_data[17]
orderedUniqueATCFID = ibtracs_data[18]

vmaxarray = np.asarray(vmax)

class storm:
  def __init__(self, IBTRACSID, year, number, basin, stormname, timestamp, stormDT, stormNature,
         lat, lon, vmax, pressure, disttoland, disttolandfall, ATCFID, SSHWS, rmw):
      self.IBTRACSID = IBTRACSID
      self.year = year
      self.number = number
      self.basin= basin
      self.stormname = stormname
      self.timestamp = timestamp
      self.stormDT = stormDT
      self.lat = lat
      self.lon = lon
      self.vmax = vmax
      self.pressure = pressure
      self.disttoland = disttoland
      self.disttolandfall = disttolandfall
      self.ATCFID = ATCFID
      self.SSHWS = SSHWS
      self.rmw = rmw
    
    
def getStorm(atcfid):
    myStorm = storm([],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[])
    for i in range(len(IBTRACSID)):
        if (ATCFID[i] == atcfid):
            myStorm.IBTRACSID.append(IBTRACSID[i])
            myStorm.year.append(int(year[i]))
            myStorm.number.append(number[i])
            myStorm.basin.append(basin[i])
            myStorm.stormname.append(stormname[i])
            myStorm.timestamp.append(timestamp[i])
            myStorm.stormDT.append(stormDT[i])
            if lat[i] != '' and lon[i] != '' and vmax[i] != ' ':
                myStorm.lat.append(float(lat[i]))
                myStorm.lon.append(float(lon[i]))
                myStorm.vmax.append(int(vmax[i]))
            else:
                myStorm.lat.append(0.0)
                myStorm.lon.append(0.0)
                myStorm.vmax.append(0)
            myStorm.pressure.append(pressure[i])
            myStorm.disttoland.append(disttoland[i])
            myStorm.disttolandfall.append(disttolandfall[i])
            myStorm.ATCFID.append(ATCFID[i])
            myStorm.SSHWS.append(SSHWS[i])
            myStorm.rmw.append(rmw[i])
    return myStorm

stormData = getStorm("EP042023")
'''
individualStormDatetimes = stormData.stormDT
individualStormLat = stormData.lat
individualStormLon = stormData.lon

'''
individualStormDatetimes = intervals.generate_intervals_datetime(datetime(2025, 9, 27, 18, 0, 0), datetime(2025, 10, 2, 15, 0, 0), timedelta(minutes=180))
individualStormLat = [22.1, 22.3, 22.4, 22.6, 22.7, 22.9, 23.1, 23.5, 23.8, 24.2, 24.6, 24.9, 25.1, 25.7, 26.4, 26.8, 27.2, 27.7, 28.1, 28.4, 28.6, 28.8, 28.9, 29.0, 29.1, 29.2, 29.4, 29.6, 29.9, 30.3, 30.7, 31.0, 31.3, 31.5, 31.8, 32.1, 32.5, 32.7, 32.9]
individualStormLon = [-76.5, -76.7, -76.8, -77.0, -77.1, -77.2, -77.3, -77.3, -77.3, -77.2, -77.1, -77.1, -77.1, -77.1, -77.1, -77.2, -77.3, -77.3, -77.3, -77.3, -77.2, -77.1, -76.9, -76.5, -76.0, -75.4, -74.7, -73.9, -73.1, -72.2, -71.3, -70.2, -69.2, -67.8, -66.4, -64.8, -63.3, -62.1, -60.9]


start_time = individualStormDatetimes[0]  # 10:00 AM
end_time = individualStormDatetimes[len(individualStormDatetimes)-1]    # 12:00 PM
interval = timedelta(minutes=10)

target_time_1 = datetime(2025, 9, 27, 18, 0, 0)
target_time_2 = datetime(2025, 10, 2, 12, 0, 0)
fullplottimes = intervals.generate_intervals_datetime(start_time, end_time, interval)
#plottimes = intervals.generate_intervals_datetime(start_time, end_time, interval)
plottimes = [dt for dt in fullplottimes if dt > target_time_1 and dt < target_time_2]

def nearest_datetime(datetime_list, target_datetime):
    """Finds the nearest datetime in the list to the target datetime.

    Args:
        datetime_list (list): List of datetime objects.
        target_datetime (datetime): Target datetime object.

    Returns:
        datetime: The nearest datetime in the list.
    """
    timediffs = []
    for i in range(len(datetime_list)):
        timediffs.append(abs(datetime_list[i]-target_datetime))
    nearest_index = np.argmin(timediffs)
    return nearest_index

plotlats = []
plotlons = []
plotyears = []
plotmonths = []
plotdays = []
plothours = []

for i in range(len(plottimes)):
    currentindex = nearest_datetime(individualStormDatetimes, plottimes[i])
    plotlats.append(individualStormLat[currentindex])
    plotlons.append(individualStormLon[currentindex])
    plotyears.append(plottimes[i].year)
    plotmonths.append(plottimes[i].month)
    plotdays.append(plottimes[i].day)
    plothours.append(plottimes[i].strftime('%H%M'))

def stormir(data, lon, lat, cmap):
    plt.figure(figsize = (18, 9))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
    if lon > 174:
        ax.set_extent([lon - 6, 180, lat - 6, lat + 6], crs=ccrs.PlateCarree())
    elif lon < -174:
        ax.set_extent([-180, lon + 6, lat - 6, lat + 6], crs=ccrs.PlateCarree())
    else: 
        ax.set_extent([lon - 6, lon + 6, lat - 6, lat + 6], crs=ccrs.PlateCarree())
    #ax.set_extent([-145, -80, -10, 30], crs=ccrs.PlateCarree())

    # Add coastlines, borders and gridlines
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5) 
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth = 1, color='gray', alpha=0.5, linestyle='--')   
    gl.xlabels_top = gl.ylabels_right = False    
    cmap, vmax, vmin = cmaps.fozir()#cmaps.wvtables[cmap]
    print(data)
    
    max_temp_celsius_list = []
    min_cdo_temp_celsius_list = []
    sigma_cdo_temp_celsius_list = []


    # GOES-18 projection parameters
    proj_params = {
            'proj': 'geos',
            'h': 35786023.0,
            'lon_0': -75.2,
            'sweep': 'x',
            'a': 6378137.0,
            'b': 6356752.31414,
            'units': 'm'
        }

  
    # Create the GOES projection
    p = Proj(proj_params)

    # Convert lat/lon to x/y in the GOES projection (in meters)
    x_meters, y_meters = p(lon, lat)

    # Convert from meters to the satellite's coordinate system
    # GOES-16 fixed grid coordinates are typically in radians
    x_rad = np.arctan2(x_meters, proj_params['h'])
    y_rad = np.arctan2(y_meters * np.cos(x_rad), proj_params['h'])

    # Find the nearest points in the data coordinates
    center_y = data.y.sel(y=y_rad, method='nearest')
    center_x = data.x.sel(x=x_rad, method='nearest')

    # Get the index positions
    y_idx = np.where(data.y == center_y)[0][0]
    x_idx = np.where(data.x == center_x)[0][0]

    # Create a window around the point (about 1 degree at GOES-16 resolution)
    n_points = 30  # approximately 0.6 degree at GOES-16 resolution
    y_slice = slice(max(0, y_idx - n_points), min(len(data.y), y_idx + n_points))
    x_slice = slice(max(0, x_idx - n_points), min(len(data.x), x_idx + n_points))

    # Select the data subset
    data_subset = data.isel(y=y_slice, x=x_slice)

    # Process the valid temperatures
    if data_subset.size > 0:
        max_temp_kelvin = data_subset.max().values
        max_temp_celsius = max_temp_kelvin - 273.15
        max_temp_celsius_list.append(max_temp_celsius)

    if max_temp_celsius_list:
        overall_max_temp_celsius = max(max_temp_celsius_list)
        print('Max temp is:' + str(overall_max_temp_celsius) + '°C')
    else:
        overall_max_temp_celsius = float('nan')
    maxtemp = str(round(overall_max_temp_celsius, 2))
    
    # Create a window around the point (about 1 degree at GOES-16 resolution)
    n_points_cdo = 110  # approximately 2 degree at GOES-16 resolution
    y_slice_cdo = slice(max(0, y_idx - n_points_cdo), min(len(data.y), y_idx + n_points_cdo))
    x_slice_cdo = slice(max(0, x_idx - n_points_cdo), min(len(data.x), x_idx + n_points_cdo))

    # Select the data subset
    data_subset_cdo = data.isel(y=y_slice_cdo, x=x_slice_cdo)

    # Process the valid temperatures
    if data_subset_cdo.size > 0:
        min_cdo_temp_kelvin = data_subset_cdo.min().values
        min_cdo_temp_celsius = min_cdo_temp_kelvin - 273.15
        min_cdo_temp_celsius_list.append(min_cdo_temp_celsius)

    if min_cdo_temp_celsius_list:
        overall_min_cdo_temp_celsius = min(min_cdo_temp_celsius_list)
        print('Min CDO temp is:' + str(overall_min_cdo_temp_celsius) + '°C')
    else:
        overall_min_cdo_temp_celsius = float('nan')
    min_cdo_temp = str(round(overall_min_cdo_temp_celsius, 2))
    
    
    if data_subset_cdo.size > 0:
    # Convert all temperatures to Celsius and filter out invalid values
        cdo_temps_pct_kelvin = data_subset_cdo.values.flatten()
        cdo_temps_pct_celsius = cdo_temps_pct_kelvin - 273.15
    
    # Remove any NaN or invalid values
    valid_pct_temps = cdo_temps_pct_celsius[~np.isnan(cdo_temps_pct_celsius)]
    
    if len(valid_pct_temps) > 0:
        # Calculate the -2 sigma percentile
        minustwosigma_cdo_temp = np.percentile(valid_pct_temps, 2.275)
        sigma_cdo_temp_celsius_list.append(minustwosigma_cdo_temp)

    if sigma_cdo_temp_celsius_list:
        # Get the minimum of all the 5th percentile values (if you're processing multiple points)
        overall_sigma_cdo_temp = min(sigma_cdo_temp_celsius_list)
        print('-2σ CDO temp is: ' + str(overall_sigma_cdo_temp) + '°C')
    else:
        overall_sigma_cdo_temp = float('nan')

    sigma_cdo_temp = str(round(overall_sigma_cdo_temp, 2))
    
    plt.imshow(data - 273, origin = 'upper', transform = ccrs.Geostationary(central_longitude = center, satellite_height=35786023.0), vmin = vmin, vmax = vmax, cmap = cmap)
    plt.colorbar(orientation = 'vertical', aspect = 50, pad = .05, label="Fosler-Lussier Enhanced Infrared Brightness Temperature (°C)")
    plt.title(f'GOES-19 Channel 13 Brightness Temperature\nSatellite Image: {time}\nMax Center Temperature: {maxtemp} °C\nMin CDO Temperature: {min_cdo_temp} °C\n-2σ CDO Temperature: {sigma_cdo_temp} °C' , fontweight='bold', fontsize=10, loc='left')
    if lon > -174 and lon < 174:
        plt.title(f'Original code by Deelan Jariwala\n Realtime functionality, cmap\nand eye temp by Elliott Fosler-Lussier', fontsize=10, loc='right')
    plt.savefig(image_folder_path + image_prefix + time + ".png", dpi = 200, bbox_inches = 'tight')
    plt.show()
    plt.close()




for x in range(len(plottimes)):
        try:
            filename = ir.getDataGOES19('19', plotyears[x], plotmonths[x], plotdays[x], plothours[x], '13')
            dataset = xr.open_dataset(base_folder_path + filename + ".nc")
            data = dataset['CMI']
            center = dataset['geospatial_lat_lon_extent'].geospatial_lon_center
            time = (dataset.time_coverage_start).split('T')
            time = f"{time[0]} at {time[1][:5]} UTC"
            year = time[0:4]
            stormir(data, plotlons[x], plotlats[x], 'fozbd')
            os.remove(base_folder_path + filename + ".nc")
        except:
            pass


