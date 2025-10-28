import s3fs
from datetime import datetime
import numpy as np
import urllib.request
import bz2
import xarray as xr 
from satpy import Scene


# GRIDSAT-GOES Data Retrieval Function
# Sample usage: getData(13, 2010, 9, 13, 1700)
# This retrieves GOES-13 data on 09/13/2010 at 1700z in the GRIDSAT-GOES dataset
def getDataGridsatGOES(satellite, year, month, day, hour):
    print(f'Downloading GRIDSAT-GOES file for {month}/{day}/{year} at {str(hour).zfill(4)}z')
    try:
        # Create string variable containing the name of the file, and use this to make a link to the GRIDSAT-GOES data
        filename = f'GridSat-GOES.goes{str(satellite).zfill(2)}.{str(year)}.{str(month).zfill(2)}.{str(day).zfill(2)}.{str(hour).zfill(4)}.v01.nc'
        url = f'https://www.ncei.noaa.gov/data/gridsat-goes/access/goes/{str(year)}/{str(month).zfill(2)}/{filename}'

        # Download data using urllib as a file called gridsatgoesfile.nc
        try:
            filename = 'gridsatgoesfile'
            urllib.request.urlretrieve(url, r"/Users/elliott/Documents/Grad Research/gridsatgoesfile.nc")
        except:
            filename = 'gridsatgoesfile2'
            urllib.request.urlretrieve(url, r"/Users/elliott/Documents/Grad Research/gridsatgoesfile2.nc")
    except:
        # Create string variable containing the name of the file, and use this to make a link to the GRIDSAT-GOES data
        filename = f'GridSat-CONUS.goes{str(satellite).zfill(2)}.{str(year)}.{str(month).zfill(2)}.{str(day).zfill(2)}.{str(hour).zfill(4)}.v01.nc'
        url = f'https://www.ncei.noaa.gov/data/gridsat-goes/access/conus/{str(year)}/{str(month).zfill(2)}/{filename}'

        # Download data using urllib as a file called gridsatgoesfile.nc
        try:
            filename = 'gridsatconusfile'
            urllib.request.urlretrieve(url, r"C:/Users/elliott/Documents/Grad Research/gridsatconusfile.nc")
        except:
            filename = 'gridsatconusfile2'
            urllib.request.urlretrieve(url, r"C:/Users/elliott/Documents/Grad Research/gridsatconusfile2.nc")
    
    return filename

# GRIDSAT-B1 Data Retrieval Function
# Sample usage: getData(1998, 10, 13, 9)
# This retrieves GRIDSAT-B1 data on 10/13/1998 at 0900z
def getDataGridsatB1(year, month, day, hour):
    print(f'Downloading GRIDSAT-B1 file for {month}/{day}/{year} at {hour}z')
    # Create string variable containing the name of the file, and use this to make a link to the GRIDSAT-B1 data
    filename = f'GRIDSAT-B1.{str(year)}.{str(month).zfill(2)}.{str(day).zfill(2)}.{str(hour).zfill(2)}.v02r01.nc'
    url = f'https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/{str(year)}/{filename}'
    print(url)

    # Download data using urllib as a file called gridsatb1file.nc
    try:
        filename = 'gridsatb1file'
        urllib.request.urlretrieve(url, r"C:/Users/elliott/Documents/Grad Research/gridsatb1file.nc")
    except:
        filename = 'gridsatb1file2'
        urllib.request.urlretrieve(url, r"C:/Users/elliott/Documents/Grad Research/gridsatb1file2.nc")
    
    return filename

# GOES-R Data Retrieval Function
# Sample usage: getDataGOES(16, 2020, 11, 16, '0900', '13')
# This retrieves GOES-16 Channel 13 infrared data on 11/16/2020 at 0900z
def getDataGOES(satellite, year, month, day, time, band):
    # Log into AWS server using anonymous credentials
    fs = s3fs.S3FileSystem(anon=True)
    fs.ls('s3://noaa-goes16/')

    date = datetime(year, month, day)
    days = date.strftime('%j')
    
    # Retrieve files using the given information, add to a numpy array
    files = np.array(fs.ls(f'noaa-goes{satellite}/ABI-L2-CMIPF/{str(year)}/{days}/{time[0:2]}/'))

    # Loop through array in order to find requested band, add to a new list called 'l'
    l = []
    for x in range(len(files)):
        if f'M6C{band.zfill(2)}' in files[x] or f'M3C{band.zfill(2)}' in files[x]:
            l.append(files[x])
    
    # Loop through l in order to find the file with the matching time
    for x in range(len(l)):
        if time in l[x]:
            file = l[x]

    # Download the file, and rename it to goesfile.nc
    try:
        filename = 'goesfile'
        fs.get(file, r"/Users/elliott/Documents/Grad Research/goesfile.nc")
    except:
        filename = 'goesfile2'
        fs.get(file, r"/Users/elliott/Documents/Grad Research/goesfile2.nc")
    print(f'GOES-{satellite} data downloaded for {month}/{day}/{year} at {time}z')

    return filename


def getDataGOES17(satellite, year, month, day, time, band):
    # Log into AWS server using anonymous credentials
    fs = s3fs.S3FileSystem(anon=True)
    fs.ls('s3://noaa-goes17/')

    date = datetime(year, month, day)
    days = date.strftime('%j')
    
    # Retrieve files using the given information, add to a numpy array
    files = np.array(fs.ls(f'noaa-goes{satellite}/ABI-L2-CMIPF/{str(year)}/{days}/{time[0:2]}/'))

    # Loop through array in order to find requested band, add to a new list called 'l'
    l = []
    for x in range(len(files)):
        if f'M6C{band.zfill(2)}' in files[x] or f'M3C{band.zfill(2)}' in files[x]:
            l.append(files[x])
    
    # Loop through l in order to find the file with the matching time
    for x in range(len(l)):
        if time in l[x]:
            file = l[x]

    # Download the file, and rename it to goesfile.nc
    try:
        filename = 'goesfile'
        fs.get(file, r"/Users/elliott/Documents/Grad Research/goesfile.nc")
    except:
        filename = 'goesfile2'
        fs.get(file, r"/Users/elliott/Documents/Grad Research/goesfile2.nc")
    print(f'GOES-{satellite} data downloaded for {month}/{day}/{year} at {time}z')

    return filename

def getDataGOES18(satellite, year, month, day, time, band):
    # Log into AWS server using anonymous credentials
    fs = s3fs.S3FileSystem(anon=True)
    fs.ls('s3://noaa-goes18/')

    date = datetime(year, month, day)
    days = date.strftime('%j')
    
    # Retrieve files using the given information, add to a numpy array
    files = np.array(fs.ls(f'noaa-goes{satellite}/ABI-L2-CMIPF/{str(year)}/{days}/{time[0:2]}/'))

    # Loop through array in order to find requested band, add to a new list called 'l'
    l = []
    for x in range(len(files)):
        if f'M6C{band.zfill(2)}' in files[x] or f'M3C{band.zfill(2)}' in files[x]:
            l.append(files[x])
    
    # Loop through l in order to find the file with the matching time
    for x in range(len(l)):
        if time in l[x]:
            file = l[x]

    # Download the file, and rename it to goesfile.nc
    try:
        filename = 'goesfile'
        fs.get(file, r"/Users/elliott/Documents/Grad Research/goesfile.nc")
    except:
        filename = 'goesfile2'
        fs.get(file, r"/Users/elliott/Documents/Grad Research/goesfile2.nc")
    print(f'GOES-{satellite} data downloaded for {month}/{day}/{year} at {time}z')

    return filename


def getDataGOES19(satellite, year, month, day, time, band):
    # Log into AWS server using anonymous credentials
    fs = s3fs.S3FileSystem(anon=True)
    fs.ls('s3://noaa-goes18/')

    date = datetime(year, month, day)
    days = date.strftime('%j')
    
    # Retrieve files using the given information, add to a numpy array
    files = np.array(fs.ls(f'noaa-goes{satellite}/ABI-L2-CMIPF/{str(year)}/{days}/{time[0:2]}/'))

    # Loop through array in order to find requested band, add to a new list called 'l'
    l = []
    for x in range(len(files)):
        if f'M6C{band.zfill(2)}' in files[x] or f'M3C{band.zfill(2)}' in files[x]:
            l.append(files[x])
    
    # Loop through l in order to find the file with the matching time
    for x in range(len(l)):
        if time in l[x]:
            file = l[x]

    # Download the file, and rename it to goesfile.nc
    try:
        filename = 'goesfile'
        fs.get(file, r"/Users/elliott/Documents/Grad Research/goesfile.nc")
    except:
        filename = 'goesfile2'
        fs.get(file, r"/Users/elliott/Documents/Grad Research/goesfile2.nc")
    print(f'GOES-{satellite} data downloaded for {month}/{day}/{year} at {time}z')

    return filename
# Each function here downloads a netCDF file that can easily be opened with packages like xarray or netCDF4. 


# GOES-R Data Retrieval Function
# Sample usage: getDataGOES(16, 2020, 11, 16, '0900', '13')
# This retrieves GOES-16 Channel 13 infrared data on 11/16/2020 at 0900z
'''
def getDataHimawari(satellite, year, month, day, time, band):
    # Log into AWS server using anonymous credentials
    fs = s3fs.S3FileSystem(anon=True)
    fs.ls('s3://noaa-himawari/')

    date = datetime(year, month, day)
    days = date.strftime('%j')
    
    # Retrieve files using the given information, add to a numpy array
    files = np.array(fs.ls(f'noaa-himawari/ABI-L1b-FLDK/{str(year)}/{days}/{time[0:2]}/'))

    # Loop through array in order to find requested band, add to a new list called 'l'
    l = []
    for x in range(len(files)):
        if f'M6C{band.zfill(2)}' in files[x] or f'M3C{band.zfill(2)}' in files[x]:
            l.append(files[x])
    
    # Loop through l in order to find the file with the matching time
    for x in range(len(l)):
        if time in l[x]:
            file = l[x]

    # Download the file, and rename it to goesfile.nc
    try:
        filename = 'himawarifile'
        fs.get(file, r"/Users/elliott/Documents/Grad Research/himawarifile.nc")
    except:
        filename = 'himawarifile2'
        fs.get(file, r"/Users/elliott/Documents/Grad Research/himawarifile2.nc")
    print(f'Himawari data downloaded for {month}/{day}/{year} at {time}z')

    return filename
'''



def getDataHimawari8(satellite, year, month, day, time, band):
    # Log into AWS server using anonymous credentials
    fs = s3fs.S3FileSystem(anon=True)
    
    # Format date as YYYYMMDD for Himawari
    '''
    date = datetime(year, month, day)
    date_str = date.strftime('%Y%m%d')  # Himawari uses YYYYMMDD format
    hour = time[0:2]
    '''
    # Himawari-9 specific path and naming
    bucket_path = f'noaa-himawari8/AHI-L1b-FLDK/{str(year)}/{month}/{day}/{time}/'
    
    try:
        files = np.array(fs.ls(bucket_path))
        
        # Create a directory to store temporary files if it doesn't exist
        import os
        temp_dir = "/Users/elliott/Documents/Grad Research/temp_himawari/"
        os.makedirs(temp_dir, exist_ok=True)
        
        # List to store paths of downloaded files
        downloaded_files = []
        base_pattern = f'HS_H08_{year}{month}{day}_{time}'
        
        # Download all slices
        for slice_num in range(1, 11):
            if int(year) <= 2019:
                slice_pattern = f'{base_pattern}_B13_FLDK_R20_S{slice_num:02d}01.DAT.bz2'
            if int(year) >= 2020:
                slice_pattern = f'{base_pattern}_B13_FLDK_R20_S{slice_num:02d}10.DAT.bz2'
            matching_files = [f for f in files if slice_pattern in f]
            
            if matching_files:
                file = matching_files[0]
                try:
                    # Download and decompress
                    with fs.open(file, 'rb') as f_in:
                        compressed_data = f_in.read()
                    decompressed_data = bz2.decompress(compressed_data)
                    
                    # Save decompressed data with .DAT extension
                    output_file = os.path.join(temp_dir, f'{os.path.basename(file)[:-4]}')
                    with open(output_file, 'wb') as f_out:
                        f_out.write(decompressed_data)
                    
                    downloaded_files.append(output_file)
                    
                except Exception as e:
                    print(f"Error downloading slice {slice_num}: {e}")
        
        if not downloaded_files:
            print("No files were downloaded successfully")
            return None
        
        # Use satpy to read the files
        try:
            scn = Scene(filenames=downloaded_files, reader='ahi_hsd')
            scn.load([f'B{band}'])
            new_scn = scn.resample('himawari_ahi_fes_2km')
            # Save the data directly from the scene
            filename = 'himawari8file.nc'
            # Get the dataset and save it
            dataset = scn[f'B{band}']
            dataset.attrs['area'] = None  # Remove area definition which might cause issues
            new_scn.save_dataset(dataset_id='B13', writer='cf', filename=f"/Users/elliott/Documents/Grad Research/{filename}")
            
                # Clean up temporary files
            for file in downloaded_files:
                try:
                    os.remove(file)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
                
            print(f'Himawari data downloaded and processed for {month}/{day}/{year} at {time}z')
            return filename
            
        except Exception as e:
            print(f"Error processing files with satpy: {e}")
            return None
            
    except Exception as e:
        print(f"Error accessing Himawari data: {e}")
        print(f"Attempted to access: {bucket_path}")
        return None
    
    

def getDataHimawari9(satellite, year, month, day, time, band):
    # Log into AWS server using anonymous credentials
    fs = s3fs.S3FileSystem(anon=True)
    
    # Format date as YYYYMMDD for Himawari
    '''
    date = datetime(year, month, day)
    date_str = date.strftime('%Y%m%d')  # Himawari uses YYYYMMDD format
    hour = time[0:2]
    '''
    # Himawari-9 specific path and naming
    bucket_path = f'noaa-himawari9/AHI-L1b-FLDK/{str(year)}/{month}/{day}/{time}/'
    
    try:
        files = np.array(fs.ls(bucket_path))
        
        # Create a directory to store temporary files if it doesn't exist
        import os
        temp_dir = "/Users/elliott/Documents/Grad Research/temp_himawari/"
        os.makedirs(temp_dir, exist_ok=True)
        
        # List to store paths of downloaded files
        downloaded_files = []
        base_pattern = f'HS_H09_{year}{month}{day}_{time}'
        
        # Download all slices
        for slice_num in range(1, 11):
            slice_pattern = f'{base_pattern}_B13_FLDK_R20_S{slice_num:02d}10.DAT.bz2'
            matching_files = [f for f in files if slice_pattern in f]
            
            if matching_files:
                file = matching_files[0]
                try:
                    # Download and decompress
                    with fs.open(file, 'rb') as f_in:
                        compressed_data = f_in.read()
                    decompressed_data = bz2.decompress(compressed_data)
                    
                    # Save decompressed data with .DAT extension
                    output_file = os.path.join(temp_dir, f'{os.path.basename(file)[:-4]}')
                    with open(output_file, 'wb') as f_out:
                        f_out.write(decompressed_data)
                    
                    downloaded_files.append(output_file)
                    
                except Exception as e:
                    print(f"Error downloading slice {slice_num}: {e}")
        
        if not downloaded_files:
            print("No files were downloaded successfully")
            return None
        
        # Use satpy to read the files
        try:
            scn = Scene(filenames=downloaded_files, reader='ahi_hsd')
            scn.load([f'B{band}'])
            new_scn = scn.resample('himawari_ahi_fes_2km')
            # Save the data directly from the scene
            filename = 'himawari9file.nc'
            # Get the dataset and save it
            dataset = scn[f'B{band}']
            dataset.attrs['area'] = None  # Remove area definition which might cause issues
            new_scn.save_dataset(dataset_id='B13', writer='cf', filename=f"/Users/elliott/Documents/Grad Research/{filename}")
            
                # Clean up temporary files
            for file in downloaded_files:
                try:
                    os.remove(file)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
                
            print(f'Himawari data downloaded and processed for {month}/{day}/{year} at {time}z')
            return filename
            
        except Exception as e:
            print(f"Error processing files with satpy: {e}")
            return None
            
    except Exception as e:
        print(f"Error accessing Himawari data: {e}")
        print(f"Attempted to access: {bucket_path}")
        return None




