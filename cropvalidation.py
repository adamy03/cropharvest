from pkgutil import get_data
from typing_extensions import Self
from idna import valid_contextj
import rasterio as ras
from rasterio.crs import CRS
import geopandas as gpd
import pandas as pd
import numpy as np 
from pyproj import Proj, transform
import sklearn.metrics


"""
Stores and manipulates validation data.
"""
class ValidationSet():
    def __init__(self, path=None, projection=None):
        self.path = path
        self.validation = gpd.read_file(path)
        self.projection = None

        if projection != None:
            self.projection = projection
            self.validation = self.validation.set_crs(projection)

        else:
            self.projection = Proj('epsg:4326') #defaults to epsg 4326 if no projection is provided
            self.validation = self.validation.set_crs(Proj('epsg:4326'))

    def reproject_validation(self, lat='lat', lon='lon', outProjection=None):
        # Reprojects validation to a given epsg projection
        newLat, newLon = reproject(self.validation[lat], self.validation[lon], self.getProjection(), outProjection)
        self.validation[lat], self.validation[lon] = newLat, newLon
        self.validation = self.validation.to_crs(outProjection)

    def getProjection(self):
        return self.projection

    def getPath(self) -> str:
        return str(self.path)
    
    def getData(self):
        return self.validation

    def getLat(self):
        return self.validation['lat']
    
    def getLon(self):
        return self.validation['lon']

"""
CropMask wrapper, given as raster/tiff file. 
"""
class CropMask:
    def __init__(self, path, projection=None) -> None:
        self.path = path
        self.data = ras.open(path)

        if projection is None:
            if self.projection is None:
                self.projection = "epsg:4326"
            else:
                self.projection = self.data.crs 
        else:  
            self.projection = Proj(projection)


    def sample_points(self, lat, lon) -> np.array:
        #samples points from a given set of coords and outputs coords and data points
        if lat.size != lon.size:
            raise Exception("Input arrays are not of equal size.")
        coord_list = [(float(x),float(y)) for x,y in zip(lat, lon)]
        extracted_values = self.data.sample(coord_list)

        return pd.DataFrame(zip(lat, lon, list(extracted_values)))

    def getProjection(self):
        return self.projection
    
    def getPath(self) -> str:
        return str(self.path)
    
    def getData(self) -> ras:
        return self.data

    def getName(self) -> str:
        return self.data.name

"""
Paired datastructure for validation and cropmask matches
"""
class MatchedSet():
    def __init__(self, cropmask, validation_data) -> None:
        if not (isinstance(cropmask, CropMask) and isinstance(validation_data, ValidationSet)):
            raise ValueError("Arguments not type CropMask or ValidationSet.")
        else:
            self.validation_data = validation_data
            self.cropmask = cropmask
            
            self.data = pd.DataFrame(validation_data.getData()).astype(float)
            self.data['cropmask'] = cropmask.sample_points(validation_data.getLat(), validation_data.getLon())[2].astype(float)
        
    def remove_filler(self):
        self.data = self.data.loc[self.data['crop_probability'].isin([0,1]) & self.data['cropmask'].isin([0,1])]

    def get_f1(self) -> float:
        #assumes dataset is already filtered 
        return sklearn.metrics.f1_score(self.data['crop_probability'], self.data['cropmask'])
    
    def getData(self):
        return self.data


def reproject(latIn, lonIn, ProjIn, ProjOut) -> np.array:
    if latIn.size != latIn.size:
        raise Exception("Input arrays are not of equal size.")

    latOut = []
    lonOut = []

    for coordinate in np.arange(latIn.size):
        x1,y1 = latIn[coordinate],lonIn[coordinate]
        newLat, newLon = transform(ProjIn,ProjOut,x1,y1)
        latOut.append(newLat), lonOut.append(newLon)
    
    return np.array(latOut), np.array(lonOut)
