from pickle import TRUE
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


class ValidationSet():
    def __init__(self, path=None, projection=None):
        self.path = path
        self.validation = gpd.read_file(path)
        self.projection = None

        if projection != None:
            self.projection = Proj(projection)
            self.validation = self.validation.set_crs(projection, allow_override=True)

        else:
            self.projection = Proj('epsg:4326')
            self.validation = self.validation.set_crs(Proj('epsg:4326'), allow_override=True)

    def reproject_validation(self, lat='lat', lon='lon', outProjection=None):
        newLat, newLon = reproject(self.validation[lat].values, self.validation[lon].values, self.projection, outProjection)
        self.validation[lat], self.validation[lon] = newLat, newLon
        #self.validation = self.validation.set_crs(outProjection, allow_override=True)

    def filter(self):
        self.validation = self.validation[(self.validation.crop_probability == 1) | (self.validation.crop_probability == 0)]

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


class CropMask:
    def __init__(self, path, projection=None) -> None:
        self.path = path
        self.data = ras.open(path)

        if projection is None:
            if self.data.crs == None:
                self.data.crs = Proj(4326)
            else:
                self.projection = self.data.crs
        else:  
            self.projection = Proj(projection)

    def sample_points(self, lat, lon):
        if lat.size != lon.size:
            raise Exception("Input arrays are not of equal size.")
        coord_list = [(float(x),float(y)) for x,y in zip(lat, lon)]
        extracted_values = self.data.sample(coord_list)

        return pd.Series([x for x in extracted_values])

    def getProjection(self):
        return self.projection
    
    def getPath(self) -> str:
        return str(self.path)
    
    def getData(self) -> ras:
        return self.data

    def getName(self) -> str:
        return self.data.name


class MatchedSet():
    def __init__(self, cropmask, validation_data) -> None:
        self.data = None
        self.validation_data = None
        self.cropmask = None

        if not (isinstance(cropmask, CropMask) and isinstance(validation_data, ValidationSet)):
            raise ValueError("Arguments not type CropMask or ValidationSet.")
        else:
            self.validation_data = validation_data
            self.cropmask = cropmask
            
            self.data = pd.DataFrame(self.validation_data.getData().reset_index(drop=True)).astype(float)
            self.data['cropmask'] = (cropmask.sample_points(validation_data.validation.lat, validation_data.validation.lon)).astype(float)
        
    def remove_filler(self):
        self.data = self.data[((self.data.crop_probability == 1) | (self.data.crop_probability == 0)) & ((self.data.cropmask == 1) | (self.data.cropmask == 0))]
        #self.data = self.data[((self.data.crop_probability == 1) | (self.data.crop_probability == 0)) and (self.data.cropmask == 1) | (self.data.cropmask == 0))]

    def get_f1(self) -> float:
        #assumes dataset is already filtered 
        return sklearn.metrics.f1_score(self.data['crop_probability'], self.data['cropmask'])
 
    def report(self, show_size=False) -> np.array:
        #assumes dataset is already filtered 

        target_names = ['non_crop', 'crop']
        class_report = sklearn.metrics.classification_report(self.data['crop_probability'], self.data['cropmask'], target_names = target_names, output_dict=True)
        accuracy = sklearn.metrics.accuracy_score(self.data['crop_probability'], self.data['cropmask'])

        report = np.array([self.get_f1(), accuracy, class_report['crop']['precision'], class_report['non_crop']['precision'], class_report['non_crop']['recall'], class_report['crop']['recall']])

        if (show_size == True):
            return np.append(report, int(len(self.data.index)))
        else:
            return report

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
