from pyproj import CRS, Transformer
import pandas as pd


crs = CRS.from_epsg(4326)
crs = CRS.from_string("epsg:4326")
crs = CRS.from_proj4("+proj=latlon")
crs = CRS.from_user_input(4326)

crs_cs = CRS.from_epsg(32648)
transformer = Transformer.from_crs(crs,crs_cs)

data = pd.read_csv('ex.csv')
lon = data.lon.values
lat = data.lat.values

transformer.transform(lat,lon)