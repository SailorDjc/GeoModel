# GeoModel
GraphSAGE  GeoModelling
## 1. Dependent Environment Installation Instructions
* pytorch
* dgl
* pyvista
* vtk
* scikit-learn
* pandas
* torchmetrics
* matplotlib
* xgboost
* pytest
* pynoddy
* imageio
* rdp
* openpyxl
* Cython==0.29.35
* pytetgen
* pyproj
* Shapely
* GDAL
* Fiona
* geopandas
* rasterio <br>
Note: <br>
1.This project uses python 3.8 environment, cuda11.6  <br>
2.The pytorch installation command is：conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia   <br>
3.The dgl installation command is：conda install -c dglteam/label/cu116 dgl  <br>
4.GDAL,Fiona,rasterio need to download the whl installation package for offline installation, the download URL is https://www.cgohlke.com/, in the selection of whl package, select the highest version of the library that specifies the version of python, and note that the package suffix is cp, which means the use of the CPython implementation, such as GDAL-3.4. 3-cp38-cp38-win_amd64.whl.
5.Other dependencies can be installed in turn using pip install. <br>
## Instructions for running the code
```
python run examples/geomodel_example.py
```
## Geological modeling data
The borehole data file is the borehole_data.dat file stored in the data folder. The borehole file format is divided into five columns of records: id, x, y, z, label.
The borehole_data.map file is the stratigraphic sequence information.
