import copy
import pyvista as pv
import numpy as np
from vtkmodules.util import numpy_support
from vtkmodules.all import vtkXMLUnstructuredGridReader, vtkPoints, vtkCellArray, vtkTriangle, vtkUnstructuredGrid, \
    vtkImageData, vtkPlaneSource
from data_structure.grids import PointSet
from tqdm import tqdm
from utils.vtk_utils import create_vtk_grid_by_rect_bounds, vtk_unstructured_grid_to_vtk_polydata, \
    create_polygon_with_sorted_points_3d
from utils.vtk_utils import create_implict_surface_reconstruct, voxelize
from utils.math_libs import bounds_intersect, get_bounds_from_coords, bounds_merge, extend_bounds
import os
import json
import time
import pickle
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, Rbf, splprep, RBFInterpolator
import scipy.spatial as spt
from importlib import util
from utils.math_libs import simplify_polyline_2d, remove_duplicate_points
import random

random.seed(1)
from matplotlib.path import Path
from vtkmodules.all import vtkPolyDataReader, vtkPolyDataMapper, vtkProperty, vtkRenderer, \
    vtkBooleanOperationPolyDataFilter
import matplotlib.pyplot as plt

has_shapely = util.find_spec("shapely")
has_rasterio = util.find_spec("rasterio")
has_geopandas = util.find_spec("geopandas")

if has_rasterio is None:
    has_rasterio = False
else:
    has_rasterio = True
    import rasterio
    import rasterio as rio
    from rasterio.warp import calculate_default_transform, reproject, transform_geom, Resampling
    from rasterio import crs
    import rasterio.features
    from rasterio.mask import mask
if has_shapely is None:
    has_shapely = False
else:
    has_shapely = True
    from shapely.geometry import box, Polygon, LineString
if has_geopandas is None:
    has_geopandas = False
else:
    has_geopandas = True
    import geopandas as gpd


# epsg code
def longitude_to_proj_zone(longitude, zone_type='6'):
    if zone_type == '3':
        zone_no = np.floor(longitude / 3 + 0.5)
        if 25 <= zone_no <= 45:
            return 4534 + (zone_no - 25)
        else:
            raise ValueError('out of the expected range.')
    else:
        zone_no = np.floor(longitude / 6) + 1
        if 13 <= zone_no <= 23:
            return 4502 + (zone_no - 13)
        else:
            raise ValueError('out of the expected range.')


# 2d polygon
def create_polygon_from_boundary(boundary_2d):
    points_2d = []
    for item in boundary_2d:
        points_2d.append((item[0], item[1]))
    boundary = Polygon(points_2d)
    return boundary


# StructuredGrid
def create_struct_mesh_from_bounds(bounds, resolution_xy):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    dim_x = np.ceil((x_max - x_min) / resolution_xy)
    dim_y = np.ceil((y_max - y_min) / resolution_xy)
    x_d, y_d = dim_x + 1, dim_y + 1
    x_d = complex(0, x_d)
    y_d = complex(0, y_d)
    x, y = np.mgrid[x_min:x_max:x_d, y_min:y_max:y_d]
    z = np.zeros_like(x)
    surface = pv.StructuredGrid(x, y, z)
    return surface


# 包围盒转点
# 由于目前很多python插值算法只能插值凸包范围内的点，
# 故通过内部点集iner_points计算bounds角点高程，取最近邻3个点的高程均值
def bounds_to_corners_2d(bounds, inner_points=None):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    z_value = (z_min + z_max) / 2
    p_a = [x_min, y_min, z_value]
    p_b = [x_min, y_max, z_value]
    p_c = [x_max, y_min, z_value]
    p_d = [x_max, y_max, z_value]
    corners_list = [p_a, p_b, p_c, p_d]
    if inner_points is not None:

        ckt = spt.cKDTree(inner_points)
        for p_i, pt in enumerate(corners_list):
            d, pid = ckt.query(pt, k=[1, 2])
            search_pts = inner_points[pid]
            z_value = np.mean(search_pts[:, 2])
            corners_list[p_i][2] = z_value
    return np.array(corners_list)


# 从二维点集中获取二维包围盒
def get_bound_2d_from_points_2d(points_2d, buffer_dist=5):
    points = np.array(points_2d)
    min_x = np.min(points[:, 0]) - buffer_dist
    max_x = np.max(points[:, 0]) + buffer_dist
    min_y = np.min(points[:, 1]) - buffer_dist
    max_y = np.max(points[:, 1]) + buffer_dist
    return np.array([min_x, max_x, min_y, max_y, 0, 0])


# 比较扩大后的范围与数据源范围，如果超出数据源范围，则剪断
def compare_data_bounds(tmp_bounds, data_bounds):
    x_min_a, x_max_a, y_min_a, y_max_a, z_min_a, z_max_a = tmp_bounds
    x_min_b, x_max_b, y_min_b, y_max_b, z_min_b, z_max_b = data_bounds
    if x_min_a < x_min_b:
        x_min_a = x_min_b
    if x_max_a > x_max_b:
        x_max_a = x_max_b
    if y_min_a < y_min_b:
        y_min_a = y_min_b
    if y_max_a > y_max_b:
        y_max_a = y_max_b
    if z_min_a < z_min_b:
        z_min_a = z_min_b
    if z_max_a > z_max_b:
        z_max_a = z_max_b
    return np.array([x_min_a, x_max_a, y_min_a, y_max_a, z_min_a, z_max_a])


# 地形曲面生成类，可以通过输入tiff文件和钻孔顶点来构建地形曲面，或者采用tiff和钻孔地表控制点联合约束的方法，钻孔顶点用来修正tiff局部高程。
# 输入tiff:     set_input_tiff_file()
# 输入控制点:    set_control_points()
# 输入边界约束:   set_boundary()

class TerrainData(object):
    def __init__(self, tiff_path=None, control_points=None, extend_buffer=0, top_surface_offset=0, is_bound=True):
        # 地形面数据
        self.grid_points = None  # 与dem对应栅格点坐标点坐标
        self._bounds = None
        self._center = None
        self.is_rtc = False  # 是否为相对坐标
        # tiff 元数据
        self.tiff_path = tiff_path
        self._transform = None
        self._src_crs = None  # 原始投影
        self._tiff_dims = None  # 数字高程栅格维度
        self.src_tiff_bounds = None  # tiff数据范围
        self.surface_offset = top_surface_offset
        # 高程控制点
        self.control_points = control_points

        self.buffer_dist = 20
        self.dst_crs = None  # 目标投影
        self.vtk_data = None  # 顶面数据
        # 边界约束
        self.mask_bounds = None  # 规则矩形边界约束
        self.mask_shp_path = None  # shp矢量文件不规则边界约束
        self.boundary_2d = None  # 点集列表不规则边界约束

        # 底面数据
        self.bottom_control_points = None
        self.bottom_vtk_data = None

        #
        self.extend_buffer = extend_buffer  # 建模范围向外扩展
        self.boundary_points = None  # 边界点集，首尾不重复
        self.boundary_type = 4  # 边界类型 0为规则矩形掩膜边界范围，1为不规则多边形边界约束(输入点集)，2为shp多边形边界输入，
        # 当没有外界输入边界，默认为3或4，3为以点集的外包络矩形范围做约束，4为以点集凸包为边界约束
        #
        self.coord_type = 'xy'  # 'xy' 平面直角坐标系（单位米）   'dd' 经纬度
        # 保存参数
        self.dir_path = None
        self.tmp_dump_str = 'tmp_terrain' + str(int(time.time()))
        if self.tiff_path is not None:
            self.set_input_tiff_file(self.tiff_path)
        self.interp = None

    def is_empty(self):
        empty_flag = True
        if self.control_points is not None:
            empty_flag = False
        if self.tiff_path is not None:
            empty_flag = False
        return empty_flag

    # 数据输入接口
    def set_control_points(self, control_points, buffer_dist=20):
        self.buffer_dist = buffer_dist
        if control_points is not None:
            if not isinstance(control_points, PointSet):
                control_points = PointSet(points=control_points)
            self.control_points = control_points

    def set_bottom_control_points(self, control_points):
        if control_points is not None:
            if not isinstance(control_points, PointSet):
                control_points = PointSet(points=control_points)
            self.bottom_control_points = control_points

    # boundary_2d是边界点按顺序排列的列表，首尾点不重复， mask_bounds 包围盒6元组，
    # mask_shp_path shp文件
    # mask_bounds
    #
    # boundary_type: 0, 1, 2, 3, 4
    # is_bound:
    def set_boundary(self, boundary_2d=None, mask_bounds=None, mask_shp_path=None, is_bound=False):
        self.boundary_2d = boundary_2d
        self.mask_bounds = mask_bounds
        self.mask_shp_path = mask_shp_path
        if self.mask_bounds is not None:
            min_x, max_x, min_y, max_y, _, _ = self.mask_bounds
            self.boundary_points = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
            self.boundary_type = 0
        elif self.mask_shp_path is not None:
            mask_gdf = gpd.read_file(self.mask_shp_path)
            x, y = mask_gdf['geometry'][0].exterior.coords
            self.boundary_points = np.array([x, y]).transpose()
            self.boundary_type = 1
        elif self.boundary_2d is not None:
            self.boundary_points = np.array(self.boundary_2d)
            self.boundary_type = 2
        elif is_bound is True:
            self.boundary_type = 3
        else:
            self.boundary_type = 4

    # 设置线状工程边界，通过线性缓冲区设置边界
    def set_boundary_from_line_buffer(self, trajectory_line_xy, buffer_dist=30, principal_axis='x'):
        line_xy_sorted = simplify_polyline_2d(polyline_points=trajectory_line_xy, principal_axis=principal_axis, eps=1)
        lineStrip = LineString(line_xy_sorted)
        buffer_area = lineStrip.buffer(distance=buffer_dist, cap_style='square')
        x, y = buffer_area.exterior.xy
        self.boundary_2d = np.array([x, y]).transpose()
        self.set_boundary(boundary_2d=self.boundary_2d)

    # 设置栅格影像输入
    def set_input_tiff_file(self, file_path, dst_crs_code=None):
        self.tiff_path = file_path
        if not os.path.exists(self.tiff_path):
            raise ValueError('Input path not exists.')
        # 预加载图像元数据
        with rasterio.open(self.tiff_path) as src:
            self._tiff_dims = (src.width, src.height)
            self._src_crs = src.crs
            self._transform = src.transform
        if dst_crs_code is not None:
            self.dst_crs = self.set_dst_crs_code(dst_crs_code=dst_crs_code)
        self.set_default_dst_crs()
        # 重投影
        if self.check_crs_change():
            self.reproject_tiff(self.tiff_path)

    def execute(self, is_rtc=False, clip_reconstruct=False, simplify=None, top_surf_offset=None, resolution_xy=5
                , extend_buffer=0):
        if extend_buffer > 0:
            self.extend_buffer = extend_buffer
        if top_surf_offset is not None:
            self.surface_offset = top_surf_offset
        if self.tiff_path is not None:
            self.read_tiff_data_from_file(file_path=self.tiff_path)
        if self.control_points is not None:
            self.append_control_points(self.control_points)
        if is_rtc:
            self.compute_relative_points()
        self.vtk_data, self.grid_points, self.interp = \
            self.create_terrain_surface_from_points(PointSet(points=self.grid_points), resolution_xy=resolution_xy
                                                    , surface_type='top')
        if self.bottom_control_points is not None:
            self.bottom_vtk_data, _, _ = self.create_terrain_surface_from_points(self.bottom_control_points
                                                                                 , resolution_xy=resolution_xy
                                                                                 , surface_type='bottom')
        if not self.boundary_type == 0 or self.boundary_type == 3:
            self.vtk_data = self.clip_terrain_surface_by_boundary_points(self.vtk_data, self.boundary_points,
                                                                         clip_reconstruct=clip_reconstruct,
                                                                         simplify=simplify)

    @property
    def bounds(self):
        cur_bounds = None
        if self.grid_points is not None:
            tmp_bounds = get_bounds_from_coords(self.grid_points)
            cur_bounds = bounds_merge(tmp_bounds, cur_bounds)
            self._bounds = bounds_merge(cur_bounds, self._bounds)
        elif self.boundary_points is not None:
            tmp_bounds = get_bounds_from_coords(self.boundary_points)
            cur_bounds = bounds_merge(tmp_bounds, cur_bounds)
            self._bounds = bounds_merge(cur_bounds, self._bounds)
        elif self.control_points is not None:
            tmp_bounds = get_bounds_from_coords(self.control_points)
            cur_bounds = bounds_merge(tmp_bounds, cur_bounds)
            self._bounds = bounds_merge(cur_bounds, self._bounds)
        else:
            self._bounds = None
        return self._bounds

    @property
    def center(self):
        if self.bounds is not None and self.is_rtc is False:
            x = (self.bounds[0] + self.bounds[1]) * 0.5
            y = (self.bounds[2] + self.bounds[3]) * 0.5
            z = (self.bounds[4] + self.bounds[5]) * 0.5
            self._center = np.array([x, y, z])
        return self._center

    # 重投影栅格影像
    def reproject_tiff(self, file_path):
        with rio.open(file_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, self.dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': self.dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            tiff_dir_path, tiff_name = os.path.split(self.tiff_path)
            tiff_name = 'reproj_' + tiff_name
            write_path = os.path.join(tiff_dir_path, tiff_name)
            with rasterio.open(write_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=self.dst_crs,
                        resampling=Resampling.nearest)
            self.tiff_path = write_path

    def read_tiff_data_from_file(self, file_path):
        with rio.open(file_path) as src:
            self.grid_points = []
            self._tiff_dims = (src.width, src.height)
            self._src_crs = src.crs
            self.src_tiff_bounds = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top, 0, 0]
            self._transform = src.transform
            z_matrix = src.read(1)
            if self.boundary_points is not None:
                # 在已有掩膜范围的基础上，扩大范围，搜索更多范围外的点
                bounds_2d = get_bound_2d_from_points_2d(self.boundary_points, buffer_dist=200)
                # 与数据源范围作比较，超出范围部分剪断
                bounds_2d = compare_data_bounds(bounds_2d, self.src_tiff_bounds)
                bounds_points_2d = bounds_to_corners_2d(bounds_2d)
                polygon_2d = create_polygon_from_boundary(boundary_2d=bounds_points_2d)
                out_image, out_transform = mask(dataset=src, shapes=[polygon_2d], crop=True, nodata=-32768)
                self._transform = out_transform
                z_matrix = out_image[0]
                self._tiff_dims = [z_matrix.shape[0], z_matrix.shape[1]]
            print('Processing the tiff matrix...')
            pbar = tqdm(range(z_matrix.shape[0]), total=z_matrix.shape[0], desc='Reading coordinates...')
            for i in pbar:
                for j in range(z_matrix.shape[1]):
                    x, y = (i + 0.5, j + 0.5) * self._transform
                    # # 排除无效值
                    if z_matrix[i, j] == -32768:
                        continue
                    self.grid_points.append(np.array([x, y, z_matrix[i, j]]))
            self.grid_points = np.array(self.grid_points)

    def append_control_points(self, control_points):
        if control_points is not None:
            self.modify_local_terrain_by_points(control_points, buffer_dist=self.buffer_dist)

    # 设置目标投影
    def set_dst_crs_code(self, dst_crs_code):
        if self.dst_crs is not None:
            if self.dst_crs.data['init'] == 'epsg:{}'.format(dst_crs_code):
                return False
            self.dst_crs = crs.CRS().from_epsg(code=dst_crs_code)
            return True
        else:
            self.dst_crs = crs.CRS().from_epsg(code=dst_crs_code)
            return True

    # 检查原来投影与目标投影间是否发生变化
    def check_crs_change(self):
        if self.dst_crs is None or self._src_crs is None:
            return False
        if self._src_crs.data['init'] == self.dst_crs.data['init']:
            return False
        else:
            return True

    # 设置默认的投影坐标系，平面坐标系默认为大地2000, 经纬度默认EPSG:4326
    def set_default_dst_crs(self):
        if self.dst_crs is None:
            if self.coord_type == 'xy':
                if not self._src_crs.is_projected:
                    epsg_code = longitude_to_proj_zone(longitude=self._transform.c)
                    self.dst_crs = crs.CRS().from_epsg(code=epsg_code)
            if self.coord_type == 'dd':
                if self._src_crs.is_projected:
                    self.dst_crs = crs.CRS().from_epsg(code=4326)

    # 生成地形曲面
    def create_terrain_surface_from_points(self, points_data: PointSet, resolution_xy=5, surface_type='top'
                                           , kernel='thin_plate_spline', smoothing=1):
        tmp_bounds = points_data.bounds
        result_bounds = bounds_merge(self.bounds, tmp_bounds)
        result_bounds = extend_bounds(result_bounds, buffer_xy=self.extend_buffer)
        self._bounds = result_bounds
        if self.boundary_type == 0 or self.boundary_type == 1 or self.boundary_type == 2:
            result_bounds = get_bound_2d_from_points_2d(self.boundary_points, buffer_dist=self.extend_buffer)
        if self.boundary_type == 3:
            result_bounds = get_bound_2d_from_points_2d(points_data.points, buffer_dist=self.extend_buffer)
        if self.boundary_type ==\
                4:  # 凸包
            self.boundary_points = points_data.get_convexhull_2d(extend_buffer=self.extend_buffer)
        terrain_surface = create_struct_mesh_from_bounds(bounds=result_bounds, resolution_xy=resolution_xy)
        known_points = points_data.points
        if points_data.nidm == 2:
            if isinstance(points_data.scalars, dict):
                z_values = points_data.scalars.get('elevation')
                if z_values is not None:
                    known_points = np.concatenate((known_points, z_values), axis=1)
                else:
                    raise ValueError('Points have no z_value data.')
        # 插值
        b_points = bounds_to_corners_2d(result_bounds, inner_points=known_points)
        # 将角点加入已知点集，角点高程是均值填充的
        known_points = np.concatenate((known_points, b_points), axis=0)
        x = known_points[:, 0]
        y = known_points[:, 1]
        z = known_points[:, 2]
        print('Terrain is being built...')
        interp = RBFInterpolator(list(zip(x, y, )), z, kernel=kernel, smoothing=smoothing)  #

        # interp = LinearNDInterpolator(list(zip(x, y)), z)
        cell_points = terrain_surface.points
        pred_x = cell_points[:, 0]
        pred_y = cell_points[:, 1]
        pred_z = interp(list(zip(pred_x, pred_y)))
        if self.surface_offset != 0 and surface_type == 'top':
            pred_z[:] += self.surface_offset
        if self.surface_offset != 0 and surface_type == 'bottom':
            pred_z[:] -= 3
        terrain_surface['Elevation'] = pred_z
        terrain_surface = terrain_surface.warp_by_scalar()
        terrain_surface = vtk_unstructured_grid_to_vtk_polydata(terrain_surface)
        grid_points = terrain_surface.cell_centers().points  # self.grid_points
        # terrain_surface.plot()
        terrain_surface.clear_data()
        # terrain_surface.plot()
        # self.vtk_data = terrain_surface
        return terrain_surface, grid_points, interp

    # 根据边界切割地形曲面
    # simplify: float 值在0-1之间，地形面三角形简化，降低地形面三角形数量
    def clip_terrain_surface_by_boundary_points(self, vtk_data, boundary_points, clip_reconstruct=False
                                                , simplify=None, surface_type='top', is_bounds=False):
        # if self.vtk_data is None:
        #     raise ValueError('The vtk surface is None.')
        if vtk_data is None:
            raise ValueError('The vtk surface is None.')
        max_z = self.bounds[5]
        min_z = self.bounds[4]
        z_max = max_z + 5 + self.surface_offset
        z_min = min_z - 5
        # boundary_type 4 or 1 or 2:
        boundary_points = copy.deepcopy(boundary_points)
        # 范围约束没有高程设置，这里使用最大高程和最小高程，创建一个范围盒子，筛选出在盒子范围内的网格点，范围外的网格删除
        if boundary_points.shape[1] == 2:
            boundary_points = np.pad(array=boundary_points, pad_width=((0, 0), (0, 1))
                                     , constant_values=((z_max, z_max), (z_max, z_max)))
        else:
            boundary_points[:, 2] = z_max

        points_3d = copy.deepcopy(boundary_points)
        points_3d, rids = remove_duplicate_points(points_3d, tolerance=10, is_remove=True)
        polygon = create_polygon_with_sorted_points_3d(points_3d=points_3d)

        print('Terrain surfaces are being cropped by range...')
        side_surf = polygon.extrude((0, 0, z_min - z_max), capping=True)
        # 筛选出在边界范围内的网格点索引
        grid_points_list = vtk_data.cell_centers().points
        grid_points = pv.PolyData(grid_points_list)
        selected = grid_points.select_enclosed_points(side_surf, tolerance=0.000000001)
        cell_indices = selected.point_data['SelectedPoints']
        if clip_reconstruct:
            if self.interp is None:
                raise ValueError('Error, cannot capture boundary points.')
            pred_x = points_3d[:, 0]
            pred_y = points_3d[:, 1]
            pred_z = self.interp(list(zip(pred_x, pred_y)))
            points_3d[:, 2] = pred_z
            insert_cell_indices = np.argwhere(cell_indices > 0).flatten()
            if simplify is not None:
                points_num = len(insert_cell_indices)
                sample_num = int(points_num * simplify)
                if sample_num > 50:
                    # 随机筛选网格点
                    pid = random.sample(list(np.arange(points_num)), sample_num)
                    insert_cell_indices = insert_cell_indices[pid]
            insert_points = grid_points_list[insert_cell_indices]
            surface_points = np.vstack((points_3d, insert_points))
            if self.surface_offset != 0 and surface_type == 'top':
                surface_points[:, 2] += self.surface_offset
            vtk_data = pv.PolyData(surface_points).delaunay_2d()
        else:
            delete_cell_indices = np.argwhere(cell_indices <= 0).flatten()
            vtk_data = vtk_data.remove_cells(delete_cell_indices)
        return vtk_data

    # 分段裁剪, 按轴向裁剪
    def clip_segment_by_axis(self, mask_bounds, seg_axis='x'):
        if self.vtk_data is None:
            return None
        axis_labels = ['x', 'y']
        label_to_index = {label: index for index, label in enumerate(axis_labels)}
        #
        origin_bounds = get_bounds_from_coords(coords=self.boundary_points)
        if seg_axis.lower() in axis_labels:
            axis_index = label_to_index[seg_axis.lower()]
            l_0 = mask_bounds[2 * axis_index]
            l_1 = mask_bounds[2 * axis_index + 1]

            grid_points = self.vtk_data.cell_centers().points
            del_inds = np.argwhere((grid_points[:, axis_index] < l_0)
                                   | (grid_points[:, axis_index] > l_1)).flatten()
            if len(del_inds) > 0:
                vtk_data = self.vtk_data.remove_cells(del_inds)
                return vtk_data
            else:
                return self.vtk_data

    # 通过地表曲面，构建延展到指定深度的建模网格
    def create_grid_from_terrain_surface(self, z_min=None, boundary_2d=None, bounds=None, cell_density=None
                                         , is_smooth=False, external_surface=None, only_closed_poly_surface=False):
        if boundary_2d is not None:
            self.set_boundary(boundary_2d=boundary_2d, mask_bounds=bounds)
        vtk_data = self.vtk_data
        if external_surface is not None:
            vtk_data = external_surface
        if vtk_data is None:
            raise ValueError('The vtk surface is None.')
        size_x = self.bounds[1] - self.bounds[0] + 5
        size_y = self.bounds[3] - self.bounds[2] + 5
        if z_min is None:
            z_min = self.bounds[4] - 3
        if z_min > self.bounds[4]:
            raise ValueError('Model depth must be less than the minimum height of top surface.')
        plane = pv.Plane(center=(vtk_data.center[0], vtk_data.center[1], z_min), direction=(0, 0, -1)
                         , i_size=size_x, j_size=size_y)
        grid_extrude_trim = vtk_data.extrude_trim((0, 0, z_min), trim_surface=plane).clean()

        if only_closed_poly_surface:
            return grid_extrude_trim
        if cell_density is None:
            raise ValueError('cell_density can not be None.')
        print('voxelize...')
        print(f'cell_density is {cell_density}...')
        grid_extrude_trim = voxelize(grid_extrude_trim, density=cell_density)
        # grid_extrude_trim.plot()
        if self.bottom_vtk_data is not None:
            # 将多余的体素裁剪掉
            plane_1 = pv.Plane(center=(vtk_data.center[0], vtk_data.center[1], z_min - 15), direction=(0, 0, -1)
                               , i_size=size_x, j_size=size_y)
            bottom_grid_extrude_trim = self.bottom_vtk_data.extrude_trim((0, 0, z_min - 15)
                                                                         , trim_surface=plane_1).clean()
            # bottom_grid_extrude_trim.plot()
            grid_points_list = grid_extrude_trim.cell_centers().points
            grid_points = pv.PolyData(grid_points_list)
            selected = grid_points.select_enclosed_points(bottom_grid_extrude_trim, tolerance=1e-12)
            cell_indices = selected.point_data['SelectedPoints']
            restore_cell_indices = np.argwhere(cell_indices <= 0).flatten()
            grid_extrude_trim = grid_extrude_trim.extract_cells(restore_cell_indices)

        if is_smooth:
            grid_extrude_trim = vtk_unstructured_grid_to_vtk_polydata(grid_extrude_trim)
            grid_extrude_trim = grid_extrude_trim.smooth(n_iter=1000)
        return grid_extrude_trim

    # 通过高程点修改局部地形， 注意points_data.buffer_dist，这个缓冲距离表示高程点的控制范围
    def modify_local_terrain_by_points(self, control_points_data: PointSet, buffer_dist):
        if control_points_data is None:
            raise ValueError('Control points is None')
        if self.grid_points is not None:
            tree = spt.cKDTree(self.grid_points)
            erase_points_list = []
            # 球状影响范围搜索，搜索临近点，做替换
            control_points = control_points_data.get_points()
            for con_pnt in control_points:
                ll = tree.query_ball_point(x=con_pnt, r=buffer_dist)
                erase_points_list.extend(ll)
            # 删除影响范围内的点集
            erase_points_list = list(set(erase_points_list))
            if len(erase_points_list) > 0:
                self.grid_points = np.delete(self.grid_points, obj=erase_points_list, axis=0)
            if len(self.grid_points) > 0:
                self.grid_points = np.stack((self.grid_points, control_points), axis=0)
            else:
                self.grid_points = control_points
        else:
            self.grid_points = control_points_data.get_points()
            if self.boundary_points is None:
                self.boundary_points = control_points_data.get_convexhull_2d(extend_buffer=self.extend_buffer)

    # @brief 坐标范围整体偏移
    # @param new_rect_2d和new_center二选一，优先考虑new_center，new_rect_2d除平移外，还涉及坐标伸缩变换
    def translation(self, new_center=None, new_rect_2d=None):
        if self.grid_points is None:
            raise ValueError('Lacking coordinate information.')
        if new_center is not None:
            pass
        elif new_rect_2d is not None:
            pass
        else:
            raise ValueError('Need to input transform parameters.')

    # 计算相对坐标，reverse=True恢复为原坐标
    def compute_relative_points(self, center=None, reverse=False):
        if reverse is False:
            if self.grid_points is not None and self.is_rtc is False:
                if center is None:
                    center = self.center
                self.grid_points = np.subtract(self.grid_points, center)
                self.is_rtc = True
        else:
            if self.grid_points is not None and self.is_rtc is True:
                if center is None:
                    center = self.center
                self.grid_points = np.add(self.grid_points, center)
                self.is_rtc = False

    def save(self, dir_path: str, out_name: str = None, replace=False):
        if not replace:
            self.tmp_dump_str = 'tmp_terrain' + str(int(time.time()))
        self.dir_path = dir_path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        save_dir = os.path.join(self.dir_path, self.tmp_dump_str)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.vtk_data is not None and isinstance(self.vtk_data, (pv.RectilinearGrid, pv.UnstructuredGrid
                                                                    , pv.StructuredGrid)):
            save_path = os.path.join(save_dir, self.tmp_dump_str + '.vtk')
            self.vtk_data.save(filename=save_path)
            self.vtk_data = 'dumped'
        file_name = self.tmp_dump_str
        if out_name is not None:
            file_name = out_name
        file_path = os.path.join(save_dir, file_name + '.dat')
        with open(file_path, 'wb') as out_put:
            out_str = pickle.dumps(self)
            out_put.write(out_str)
            out_put.close()
        print('save terrain file into {}'.format(file_path))
        return self.__class__.__name__, file_path

    # 加载该类附属的vtk模型
    def load(self, dir_path=None):
        if self.dir_path is not None:
            if self.vtk_data == 'dumped':
                if dir_path is not None:
                    self.dir_path = dir_path
                save_path = os.path.join(self.dir_path, self.tmp_dump_str + '.vtk')
                if os.path.exists(save_path):
                    self.vtk_data = pv.read(filename=save_path)
                else:
                    raise ValueError('vtk data file does not exist')
