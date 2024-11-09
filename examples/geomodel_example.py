# -*- coding: utf-8 -*-
import os.path
from dgl_geodataset import DglGeoDataset
from geomodel_analysis import *
from gme_trainer import GmeTrainer, GmeTrainerConfig, GraphTransConfig
from models.model import GraphTransfomer
from data_structure.reader import ReadExportFile
from data_structure.geodata import *
from data_structure.terrain import TerrainData
from utils.plot_utils import control_visibility_with_layer_label
import random
import sys
root_dir = os.path.abspath('..')
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'data_structure'))
sys.path.append(os.path.join(root_dir, 'models'))
sys.path.append(os.path.join(root_dir, 'utils'))
# This example gives a complete process of training on geologic data and finally generating a geologic model
# This example uses drill data read from a boreholes file
if __name__ == '__main__':
    # load and preprocess dataset
    print('Loading data')
    root_path = os.path.abspath('..')
    xm_file_path = os.path.join(root_path, 'data', 'borehole_data.dat')
    # # # 从外部数据文本中加载钻孔数据
    reader = ReadExportFile()
    boreholes_data = reader.read_boreholes_data_from_text_file(dat_file_path=xm_file_path)
    # Setting the radius of the control range of the borehole
    boreholes_data.set_boreholes_control_buffer_dist_xy(radius=2)
    # Loading borehole data into geologic data integration containers
    gd = GeodataSet()
    gd.append(boreholes_data)
    # Normalize stratigraphic labels to be consecutive non-negative integers starting at 0.
    # The default unknown prediction label is -1
    gd.standardize_labels()
    # Construct a terrain surface based on the borehole data and pass it into the GmeModelGraphList class,
    # or if no terrain is constructed, pass None
    terrain_data = TerrainData(extend_buffer=10, top_surface_offset=0.5)
    # Constructing 3D model regular mesh data into graph mesh data
    gme_models = GmeModelGraphList('gme_model', root=root_path,
                                   split_ratio=DataSetSplit(1),
                                   input_sample_data=gd,
                                   add_inverse_edge=True,
                                   grid_dims=[120, 120, 200],
                                   terrain_data=terrain_data)
    model_idx = 0
    # Convert the graph data into an organized structure for the dgl library so that graph computations
    # can be implemented using the dgl library
    dataset = DglGeoDataset(gme_models)
    # Parameters related to model training, initial training parameter settings, currently in the windows system,
    # dgl only supports single gpu training
    trainer_config = GmeTrainerConfig(max_epochs=1500, batch_size=512, num_workers=4, learning_rate=1e-5,
                                      ckpt_path=os.path.join(root_path, 'processed', 'latest_tran.pth'),
                                      output_dir=os.path.join(root_path, 'output'),
                                      out_put_grid_file_name='vtk_model',
                                      sample_neigh=[10, 10, 15, 15], gpu_num=1)
    # Take a graph from the graph dataset
    g = dataset[model_idx]
    # Get the dimensions of the input features
    in_size = g.ndata['feat'].shape[-1]
    # Get the number of output stratigraphic categories
    out_size = dataset.num_classes['labels'][model_idx]
    # Set the parameters related to the model structure, here they are used to define the network structure
    model_config = GraphTransConfig(in_size=in_size, out_size=out_size, n_head=4, n_embd=512, gnn_layer_num=3,
                                    n_layer=4)
    # Building predictive models
    model = GraphTransfomer(model_config)
    # Creating a Model Trainer
    trainer = GmeTrainer(model, dataset, trainer_config)
    # model training
    print('Training...')
    # Start training with early stopping method, early_stop_patience parameter is 50,
    # and the validation loss does not decrease at 50 epoch, then training stops
    trainer.train(data_split_idx=model_idx, has_test_label=True, early_stop_patience=50)
    # After the model is run, the modeled model is generated at the out_put_grid_file_name path
    # Visualization of the built geological model
    result_grid = reader.read_geodata(file_path=os.path.join(root_path, 'output', 'vtk_model', 'vtk_model.dat'))
    plotter = control_visibility_with_layer_label([result_grid, gd.geodata_list[0]])
    plotter.show()
