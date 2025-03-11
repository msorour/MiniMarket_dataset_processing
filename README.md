# MiniMarket_dataset_processing

generate_hdf5_dataset_with_padding.py is used to generate a fixed size dataset in the HDF5 format out of the raw .pcd point cloud files.
The output file will have 1200 partial view samples, each with 2048 coloured points. Where a partial view RGB-D camera output is less than 2048 points, the sample is padded with zeros.
The 2048 can be changed to generate a new dataset shall the need arise for more points per sample. 
