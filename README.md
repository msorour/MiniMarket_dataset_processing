# MiniMarket_dataset_processing

generate_hdf5_dataset_with_padding.py is used to generate a fixed size dataset in the HDF5 format out of the raw .pcd point cloud files.
The output file will have 1200 partial view samples, each with 2048 coloured points. Where a partial view RGB-D camera output is less than 2048 points, the sample is padded with zeros.
The 2048 can be changed to generate a new dataset shall the need arise for more points per sample.

segmentation_dataset_prep.py is used to generate one hot encoded segmentation mask dataset for a single object.
A segmentation sample size can be controlled, we use here 20480 coloured points per sample. In each sample we can have arbitrary number of alien objects between 0 to 9 and a single target object.
In case of less than 9 alien objects in a sample, the remaining points are padded with zeros.
