import numpy as np
import h5py
import random
#import time
import glob
import math
import getpass

print ("numpy version: ",np.__version__)


# List of objects
#dishwash_fairy_lemon_320ml_1200_2048
#dishwash_fairy_original_383ml_1200_2048

#hot_chocinstant_cadbury_400gm_1200_2048
#
#hot_choc_cadbury_250gm_1200_2048

#coffee_kenco_100gm_1200_2048

#jam_hartleys_apricot_300gm_1200_2048
#jam_hartleys_strawberry_300gm_1200_2048
#jam_hartleys_pineapple_300gm_1200_2048

#
#honey_rowse_340gm_1200_2048


#ketchup_heinz_400ml_1200_2048

#biscuit_spread_lotus_400gm_1200_2048
#hazelnut_cocoa_spread_nutella_350gm_1200_2048
#hazelnut_cocoa_spread_nutella_200gm_1200_2048
#crunchy_peanut_butter_sunpat_400gm_1200_2048
#crunchy_peanut_butter_sunpat_400gm_1200_2048

#shampoo_head_and_shoulders_classic_400ml_1200_2048
#shampoo_head_and_shoulders_classic_225ml_1200_2048
#shampoo_head_and_shoulders_citrus_250ml_1200_2048  
#shampoo_head_and_shoulders_citrus_400ml_1200_2048

#pasta_napolina_penne_no_50_500gm_1200_2048

#cereal_kelloggs_crunchy_nut_honey_and_nut_300gm_1200_2048

#TARGET_OBJECT_NAME = "shampoo_head_and_shoulders_citrus_400ml_1200_2048"
TARGET_OBJECT_NAME = "coffee_nescafe_3in1_original_6cups_1200_2048"
OBJECT_DATASET_PATH = "/home"+getpass.getuser()+"/MiniMarket_dataset_processing/MiniMarket77/"
OBJECT_MODEL_PATH = "/home"+getpass.getuser()+"/MiniMarket_dataset_processing/object_models/"
GENERATED_DATASET_PATH = "/home"+getpass.getuser()+"/MiniMarket_dataset_processing/object_segmentation_dataset/"

orientation_samples = 10

#
##
###
####
# The objective is to generate dataset, made of NUM_SAMPLES
# Each sample has the size of NUM_POINTS
# Each sample contains a point cloud of the target object, and a number of alien objects NUM_ALIEN
# Each point in the sample is labeled using one-hot-encoded
# Each sample as such has a size of NUM_POINTS x 8 consisting of:
# 1. NUM_POINTS x 3 point_cloud component
# 2. NUM_POINTS x 3 color_cloud component
# 3. NUM_POINTS x 8 one-hot-encoded label component

# We want that:
# 1. Each sample has fixed number of points NUM_POINTS
# 2. Each sample has one target object (for the moment till we figure out how to detect multiple instances!)
# 3. Each sample has variable number of alien objects (from 0 till 9)
# 4. Each object in the sample should be located in a different pose with respect to each other object, all objects should move specially the target object


def Rotx(t):
    return np.matrix([[1, 0, 0, 0], [0, np.cos(t), -np.sin(t), 0], [0, np.sin(t), np.cos(t), 0], [0, 0, 0, 1]])

def Roty(t):
    return np.matrix([[np.cos(t), 0, np.sin(t), 0], [0, 1, 0, 0], [-np.sin(t), 0, np.cos(t), 0], [0, 0, 0, 1]])

def Rotz(t):
    return np.matrix([[np.cos(t), -np.sin(t), 0, 0], [np.sin(t), np.cos(t), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def random_homogeneous_3d_rotation(x_min, x_max, y_min, y_max, z_min, z_max):
    angle_about_x = random.uniform(x_min, x_max)
    angle_about_y = random.uniform(y_min, y_max)
    angle_about_z = random.uniform(z_min, z_max)
    return Rotx(angle_about_x*math.pi/180) * Roty(angle_about_y*math.pi/180) * Rotz(angle_about_z*math.pi/180)


def random_homogeneous_3d_translation(x_min, x_max, y_min, y_max, z_min, z_max):
    distance_along_x = random.uniform(x_min, x_max)
    distance_along_y = random.uniform(y_min, y_max)
    distance_along_z = random.uniform(z_min, z_max)
    return np.matrix([
    [1, 0, 0, distance_along_x],
    [0, 1, 0, distance_along_y],
    [0, 0, 1, distance_along_z],
    [0, 0, 0, 1]
    ])    






NUM_POINTS_PER_SEG_SAMPLE = 20480
#NUM_POINTS_PER_SEG_SAMPLE = 40960
Max_number_of_objects_per_seg_sample = 10  #each object has 2048 colored points

def generate_object_segmentation_dataset( object_dataset_path, target_object_name, generated_dataset_path, Max_number_of_objects_per_seg_sample ):

    all_seg_sample_points=[]
    all_seg_sample_colors=[]
    all_seg_sample_labels=[]
    
    # Get files
    target_hdf5_file = object_dataset_path + target_object_name
    alien_hdf5_files = sorted(glob.glob(object_dataset_path+"*"))
    alien_hdf5_files.remove(target_hdf5_file)
    
    # Target object samples collection (each object file has 1200 samples)
    # Assuming each sample has 2048 points, the shape of all samples is (1200,2048,6)
    with h5py.File(target_hdf5_file, "r") as f:
        target_points = f["point_clouds"][()]  # returns as a numpy array
        target_colors = f["color_clouds"][()]  # returns as a numpy array
    object_data_size = target_points.shape[1] # assuming all objects will have the same sample size

    # For each sample in the target object (each object has 1200 samples), we take such sample and randomly rotate it 4 times to enrich the data
    # This means we have in total 4800 segmentation samples, each consist of 20480 points, making up 10 objects at most (ranging from 1 to 10), one of them is the target object.
    # Each point is 8D -> 3D position + 3D color + 2D one-hot-shot label
    # Dataset dimension as such should be 4800 x 20480 x 8
    for it in range(orientation_samples):
        for target_sample_index in range(target_points.shape[0]):
        #for target_sample_index in range(5):
            # Select one sample
            selected_target_sample_point = target_points[target_sample_index,:,:]
            selected_target_sample_color = target_colors[target_sample_index,:,:]
            
            # Plot sample
            #fig = plt.figure(figsize=(5, 5))
            #ax = Axes3D(fig)
            #ax.set_xlim3d(-0.4, 0.4)
            #ax.set_ylim3d(-0.4, 0.4)
            #ax.set_zlim3d(-0.4, 0.4)
            #ax.scatter(selected_target_sample_point[:,0], selected_target_sample_point[:,1], selected_target_sample_point[:,2], c=selected_target_sample_color[:,:])
            #ax.set_axis_off()
            #plt.show()

            # Convert to homogeneous coordinates (add a column of ones to the 3d points to be able to multiply with 4x4 homogeneous transform afterwards)
            selected_target_sample_point = np.concatenate( (selected_target_sample_point, np.ones((object_data_size,1))),axis=1)
            
            # Apply homogeneous transformation
            random_homogeneous_transformation = random_homogeneous_3d_translation(-0.25,0.25,-0.25,0.25,0.0,0.25) * random_homogeneous_3d_rotation(-180,180,-180,180,-180,180)
            
            selected_target_sample_point = np.matmul( random_homogeneous_transformation, np.transpose(selected_target_sample_point) )
            selected_target_sample_point = np.transpose(selected_target_sample_point)
            
            # Remove the column of ones added previously
            selected_target_sample_point = np.delete( selected_target_sample_point, 3, 1 )
            
            # Add the target object sample to the collective segmentation sample
            seg_sample_point = selected_target_sample_point
            seg_sample_color = selected_target_sample_color
            #seg_sample_label = np.ones(object_data_size)
            seg_sample_label = np.concatenate( (np.ones((object_data_size,1)), np.zeros((object_data_size,1))),axis=1)  #one-hot encoding


            # Randomly generate the number of alien objects in this sample
            NUM_ALIEN_OBJECTS = random.randrange(Max_number_of_objects_per_seg_sample-1)
            print("iteration: ", str(it+1) + " , processing sample: " + str(target_sample_index) + " , number of alien objects = " + str(NUM_ALIEN_OBJECTS))

            # Remove files randomly to keep only the required number of alien objects NUM_ALIEN_OBJECTS
            copy_alien_files = alien_hdf5_files.copy()
            while len(copy_alien_files) > NUM_ALIEN_OBJECTS:
                copy_alien_files.pop(random.randrange(len(copy_alien_files)))
                    
            
            # Alien objects data collection
            for i, alien_object_file in enumerate(copy_alien_files):
                with h5py.File(copy_alien_files[i], "r") as f:
                    alien_points = f["point_clouds"][()]
                    alien_colors = f["color_clouds"][()]
                    
                    # Pick a random sample of the alien object i
                    alien_sample_index = random.randrange(alien_points.shape[0])
                    selected_alien_sample_point = alien_points[alien_sample_index,:,:]
                    selected_alien_sample_color = alien_colors[alien_sample_index,:,:]

                    # Convert to homogeneous coordinates (add a column of ones to the 3d points to be able to multiply with 4x4 homogeneous transform afterwards)
                    selected_alien_sample_point = np.concatenate( (selected_alien_sample_point, np.ones((object_data_size,1))),axis=1)

                    # Apply homogeneous transformation
                    random_homogeneous_transformation = random_homogeneous_3d_translation(-0.25,0.25,-0.25,0.25,0.0,0.25) * random_homogeneous_3d_rotation(-180,180,-180,180,-180,180)
                    selected_alien_sample_point = np.matmul( random_homogeneous_transformation, np.transpose(selected_alien_sample_point) )
                    selected_alien_sample_point = np.transpose(selected_alien_sample_point)

                    # Remove the column of ones added previously
                    selected_alien_sample_point = np.delete( selected_alien_sample_point, 3, 1 )

                    # Add the alien object sample to the collective segmentation sample
                    seg_sample_point = np.concatenate( (seg_sample_point, selected_alien_sample_point), axis=0 )
                    seg_sample_color = np.concatenate( (seg_sample_color, selected_alien_sample_color), axis=0 )
                    #seg_sample_label = np.concatenate( (seg_sample_label, np.zeros(object_data_size)), axis=0 )
                    seg_sample_label = np.concatenate( (seg_sample_label, np.concatenate( (np.zeros((object_data_size,1)), np.ones((object_data_size,1))),axis=1)), axis=0 )  # one-hot encoding


            # Pad the remaining sample size with zeros since we have varying number of objects per segmentation sample generated
            seg_sample_point = np.concatenate( (seg_sample_point, np.zeros((NUM_POINTS_PER_SEG_SAMPLE - seg_sample_point.shape[0], 3)) ), axis=0 )
            seg_sample_color = np.concatenate( (seg_sample_color, np.zeros((NUM_POINTS_PER_SEG_SAMPLE - seg_sample_color.shape[0], 3)) ), axis=0 )
            #seg_sample_label = np.concatenate( (seg_sample_label, np.zeros((NUM_POINTS_PER_SEG_SAMPLE - seg_sample_label.shape[0])) ), axis=0 )
            seg_sample_label = np.concatenate( (seg_sample_label, np.concatenate( (np.zeros(((NUM_POINTS_PER_SEG_SAMPLE - seg_sample_label.shape[0]),1)), np.ones(((NUM_POINTS_PER_SEG_SAMPLE - seg_sample_label.shape[0]),1))),axis=1)), axis=0 )  # one-hot encoding
            
            # Plot sample
            #fig = plt.figure(figsize=(7, 7))
            #ax = Axes3D(fig)
            #ax.set_xlim3d(-0.4, 0.4)
            #ax.set_ylim3d(-0.4, 0.4)
            #ax.set_zlim3d(-0.4, 0.4)
            #ax.scatter(seg_sample_point[:,0], seg_sample_point[:,1], seg_sample_point[:,2], c=seg_sample_color[:,:])
            #ax.set_axis_off()
            #plt.show()

            # Plot sample to verify labeling
            #fig = plt.figure(figsize=(7, 7))
            #ax = Axes3D(fig)
            #ax.set_xlim3d(-0.4, 0.4)
            #ax.set_ylim3d(-0.4, 0.4)
            #ax.set_zlim3d(-0.4, 0.4)
            #ax.scatter(seg_sample_point[:,0], seg_sample_point[:,1], seg_sample_point[:,2], c=seg_sample_label[:,0])
            #ax.set_axis_off()
            #plt.show()

            # add current segmentation sample to the dataset
            all_seg_sample_points.append(seg_sample_point)
            all_seg_sample_colors.append(seg_sample_color)
            all_seg_sample_labels.append(seg_sample_label)
    
    print(np.asarray(all_seg_sample_points).shape)
    print(np.asarray(all_seg_sample_colors).shape)
    print(np.asarray(all_seg_sample_labels).shape)

    # Save the segmentation samples to a new file
    hdf5_filename = generated_dataset_path + target_object_name + "_segmentation_" + str(NUM_POINTS_PER_SEG_SAMPLE) + "_" + str(orientation_samples*1200)
    with h5py.File(hdf5_filename, 'w') as f:
        # Create a point clouds dataset in the file
        f.create_dataset("seg_points", data = np.asarray(all_seg_sample_points))
        f.create_dataset("seg_colors", data = np.asarray(all_seg_sample_colors))
        f.create_dataset("seg_labels", data = np.asarray(all_seg_sample_labels))
    print("done!")
    return






generate_object_segmentation_dataset(OBJECT_DATASET_PATH, TARGET_OBJECT_NAME, GENERATED_DATASET_PATH, Max_number_of_objects_per_seg_sample)

