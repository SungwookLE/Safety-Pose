import numpy as np

"""
'this file check and convert the pose data'
COCO_AND_EXTRA_PARTS
'-----------------'
0'Nose'
1'Neck', 8
2'Right Shoulder', 13
3'Right Elbow', 14
4'Right Wrist', 15
5'Left Shoulder', 10
6'Left Elvow', 11
7'Left Wrist', 12
8'Right Hip', 1
9'Right Knee', 2
10'Right Ankle', 3
11'Left Hip', 4
12'Left Knee', 5 
13'Left Ankle', 6
14'Right Eye' 
15'Left Eye' 
16'Right Ear'
17'Left Ear'
18'Head', 9
19'Pelvis', 0
20'Center Shoulder, 7

reorder = [19,8,9,10,11,12,13,20,1,18,5,6,7,2,3,4]

"""

# load the download data
safety_val_path = './dataset_extras/safety_valid.txt'
safety_valid = np.loadtxt(safety_val_path, dtype=float)

# convert the data to a list to processing.
safety_val_list = []
for i, value in enumerate(safety_valid):
    tmp_dict = {}
    tmp_dict['kpts2d'] = value[2:44].reshape(-1,2)
    tmp_dict['kpts3d'] = value[44:].reshape(-1,3)

    # prepare the image width for 2D keypoint normalization.
    tmp_dict['width'] = 640
    tmp_dict['height'] = 360

    safety_val_list.append(tmp_dict)

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def get_2d_pose_reorderednormed(source):
    reorder = [19,8,9,10,11,12,13,20,1,18,5,6,7,2,3,4]
    tmp_array = source['kpts2d'][reorder][:, :2]
    tmp_array1 = normalize_screen_coordinates(tmp_array, source['width'], source['height'])
    return tmp_array1

def get_3d_pose_reordered(source):
    reorder = [19,8,9,10,11,12,13,20,1,18,5,6,7,2,3,4]
    tmp_array = source['kpts3d'][reorder][:, :3]
    return tmp_array

# convert the pose to 16 joints and put into array
safety_data_2dpose = []
safety_data_3dpose = []

for source in safety_val_list:
    tmp2d = get_2d_pose_reorderednormed(source)
    tmp3d = get_3d_pose_reordered(source)
    tmp3d = tmp3d / 1000.0
    safety_data_2dpose.append(tmp2d)
    safety_data_3dpose.append(tmp3d)

safety_data_2dpose = np.array(safety_data_2dpose)
safety_data_3dpose = np.array(safety_data_3dpose)

# save the npz for test purpose
print(safety_data_3dpose.shape)
print(safety_data_2dpose.shape)
np.savez('./test_set/test_safety.npz',pose3d=safety_data_3dpose,pose2d=safety_data_2dpose)
