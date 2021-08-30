import numpy as np
import cv2
"""
'this file check and convert the pose data'
Light_weight_openpose
0'Nose', 9
1'Neck', 8
2'Right Shoulder', 13
3'Right Elbow', 14
4'Right Wrist', 15
5'Left Shoulder', 10
6'Left Elbow', 11
7'Left Wrist', 12
8'Right Hip', 1
9'Right Knee', 2
10'Right Ankle', 3
11'Left Hip', 4
12'Left Knee', 5
13'Left Ankle', 6
14'Right eye'
15'Left eye'
16'Right ear'
17'Left ear'
18'Center Hip', 0
19'Center Shoulder', 7

reorder = [ (11+8)/2 , 8, 9, 10, 11, 12, 13, (2+5)/2
, 1, 0, 5, 6, 7, 2, 3, 4]
""" 


def openpose_to_h36m(openpose, img_shape):
    
    openpose = np.array(openpose)
    #openpose2d=openpose.reshape(openpose.shape[0], -1, 2)
    openpose2d = openpose

    def normalize_screen_coordinates(X, w, h):
        assert X.shape[-1] == 2
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio

        tmp = []
        no_detected_points= []
        for i, (x,y) in enumerate(X):
            if (x == -1 and y==-1):
                temp= np.array([0, 0])
                no_detected_points.append(i)

            else:
                temp= (X[i] / w * 2 - [1, h / w] )    
            
            tmp.append(temp)

        ret = np.array(tmp)

        return ret, no_detected_points

    def get_2d_pose_reorderednormed_signle(source):
        
        # Interpolation if keypoint is not detected
        if ( (source[11][0] == -1 and source[11][1] == -1) or (source[8][0]== -1 and source[8][1]== -1)):
            pelvis0 = np.ones((1,2)) * -1
        else:
            pelvis0 = ((source[11] + source[8])/2)
        
        if ( (source[2][0] == -1 and source[2][1] == -1) or (source[5][0] == -1 and source[5][1] == -1)):
            spine7 = np.ones((1,2)) * -1
        else:
            spine7 = ((source[2] + source[5])/2)  

        if ( (source[4][0] == -1 and source[4][1] == -1 )):
            source[4] = source[3]
        
        if ( (source[7][0] == -1 and source[7][1] ==-1)):
            source[7] = source[6]
        
        if ( (source[10][0] == -1 and source[10][1] ==-1)):
            source[10] = source[9]
        
        if ( (source[13][0] == -1 and source[13][1] ==-1)):
            source[13] = source[12]

        reorder = [-1,8,9,10,11,12,13,-1,1,0,5,6,7,2,3,4]
        tmp_array = source[reorder][:, :2]
        
        tmp_array[0] = pelvis0
        tmp_array[7] = spine7

        tmp_array1, no_detected_points = normalize_screen_coordinates(tmp_array,img_shape[1], img_shape[0])
        
        return tmp_array1, no_detected_points

    tmp2d = []
    no2d = []
    for i in range(openpose2d.shape[0]):
        tmp_array1, no_detected_points =get_2d_pose_reorderednormed_signle(openpose2d[i])
        tmp2d.append(tmp_array1) 
        no2d.append(no_detected_points)

    tmp2d= np.array(tmp2d)
    no2d = np.array(no2d, dtype=object)

    return tmp2d, no2d

import matplotlib.pyplot as plt

def show2Dpose(img, channels, ax, no2d_index= [],lcolor="#3498db", rcolor="#e74c3c", add_labels=True):
    """
    Visualize a 2d skeleton

    Args
    channels: 64x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
    Returns
    Nothing. Draws on ax.
    """
    
    #img = cv2.resize(img, dsize=(640, 360), interpolation=cv2.INTER_AREA)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)

    def back_normalize_screen_coordinates(X, w, h):
        assert X.shape[-1] == 2
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio

        tmp = []

        for i, (x,y) in enumerate(X):
            if (x == 0 and y==0):
                temp= np.array([0, 0])
            else:
                temp= (X[i] / 2 * w + [w/2, h/2])    
            
            tmp.append(temp)

        ret = np.array(tmp)

        return ret

    vals = np.reshape( channels, (-1, 2) )
    vals = back_normalize_screen_coordinates(vals, img.shape[1], img.shape[0])
    #plt.plot(vals[:,0], vals[:,1], 'ro')
    I  = np.array([0,1,2,0,4,5,0,7,8,8,10,11,8,13,14]) # start points
    J  = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) # end points
    LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange( len(I) ):
        if I[i] not in no2d_index and J[i] not in no2d_index:
            x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
            ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

    RADIUS = 1 # space around the subject
    xroot, yroot = vals[0,0], vals[0,1]
    
    #ax.set_xlim([-1, 1])
    #ax.set_ylim([-1, 1])
    
    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
    ax.axis('off')
    #ax.set_aspect('equal')

import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

def show3Dpose(channels, ax, no2d_index= [], lcolor="#3498db", rcolor="#e74c3c", add_labels=True,
               gt=False,pred=False): # blue, orange
    """
    Visualize a 3d skeleton

    Args
    channels: 96x1 vector. The pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
    Returns
    Nothing. Draws on ax.
    """

    #   assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (16, -1) )

    I  = np.array([0,1,2,0,4,5,0,7,8,8,10,11,8,13,14]) # start points
    J  = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) # end points
    LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange( len(I) ):
        if I[i] not in no2d_index and J[i] not in no2d_index:
            x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
            if gt:
                ax.plot(x,z, -y,  lw=2, c='k')
            elif pred:
                ax.plot(x,z, -y,  lw=2, c='r')
            else:
                ax.plot(x, z, -y,  lw=2, c=lcolor if LR[i] else rcolor)

    RADIUS = 1 # space around the subject
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_zlim3d([-RADIUS-yroot, RADIUS-yroot])


    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("-y")

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)