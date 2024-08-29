import os
from pathlib import Path
from typing import Union

import numpy as np

def save_pointcloud_from_ORB_SLAM(input_file: Union[Path, str], output_file: Union[Path,str] = "out.ply"):
    """Converts a comma separated list of map point coordinates into
    PLY format for viewing the generated map.

    Args:
        input_file (str or Path): Path to the input file which is expected to
        be a .csv file with the columns pos_x, pos_y, pos_z designating the
        coordinates of the points in the world reference frame.

        output_file (str or Path): Path to the output .ply file, format is
        described here: https://paulbourke.net/dataformats/ply/
    """

    coords = np.genfromtxt(input_file, delimiter=", ", skip_header=1)

    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]

    ply_header = 'ply\n' \
                'format ascii 1.0\n' \
                'element vertex %d\n' \
                'property float x\n' \
                'property float y\n' \
                'property float z\n' \
                'end_header' % x.shape[0]

    np.savetxt(output_file, np.column_stack((x, y, z)), fmt='%f %f %f', header=ply_header, comments='')

def save_trajectory_from_ORB_SLAM(input_file: Union[Path, str], output_file: Union[Path, str] = "out_trajectory.ply"):
    """Converts the saved trajectory file from TUM format to a point cloud.

    The input file is expected to be in the TUM format:
    timestamp x y z qx qy qz qw
    """
    x, y, z = [], [], []

    with open(input_file, "r") as file:
        lines = file.readlines()

    for line in lines:
        cols = line.strip().split()
        if len(cols) == 8:
            # Extract the x, y, z coordinates
            x.append(float(cols[1]))
            y.append(float(cols[2]))
            z.append(float(cols[3]))
        else:
            print(f"Skipping malformed line: {line.strip()}")

    # Convert lists to numpy arrays
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # RGB values for each point on the trajectory, set to be light green
    r = np.ones_like(x) * 144
    g = np.ones_like(x) * 238
    b = np.ones_like(x) * 144

    # Create PLY file header
    ply_header = 'ply\n' \
                 'format ascii 1.0\n' \
                 'element vertex %d\n' \
                 'property float x\n' \
                 'property float y\n' \
                 'property float z\n' \
                 'property uchar red\n' \
                 'property uchar green\n' \
                 'property uchar blue\n' \
                 'end_header' % x.shape[0]

    # Save the points along with RGB colors to the output PLY file
    np.savetxt(output_file, np.column_stack((x, y, z, r, g, b)), fmt='%f %f %f %d %d %d', header=ply_header, comments='')


if __name__ == "__main__":
    input_file = "/home/fang/PointCloud.txt" # ReferenceMapPoints seem to work better, use that file
    output_file = "./pointcloudBOT2.ply"

    input_trajectory = "/home/fang/CameraTrajectory.txt"
    output_trajectory = "./trajectoryBOT2.ply"

    save_pointcloud_from_ORB_SLAM(input_file, output_file)
    save_trajectory_from_ORB_SLAM(input_trajectory, output_trajectory)
