import numpy as np
import cv2

def read_point_cloud(filename):
    """Read point cloud data from a .txt file, skipping the header line"""
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  # Skip the header line
    return data

def bresenham(x0, y0, x1, y1):
    """Bresenham's Line Algorithm to generate points on a line"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

def project_to_xy_plane(points):
    """Project the point cloud onto the best-fit XY plane"""
    # Subtract the centroid
    centroid = np.mean(points[:, 1:], axis=0)
    points_centered = points[:, 1:] - centroid
    
    # PCA to find the best-fit plane
    cov_matrix = np.cov(points_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # The normal of the best-fit plane is the eigenvector with the smallest eigenvalue
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    
    # Project the points onto the plane by removing the component along the normal
    projection = points_centered - (points_centered @ normal)[:, np.newaxis] * normal
    
    # Return the original map IDs and projected x, y, z coordinates
    projected_points = np.hstack((points[:, :1], projection + centroid))
    
    return projected_points

def create_grid_map(points, grid_size, cell_size):
    """Create a grid map from point cloud data using Bresenham's Line Algorithm"""
    grid_map = np.zeros(grid_size, dtype=np.uint8)

    for i in range(len(points) - 1):
        # Extract point coordinates (now correctly aligned to xy-plane)
        x0, y0 = points[i][1], points[i][2]
        x1, y1 = points[i + 1][1], points[i + 1][2]

        # Convert coordinates to grid cell indices
        x0_idx = int(np.floor((x0 - grid_min_x) / cell_size))
        y0_idx = int(np.floor((y0 - grid_min_y) / cell_size))
        x1_idx = int(np.floor((x1 - grid_min_x) / cell_size))
        y1_idx = int(np.floor((y1 - grid_min_y) / cell_size))

        # Get the points on the line between (x0_idx, y0_idx) and (x1_idx, y1_idx) using Bresenham's algorithm
        line_points = bresenham(x0_idx, y0_idx, x1_idx, y1_idx)

        # Mark the cells along the line as occupied
        for x_idx, y_idx in line_points:
            if 0 <= x_idx < grid_size[0] and 0 <= y_idx < grid_size[1]:
                grid_map[x_idx, y_idx] = 255  # Mark the cell as occupied

    return grid_map

# Parameters
point_cloud_fname = '/home/fang/PointCloud.txt'
output_pgm_fname = 'gridmapBOT2.pgm'
grid_size = (1000, 1000)  # Define the grid size
cell_size = 0.1  # Define the size of each cell in the grid

# Read point cloud data
points = read_point_cloud(point_cloud_fname)

# Project the points onto the best-fit XY plane
projected_points = project_to_xy_plane(points)

# Determine grid boundaries using the projected points
grid_min_x = np.min(projected_points[:, 1])
grid_min_y = np.min(projected_points[:, 2])
grid_max_x = np.max(projected_points[:, 1])
grid_max_y = np.max(projected_points[:, 2])

# Create grid map using Bresenham's Line Algorithm
grid_map = create_grid_map(projected_points, grid_size, cell_size)

# Save grid map as a .pgm file
cv2.imwrite(output_pgm_fname, grid_map)
print(f'Grid map saved as {output_pgm_fname}')
