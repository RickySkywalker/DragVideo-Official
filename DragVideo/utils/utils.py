import cv2
import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import torchvision.io as io
from torchvision import transforms

def get_best_device(i=0):
    if torch.cuda.is_available():
        return torch.device("cuda:" + str(i))
    elif torch.backends.mps.is_built():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def point_and_save_video(video, handle_points, target_points, save_path, fps=24, res_shape=(512, 512), sup_res_shape=(256, 256)):
    
    target_points = target_points.astype(int)
    handle_points = handle_points.astype(int)

    video = video.copy()
    print("video length",len(video))
    print("pointt length",len(handle_points))
    for i in range(len(video)):
        video[i] = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
        for j in range(len(handle_points[i])):
            cv2.circle(video[i], handle_points[i][j], radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(video[i], target_points[i][j], radius=5, color=(255, 0, 0), thickness=-1)
            cv2.arrowedLine(video[i], handle_points[i][j], target_points[i][j], (255, 255, 255), 2, tipLength=0.5)
        video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
    video = torch.from_numpy(video)
    print("video shape",video.shape)
    print(handle_points[30:])
    io.write_video(save_path, video, fps)


def get_edge_points(mask):
    gray = mask.astype(np.uint8) * 255

    edges = cv2.Canny(gray, 0, 1)

    indices = np.argwhere(edges != 0)

    expanded_indices = np.concatenate(
        (indices, indices + np.array([-1, -1]), indices + np.array([-1, 0]), indices + np.array([-1, 1]),
         indices + np.array([0, -1]), indices + np.array([0, 1]), indices + np.array([1, -1]), indices + np.array([1, 0]),
         indices + np.array([1, 1])),
        axis=0
    )
    return expanded_indices


def first_extend(mask, radius=30, max=512):
    # Convert mask to grayscale image
    gray = mask.astype(np.uint8) * 255

    # Perform Canny edge detection
    edges = cv2.Canny(gray, 0, 1)

    # Find the coordinates of edge points
    indices = np.argwhere(edges != 0)

    # Create a radius matrix
    r = np.arange(-radius, radius + 1)
    radius_matrix = np.transpose([np.tile(r, len(r)), np.repeat(r, len(r))])

    # Calculate the expanded indices
    expanded_indices = indices[:, None, :] + radius_matrix[None, :, :]

    # Clip the expanded indices to stay within the bounds of the mask
    expanded_indices = np.clip(expanded_indices.astype(int), 0, max-1)

    # Set the values of pixels at the expanded indices to 1
    mask[expanded_indices[:, :, 0], expanded_indices[:, :, 1]] = 1

    return mask


def draw_line(coord1, coord2, mask):
    row1, col1 = coord1
    row2, col2 = coord2
    max_row, max_col = mask.shape
    delta_row = row2 - row1
    delta_col = col2 - col1
    
    length = max(abs(int(delta_row)), abs(int(delta_col))) + 1
    t = np.linspace(0, 1, length)
    
    row_step = delta_row * t
    col_step = delta_col * t
    
    row_indices = np.round(row1 + row_step).astype(int)
    col_indices = np.round(col1 + col_step).astype(int)
    
    row_indices = np.clip(row_indices, 0, max_row - 1)
    col_indices = np.clip(col_indices, 0, max_col - 1)

    mask[row_indices, col_indices] = 1
    
    return mask

def extend_mask(mask:np.ndarray, offset:np.ndarray): # enxtend singe mask and single offset
    indice = get_edge_points(mask)
    offset = np.array([offset[1],offset[0]])
    """
    for x,y in zip(indice[0], indice[1]):
        if offset[0] > 0:
            row_start = x
            row_end = x + int(offset[1])
        else:
            row_start = x + int(offset[1])
            row_end = x
        if offset[1] > 0:
            col_start = y
            col_end = y + int(offset[0])
        else:
            col_start = y + int(offset[0])
            col_end = y
    """
    
    for points in indice:
         target_points = points + offset   
         mask = draw_line(points, target_points, mask)
    return mask

def colormap(rgb=True):
	color_list = np.array(
		[
			0.000, 0.000, 0.000,
			1.000, 1.000, 1.000,
			1.000, 0.498, 0.313,
			0.392, 0.581, 0.929,
			0.000, 0.447, 0.741,
			0.850, 0.325, 0.098,
			0.929, 0.694, 0.125,
			0.494, 0.184, 0.556,
			0.466, 0.674, 0.188,
			0.301, 0.745, 0.933,
			0.635, 0.078, 0.184,
			0.300, 0.300, 0.300,
			0.600, 0.600, 0.600,
			1.000, 0.000, 0.000,
			1.000, 0.500, 0.000,
			0.749, 0.749, 0.000,
			0.000, 1.000, 0.000,
			0.000, 0.000, 1.000,
			0.667, 0.000, 1.000,
			0.333, 0.333, 0.000,
			0.333, 0.667, 0.000,
			0.333, 1.000, 0.000,
			0.667, 0.333, 0.000,
			0.667, 0.667, 0.000,
			0.667, 1.000, 0.000,
			1.000, 0.333, 0.000,
			1.000, 0.667, 0.000,
			1.000, 1.000, 0.000,
			0.000, 0.333, 0.500,
			0.000, 0.667, 0.500,
			0.000, 1.000, 0.500,
			0.333, 0.000, 0.500,
			0.333, 0.333, 0.500,
			0.333, 0.667, 0.500,
			0.333, 1.000, 0.500,
			0.667, 0.000, 0.500,
			0.667, 0.333, 0.500,
			0.667, 0.667, 0.500,
			0.667, 1.000, 0.500,
			1.000, 0.000, 0.500,
			1.000, 0.333, 0.500,
			1.000, 0.667, 0.500,
			1.000, 1.000, 0.500,
			0.000, 0.333, 1.000,
			0.000, 0.667, 1.000,
			0.000, 1.000, 1.000,
			0.333, 0.000, 1.000,
			0.333, 0.333, 1.000,
			0.333, 0.667, 1.000,
			0.333, 1.000, 1.000,
			0.667, 0.000, 1.000,
			0.667, 0.333, 1.000,
			0.667, 0.667, 1.000,
			0.667, 1.000, 1.000,
			1.000, 0.000, 1.000,
			1.000, 0.333, 1.000,
			1.000, 0.667, 1.000,
			0.167, 0.000, 0.000,
			0.333, 0.000, 0.000,
			0.500, 0.000, 0.000,
			0.667, 0.000, 0.000,
			0.833, 0.000, 0.000,
			1.000, 0.000, 0.000,
			0.000, 0.167, 0.000,
			0.000, 0.333, 0.000,
			0.000, 0.500, 0.000,
			0.000, 0.667, 0.000,
			0.000, 0.833, 0.000,
			0.000, 1.000, 0.000,
			0.000, 0.000, 0.167,
			0.000, 0.000, 0.333,
			0.000, 0.000, 0.500,
			0.000, 0.000, 0.667,
			0.000, 0.000, 0.833,
			0.000, 0.000, 1.000,
			0.143, 0.143, 0.143,
			0.286, 0.286, 0.286,
			0.429, 0.429, 0.429,
			0.571, 0.571, 0.571,
			0.714, 0.714, 0.714,
			0.857, 0.857, 0.857
		]
	).astype(np.float32)
	color_list = color_list.reshape((-1, 3)) * 255
	if not rgb:
		color_list = color_list[:, ::-1]
	return color_list

color_list = colormap()
color_list = color_list.astype('uint8').tolist()

def vis_add_mask(image, mask, color, alpha):
	color = np.array(color_list[color])
	mask = mask > 0.5
	image[mask] = image[mask] * (1-alpha) + color * alpha
	return image.astype('uint8')


def mask_painter(input_image, input_mask, mask_color=5, mask_alpha=0.7, contour_color=1, contour_width=3):
	assert input_image.shape[:2] == input_mask.shape, 'different shape between image and mask'
	# 0: background, 1: foreground
	mask = np.clip(input_mask, 0, 1)
	contour_radius = (contour_width - 1) // 2

	dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
	dist_transform_back = cv2.distanceTransform(1-mask, cv2.DIST_L2, 3)
	dist_map = dist_transform_fore - dist_transform_back

	contour_radius += 2
	contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
	contour_mask = contour_mask / np.max(contour_mask)
	contour_mask[contour_mask>0.5] = 1.

	# paint mask
	painted_image = vis_add_mask(input_image.copy(), mask.copy(), mask_color, mask_alpha)
	# paint contour
	painted_image = vis_add_mask(painted_image.copy(), 1-contour_mask, contour_color, 1)

	return painted_image