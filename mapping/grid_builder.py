import numpy as np
import cv2

def make_occupancy_grid(road_mask_u8: np.ndarray, obst_mask_u8: np.ndarray, cell: int, inflate_cells: int):
    H, W = road_mask_u8.shape[:2]
    cell = int(cell)
    gh, gw = H // cell, W // cell

    road_small = cv2.resize((road_mask_u8 > 0).astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST)
    obst_small = cv2.resize((obst_mask_u8 > 0).astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST)

    grid = np.ones((gh, gw), dtype=np.uint8)
    grid[road_small > 0] = 0
    grid[obst_small > 0] = 1

    if inflate_cells and inflate_cells > 0:
        k = 2 * int(inflate_cells) + 1
        grid_img = (grid * 255).astype(np.uint8)
        grid_img = cv2.dilate(grid_img, np.ones((k, k), np.uint8))
        grid = (grid_img > 0).astype(np.uint8)

    return grid, cell

def grid_distance_to_obstacles(grid: np.ndarray) -> np.ndarray:
    free = (grid == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 3)
    return dist.astype(np.float32)
