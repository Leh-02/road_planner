import numpy as np
import cv2


def make_occupancy_grid(road_mask_u8: np.ndarray, obst_mask_u8: np.ndarray, cell: int, inflate_cells: int):
    """Build grid where 0 is free road and 1 is occupied.

    Important: obstacle inflation is applied ONLY to obstacle cells, not to the whole
    occupied grid. Inflating the whole grid also inflates the non-road background and
    can close narrow lanes, which makes A* return an empty path when cars are present.
    """
    H, W = road_mask_u8.shape[:2]
    cell = max(1, int(cell))
    gh, gw = max(1, H // cell), max(1, W // cell)

    road_small = cv2.resize((road_mask_u8 > 0).astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST)
    obst_small = cv2.resize((obst_mask_u8 > 0).astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST)

    if inflate_cells and inflate_cells > 0:
        k = 2 * int(inflate_cells) + 1
        obst_img = (obst_small * 255).astype(np.uint8)
        obst_img = cv2.dilate(obst_img, np.ones((k, k), np.uint8), iterations=1)
        obst_small = (obst_img > 0).astype(np.uint8)

    grid = np.ones((gh, gw), dtype=np.uint8)
    grid[road_small > 0] = 0
    grid[obst_small > 0] = 1
    return grid, cell


def grid_distance_to_obstacles(grid: np.ndarray) -> np.ndarray:
    free = (grid == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 3)
    return dist.astype(np.float32)
