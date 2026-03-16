import heapq
import math
import numpy as np

def astar_weighted(grid: np.ndarray, start, goal, clearance: np.ndarray | None = None,
                   w_clear: float = 0.9, w_turn: float = 0.05, prev_dir=None):
    R, C = grid.shape
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < R and 0 <= sc < C): return [], float("inf")
    if not (0 <= gr < R and 0 <= gc < C): return [], float("inf")
    if grid[sr, sc] == 1 or grid[gr, gc] == 1: return [], float("inf")

    def h(r, c):
        return abs(r - gr) + abs(c - gc)

    nbrs = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
    ]

    pq = []
    heapq.heappush(pq, (h(sr, sc), 0.0, (sr, sc)))
    came = {(sr, sc): None}
    gscore = {(sr, sc): 0.0}
    dirmap = {(sr, sc): (0, 0)}

    while pq:
        _, g, (r, c) = heapq.heappop(pq)
        if (r, c) == (gr, gc):
            path = []
            cur = (r, c)
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path, g

        for dr, dc, step_cost in nbrs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            if grid[nr, nc] == 1:
                continue

            pen = 0.0
            if clearance is not None:
                cl = float(clearance[nr, nc])
                pen += w_clear * (1.0 / (cl + 1.0))

            cur_dir = dirmap.get((r, c), (0, 0))
            turn = 0.0
            if cur_dir != (0, 0):
                turn = 0.0 if (dr, dc) == cur_dir else 1.0
            elif prev_dir is not None:
                turn = 0.0 if (dr, dc) == prev_dir else 0.5
            pen += w_turn * turn

            ng = g + step_cost * (1.0 + pen)
            if (nr, nc) not in gscore or ng < gscore[(nr, nc)]:
                gscore[(nr, nc)] = ng
                came[(nr, nc)] = (r, c)
                dirmap[(nr, nc)] = (dr, dc)
                f = ng + h(nr, nc)
                heapq.heappush(pq, (f, ng, (nr, nc)))

    return [], float("inf")

