# Central Road Routing (Segmentation + YOLO + A*), no manual BEV points

## What it does
- Segments road on each frame (SegFormer/Cityscapes)
- Selects ONLY the road connected to the bottom-center (central road) even if multiple roads exist
- Detects obstacles (YOLOv8) and builds an occupancy grid (image-space, no BEV trapezoid)
- Plans a path with A* using clearance cost (keeps distance from obstacles)
- Shows:
  - chosen path (polyline)
  - direction arrow from the start every frame
  - candidate direction arrows at intersections/merges (left/center/right), then follows chosen best route
- Writes output video at the original FPS (not slowed) and full length

## Setup
```bash
pip install -r requirements.txt
```


Put your input video:
- `video/input.mp4`

Run:
```bash
python main.py --source video/input.mp4 --save output/result.mp4
```
```bash
python main.py --source video/usa.mp4 --save output/usa.mp4
```
```bash
python main.py --source video/japan.mp4 --save output/japan.mp4
```
```bash
python main.py --source video/highway.mp4 --save output/highway.mp4
```

Optional: show preview window:
```bash
python main.py --source video/input.mp4 --save output/result.mp4 --show
```
```bash
python main.py --source video/usa.mp4 --save output/usa.mp4 --show
```
```bash
python main.py --source video/japan.mp4 --save output/japan.mp4 --show
```
```bash
python main.py --source video/highway.mp4 --save output/highway.mp4 --show
```

Speed tip:
- Set `seg_every = 2` or `3` in `config.py` to run segmentation less often.
