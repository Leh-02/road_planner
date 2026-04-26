# Central Road Routing (Segmentation + YOLO + A*), no manual BEV points


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
```bash
python main.py --source video/s_us.mp4 --save output/s_us.mp4
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
```bash
python main.py --source video/s_us.mp4 --save output/s_us.mp4 --show
```
Speed tip:
- Set `seg_every = 2` or `3` in `config.py` to run segmentation less often.
