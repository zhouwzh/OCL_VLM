from PIL import Image, UnidentifiedImageError
import glob
from tqdm import *

bad = []
for f in tqdm(glob.glob("/LifelongBenchTC/Samcam/**/*.jpg", recursive=True)):
    try:
        Image.open(f).verify()
    except Exception:
        bad.append(f)
print(f"Bad images: {len(bad)}")
for b in bad[:10]:
    print(b)

