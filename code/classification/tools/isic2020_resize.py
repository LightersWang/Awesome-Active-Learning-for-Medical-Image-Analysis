import os
import argparse
from tqdm import tqdm
from glob import glob
from torchvision.io import read_image, write_jpeg, ImageReadMode
from torchvision import transforms


parser = argparse.ArgumentParser(description='ISIC 2020 image resizing')
parser.add_argument('--isic2020_root', type=str, default='code/classification/data/ISIC2020')
args = parser.parse_args()


ops = transforms.Compose([
    transforms.Resize([300, 300], antialias=True),
])

new_root = os.path.join(args.isic2020_root, 'ISIC_2020_Training_JPEG_300x300/train/')
os.makedirs(new_root, exist_ok=True)

all_img_path = sorted(glob(
    os.path.join(args.isic2020_root, 'ISIC_2020_Training_JPEG/train/*.jpg')))
print(len(all_img_path))
for img_path in tqdm(all_img_path):
    img = read_image(img_path, mode=ImageReadMode.RGB)  # tensor: 0.1s per image
    img = ops(img)
    write_jpeg(img, filename=os.path.join(new_root, os.path.basename(img_path)), quality=100)
