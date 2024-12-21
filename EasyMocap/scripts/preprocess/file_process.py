import random
import os.path
import os
import glob
import argparse
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('path',type=str)
args = parser.parse_args()

def find_common_img(images):
    temp_dict = {}
    for key, values in images.items():
        for value in values:
            if value not in temp_dict:
                temp_dict[value] = 0
            temp_dict[value] += 1
    common_img = [value for value, count in temp_dict.items() if count > 1]
    return common_img

def get_file(path):
    images = {}
    calib_path = os.path.join(path, 'output', 'calibration')
    calib_files = glob.glob(os.path.join(calib_path, '*')) # *[1, 2, 3, 4]
    for file in calib_files:
        images[str(os.path.basename(file))] = []
        imgs = glob.glob(os.path.join(file, '*.jpg'))
        for img in imgs:
            images[str(os.path.basename(file))].append(os.path.basename(img))

    common_imgs = find_common_img(images)
    selected_img = common_imgs[0]
    imgs = glob.glob(os.path.join(path, 'images', '*'))
    for img in imgs:
        for pic in glob.glob(os.path.join(img, '*')):
            if os.path.basename(pic) != selected_img:
                os.remove(pic)
            
    for file in calib_files:
        index = os.path.basename(file)
        os.remove(os.path.join(imgs[int(index)-1], selected_img))
        shutil.move(os.path.join(file, selected_img), imgs[int(index)-1])
            

path = 'C:/MOCAP/EasyMocap/cam_group_2_deadlift/extri_test'
get_file(args.path)
