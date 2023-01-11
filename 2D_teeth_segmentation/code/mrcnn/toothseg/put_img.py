import cv2 as cv
import sys
import argparse

parser = argparse.ArgumentParser(description='Processing...')

parser.add_argument('--name', required=True,
        metavar='//',
        help="filename")

args = parser.parse_args()
print(args.name)

dir_prefix = '/home/boneage/toothapi/code/'
img_origin = dir_prefix + 'mrcnn/toothseg/yolo/data/custom/test_data/' + args.name + ".png"
crop_result = dir_prefix + 'mrcnn/toothseg/outputs/tooth_output.png'
save_path = dir_prefix + 'mrcnn/toothseg/outputs/'

#print(img_origin)
img = cv.imread(img_origin, cv.IMREAD_COLOR)
cropped = cv.imread(crop_result, cv.IMREAD_COLOR)
dst = img.copy()

read_txt = open(dir_prefix + 'mrcnn/toothseg/yolo/bbox_txt/' + args.name + '.txt', 'r')

for x in range(2):
    if x == 1:
        crop_x = read_txt.readline()[:-1]
    if x == 0:
        crop_y = read_txt.readline()[:-1]

c = cropped[0:cropped.shape[0],0:cropped.shape[1]]
dst[int(crop_x):int(crop_x)+int(cropped.shape[0]), int(crop_y):int(crop_y)+int(cropped.shape[1])] = c
cv.imwrite(save_path+'tooth_output.png', dst)


