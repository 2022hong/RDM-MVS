import os.path
import shutil
import imageio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import sys

def save_pfm(file, image, scale=1):
    file = open(file, mode='wb')

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(bytes('PF\n' if color else 'Pf\n', encoding='utf8'))
    file.write(bytes('%d %d\n' % (image.shape[1], image.shape[0]), encoding='utf8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(bytes('%f\n' % scale, encoding='utf8'))

    image_string = image.tostring()
    file.write(image_string)

    file.close()



path = './WHU_MVS_dataset/train'
out_path = './WHU_MVS_dataset/train/train_new'
os.makedirs(out_path, exist_ok=True)
cam_path = os.path.join(path, 'Cams')
dep_path = os.path.join(path, 'Depths')

files_list = os.listdir(cam_path)
for file in files_list:
    for i in range(5):
        cam_dir = os.path.join(path, 'Cams', file, '{}'.format(i))
        for name in os.listdir(cam_dir):
            cam_name = os.path.join(path, 'Cams', file, '{}'.format(i),name)
            dep_name = os.path.join(path, 'Depths', file, '{}'.format(i), '{}.png'.format(name[:-4]))
            img_name = os.path.join(path, 'Images', file, '{}'.format(i), '{}.png'.format(name[:-4]))
            os.makedirs(os.path.join(out_path, 'image'), exist_ok=True)
            os.makedirs(os.path.join(out_path, 'depth'), exist_ok=True)
            os.makedirs(os.path.join(out_path, 'camera'), exist_ok=True)
            os.makedirs(os.path.join(out_path, 'image', '{}'.format(i)), exist_ok=True)
            os.makedirs(os.path.join(out_path, 'depth', '{}'.format(i)), exist_ok=True)
            os.makedirs(os.path.join(out_path, 'camera', '{}'.format(i)), exist_ok=True)
            # cam = np.loadtxt(cam_name)
            with open(cam_name, "r") as f:
                all_text = f.read().splitlines()

            E = np.array([[float(at) for at in all_text[1].split(" ")],
                          [float(at) for at in all_text[2].split(" ")],
                          [float(at) for at in all_text[3].split(" ")],
                          [float(at) for at in all_text[4].split(" ")]])

            K_text = [float(at) for at in all_text[6].split(" ")]
            K = np.array([[-K_text[0], 0, K_text[1]],
                          [0, K_text[0], K_text[2]],
                          [0, 0, 1]])

            d_min, d_max, d_inter = [float(at) for at in all_text[8].split(" ")]
            cam = all_text[9]

            cam_out = os.path.join(out_path, 'camera', '{}'.format(i), '{}_{}.txt'.format(file,name[:-4]))
            text_ = ""
            text_ += str(E[0][0]) + " " + str(E[0][1]) + " " + str(E[0][2]) + " " + str(E[0][3]) + "\n"
            text_ += str(E[1][0]) + " " + str(E[1][1]) + " " + str(E[1][2]) + " " + str(E[1][3]) + "\n"
            text_ += str(E[2][0]) + " " + str(E[2][1]) + " " + str(E[2][2]) + " " + str(E[2][3]) + "\n"
            text_ += "0 0 0 1\n\n"

            text_ += str(-K[0, 0]) + " " + str(K[0, 2]) + " " + str(K[1, 2]) + "\n\n"

            text_ += str(d_min) + " " + str(d_max) + " " + str(d_inter) + "\n"
            text_ += cam + "\n"

            with open(cam_out, "w") as f:
                f.write(text_)

            depimg = imageio.v2.imread(dep_name)
            depth = (np.float32(depimg) / 64.0)

            sou_file_img = img_name
            des_file_img = os.path.join(out_path, 'image', '{}'.format(i), '{}_{}.png'.format(file,name[:-4]))
            shutil.copy2(sou_file_img, des_file_img)

            save_pfm(os.path.join(out_path, 'depth', '{}'.format(i), '{}_{}.pfm'.format(file,name[:-4])), depth)
            # plt.imsave(os.path.join(out_path, 'depth', '{}'.format(i), '{}_{}.jpg'.format(file,name[:-4])), depth)


