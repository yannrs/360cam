# coding=utf-8

from scipy import misc
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np


# Import the raw Photo
path = 'C:\Users\Yann\Pictures\\'
# path = 'C:\Users\yann\Videos\Road trip\Graduation\\'
filename = '360_1079'
# filename = '360_1439'
# filename = '360_1440'
# filename = '360_1441'
path = 'Data\\'
filename = "360_1079"
extension = '.JPG'

face = misc.imread(path+filename+extension)
n,m,d = face.shape


tps_max = 45.0
max_height = 1663
transate_pixel_max = m
trans_step = 200


# create & translate the new image
def translate_image(img, rate):
    n,m,d = face.shape
    transate_pixel = int(m*rate)

    new_photo = np.roll(img[:max_height], transate_pixel, axis=1)

    # Save the new photo
    misc.imsave(path+filename+'_rot_'+str(rate)+extension, new_photo)


# for transate_pixel in range(0,transate_pixel_max, trans_step):
#     new_photo = np.roll(face, transate_pixel, axis=1)
#
#     # Save the new photo
#     misc.imsave(path+'copy_'+str(transate_pixel)+'_'+filename, new_photo)


def make_frame(t):
    tranlation = int(transate_pixel_max*t/tps_max)	
    # return np.roll(face[0:max_height:2,::2], tranlation, axis=1)
    return np.roll(face[0:max_height], tranlation, axis=1)


fig, ax = plt.subplots(1, figsize=(4, 4), facecolor=(1,1,1))
def make_frame_(t):
    tranlation = int(transate_pixel_max*t/tps_max)
    # ax.imshow(new_photo, cmap=plt.cm.gray)
    # return mplfig_to_npimage(fig)
    # return np.zeros((100, 100, 3))

    # plt.show()
    return np.roll(face[0:max_height], tranlation, axis=1)


def translate_image_(img, rate):
    n, m, d = face.shape
    transate_pixel = int(m*rate) #1500
    new_photo = np.roll(img, transate_pixel, axis=1)

    # plt.imshow(new_photo, cmap=plt.cm.gray)
    # plt.show()

    # Save the new photo
    misc.imsave(path+'copy_'+str(transate_pixel)+'_'+filename, new_photo)


# from scipy.ndimage.interpolation import zoom
# rate = 2
# # face_smaller = face.copy()
# # misc.imresize(face, (int(n*rate), int(m*rate)))
# # face_smaller.resize((int(n*rate), int(m*rate), 3))
# print 'zoom'
# face_smaller = zoom(face, rate)
#
# def make_frame_rescale(t):
#     tranlation = int(transate_pixel_max*t/tps_max)
#     return np.roll(face_smaller, tranlation, axis=1)






if __name__ == '__main__':
    print 'start'

    print 'step 1'

    ## Creation of rotated frame
    face = misc.imread(path+filename+extension)
    translate_image(face, 0)
    translate_image(face, 0.33)
    translate_image(face, 0.5)
    translate_image(face, 0.66)

    print 'step 2'

    ## Creation of the video or Gif
    # clip = mpy.VideoClip(make_frame_rescale, duration=tps_max)
    # clip = mpy.VideoClip(make_frame, duration=tps_max)
    # clip.write_videofile(path+filename+'_anim_4k'+'.mp4', fps=25, codec='libx264')
    # clip.write_gif(path+'animG_'+filename+'.gif', fps=15)
    # clip.write_gif(path+'animG2_'+filename+'.gif', fps=15, fuzz=5)

    print 'End'