from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import random


# Import the raw Photo
path = 'C:\Users\Yann\Pictures\\'
filename = '360_1079.JPG'

face = misc.imread(path+filename)
n,m,d = face.shape


# create & translate the new image
transate_pixel = 1500
new_photo = np.roll(face, transate_pixel, axis=1)

# plt.imshow(new_photo, cmap=plt.cm.gray)
# plt.show()
0
# Save the new photo
misc.imsave(path+'copy_'+str(transate_pixel)+'_'+filename, new_photo)


transate_pixel_max = m
trans_step = 200

#
# for transate_pixel in range(0,transate_pixel_max, trans_step):
#     new_photo = np.roll(face, transate_pixel, axis=1)
#
#     # Save the new photo
#     misc.imsave(path+'copy_'+str(transate_pixel)+'_'+filename, new_photo)

print type(face)

import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage


tps_max = 15.0
fig, ax = plt.subplots(1, figsize=(4, 4), facecolor=(1,1,1))

max_height = 1663

def make_frame(t):
    tranlation = int(transate_pixel_max*t/tps_max)
    print tranlation
    # ax.imshow(new_photo, cmap=plt.cm.gray)
    # return mplfig_to_npimage(fig)
    # return np.zeros((100, 100, 3))

    # plt.show()
    return np.roll(face[0:max_height], tranlation, axis=1)
from scipy.ndimage.interpolation import zoom
rate = 2
# face_smaller = face.copy()
# misc.imresize(face, (int(n*rate), int(m*rate)))
# face_smaller.resize((int(n*rate), int(m*rate), 3))
print 'zoom'
face_smaller = zoom(face, rate)

def make_frame_rescale(t):
    tranlation = int(transate_pixel_max*t/tps_max)
    return np.roll(face_smaller, tranlation, axis=1)


if __name__ == '__main__':
    print 'start'
    clip = mpy.VideoClip(make_frame_rescale, duration=tps_max)
    clip.write_videofile(path+'animS_small_'+filename+'.mp4', fps=25, codec='libx264')
    # clip.write_gif(path+'animG_'+filename+'.gif', fps=15)
    # clip.write_gif(path+'animG2_'+filename+'.gif', fps=15, fuzz=5)
