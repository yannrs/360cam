# coding=utf-8

import numpy as np
import math, random
from scipy.misc import imread, imsave
from time import time

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt


from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import find_boundaries
from skimage.util import img_as_float


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import LineModelND, ransac

@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)


@adapt_rgb(hsv_value)
def sobel_hsv(image):
    return filters.sobel(image)


def clean_image_color(image):
    fig = plt.figure(figsize=(14, 7))
    ax_each = fig.add_subplot(121, adjustable='box-forced')
    ax_hsv = fig.add_subplot(122, sharex=ax_each, sharey=ax_each,
                             adjustable='box-forced')

    # We use 1 - sobel_each(image)
    # but this will not work if image is not normalized
    ax_each.imshow(rescale_intensity(1 - sobel_each(image)))
    ax_each.set_xticks([]), ax_each.set_yticks([])
    ax_each.set_title("Sobel filter computed\n on individual RGB channels")
    plt.show()


def test():
    # generate coordinates of line
    point = np.array([0, 0, 0], dtype='float')
    direction = np.array([1, 1, 1], dtype='float') / np.sqrt(3)
    xyz = point + 10 * np.arange(-100, 100)[..., np.newaxis] * direction

    # add gaussian noise to coordinates
    noise = np.random.normal(size=xyz.shape)
    xyz += 0.5 * noise
    xyz[::2] += 20 * noise[::2]
    xyz[::4] += 100 * noise[::4]

    print type(xyz), len(xyz), xyz[:5]
    print type(xyz[0])
    line_estimation(xyz)



def clean_image(image):
    edge_roberts = roberts(image)
    edge_sobel = sobel(image)

    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                           figsize=(8, 4))

    ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
    ax[0].set_title('Roberts Edge Detection')

    ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
    ax[1].set_title('Sobel Edge Detection')

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()



def find_line_sphere(image):
    print "find_line_sphere"
    img = img_as_float(image[::2, ::2])

    # Segmentation of the photo
    # Computes Felsenszwalb’s efficient graph based image segmentation
    segments_fz = felzenszwalb(img, scale=150, sigma=0.7, min_size=350)

    plt.imshow(segments_fz)
    plt.show()
    # Extract inner boundaries of segments found
    boundaries = find_boundaries(segments_fz, mode='inner')

    # Convert points to cartesian coordinate then apply a RANSAC detection of line
    line = line_estimation(get_contour_3d_pts(boundaries))

    # Plot the line on the initial image
    image = convert_3d_pts_to_sphere(image, line)

    return image


def get_contour_3d_pts(contour):
    print "get_contour_3d_pts"
    n, m = contour.shape

    theta_step = math.pi/float(n)
    phi_step = 2*math.pi/float(m)
    m_d2 = m/2.0

    list_points = []

    for i in range(0, n):
        for j in range(0, m):
            if contour[i, j] != 0:
                theta = i*theta_step
                phi = (j - m_d2)*phi_step

                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                cos_phi = math.cos(phi)
                sin_phi = math.sin(phi)
                x = cos_phi*sin_theta
                y = sin_phi*sin_theta
                z = cos_theta

                list_points.append(np.asarray([x, y, z]))

    return np.asarray(list_points)


BORDER_COLOR = np.asarray([0, 254, 0])

def convert_3d_pts_to_sphere(img, line_pts):
    print "convert_3d_pts_to_sphere"
    n, m, d = img.shape
    nb_pt = len(line_pts)

    theta_step = math.pi/float(n)
    phi_step = 2*math.pi/float(m)
    x_step = 1/theta_step
    y_step = 1/phi_step
    m_d2 = m/2.0

    N = 5

    for i in range(0, nb_pt):
        x = line_pts[i][0]
        y = line_pts[i][1]
        z = line_pts[i][2]

        # Convert to Spherical coordinate
        if z <= -1:
            theta = 0
        elif z >= 1:
            theta = math.pi
        else:
            theta = math.acos(z)
        phi = math.atan2(y, x)

        # Update the new Image
        x_img = int(theta*x_step) % n
        y_img = int(phi*y_step + m_d2) % m

        img[x_img, y_img] = BORDER_COLOR

        if x_img > N and x_img < n - N and y_img > N and y_img < m - N:
            for u in range(-N, N, 1):
                for v in range(-N, N, 1):
                    img[x_img + u, y_img + v] = BORDER_COLOR

    return img


def hough_plane(pts, tolerance=0.0001):
    print "hough_plane"
    nb_pt = len(pts)

    nb_error = 0
    error_mean = 0
    N = 200

    a_min = -5
    a_max = 5
    a_step = (a_max - a_min) / float(N)
    b_min = -5
    b_max = 5
    b_step = (b_max - b_min) / float(N)
    a_nb = N
    b_nb = N

    accumulator = np.zeros((a_nb, b_nb))
    for i in range(0, nb_pt):
        x = pts[i, 0]
        y = pts[i, 1]
        z = pts[i, 2]

        div = (y*b_step)
        if abs(div) > 0.00001:
            for a in range(0, a_nb):
                b = int((z - (a_min + (a*a_step))*x + b_min) / div)

                if b < N and b >= 0:
                    accumulator[a, b] += 1
                else:
                    nb_error += 1
                    error_mean += b
    print "[hough_plane]Nb error: ", nb_error, error_mean/nb_error

    plt.imshow(accumulator)
    plt.show()

    max_accu = 0
    n_max = [0, 0]
    for a in range(0, N):
        for b in range(0, N):
            if accumulator[a, b] > max_accu:
                max_accu = accumulator[a, b]
                n_max = [a_min + a*a_step, b_min + b*b_step, -1]

    print n_max, max_accu

    n_max = np.asarray(n_max)
    label = np.asarray([False for i in range(0, nb_pt)])
    for i in range(0, nb_pt):
        if (abs(np.dot(n_max, pts[i])) < tolerance):
            label[i] = 1
            # print pts[i], np.dot(n_max, pts[i])
        else:
            label[i] = 0

    return label, n_max


def ransac_plane(pts, nb_attempt=200, tolerance=0.001):
    print "ransac_plane"
    nb_pt = len(pts)

    r = np.asarray([0, 0, 0])
    max_nb_pt = 0
    nb_pt_near = np.asarray([0, 0, 0])
    n_max = 0

    for i in range(0, nb_attempt):
        nb_pt_near = 0

        p = random.randint(0, nb_pt)
        q = random.randint(0, nb_pt)
        while p == q:
            q = random.randint(0, nb_pt)

        p = pts[p]
        q = pts[q]

        n = np.cross(p - q, p - r)
        n /= np.linalg.norm(n)

        for j in range(0, nb_pt):
            if np.dot(n, pts[j] - p) < tolerance:
                nb_pt_near += 1

        if max_nb_pt < nb_pt_near:
            max_nb_pt = nb_pt_near
            n_max = n

    print nb_pt, max_nb_pt

    label = np.asarray([False for i in range(0, nb_pt)])
    for i in range(0, nb_pt):
        if (np.dot(n_max, pts[i] - p) < tolerance):
            label[i] = 1
        else:
            label[i] = 0
    return label, n_max


def line_estimation(xyz):
    print "line_estimation"
    print type(xyz), len(xyz)#, xyz[:5]
    # robustly fit line only using inlier data with RANSAC algorithm
    # model_robust, inliers = ransac(xyz, LineModelND, min_samples=500,
    #                                residual_threshold=0.01, max_trials=10000)
    # print type(inliers)
    # inliers, vector_n = ransac_plane(xyz, nb_attempt=200, tolerance=0.00001)
    inliers, vector_n = hough_plane(xyz, tolerance=0.1)

    outliers = inliers == False

    print len(xyz[inliers])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[inliers][:, 0], xyz[inliers][:, 1], xyz[inliers][:, 2], c='b', s=200,
               marker='o', label='Inlier data')
    ax.scatter(xyz[outliers][::8, 0], xyz[outliers][::8, 1], xyz[outliers][::8, 2], c='r',
               marker='o', label='Outlier data')
    ax.legend(loc='lower left')
    plt.show()

    return xyz[inliers]


def detect_line(filename):
    img = imread(filename)
    filename = '.'.join(filename.split('.')[:-1])
    image = find_line_sphere(img)
    imsave(filename + "_line_detect.jpg", image)


if __name__ == '__main__':
    print 'start: Projection'
    t0 = time()

    path = 'C:\Users\Yann\Pictures\Data\\'
    path = 'Data\\'
    filename = "360_1079.jpg"
    filename = "360_0878.jpg"
    filename = "360_0871.jpg"
    filename = "360_0215.jpg"
    detect_line(path + filename)

    print("done in %0.3fs" % (time() - t0))
    print 'End'