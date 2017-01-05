# coding=utf-8

import numpy as np
import math, random, copy
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt

from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import find_boundaries, mark_boundaries
from skimage.util import img_as_float
import calendar, time

from projection import rotation_image, del_black_pixel
from mpl_toolkits.mplot3d import Axes3D


BORDER_COLOR = np.asarray([0, 254, 0])
_id = 0
def get_id():
    return int(calendar.timegm(time.gmtime()))


""" Detect the plan on the image and draw points from the plan on it
Input:
    - image: ndarray n*m*3
Output:
    - image: ndarray n*m*3
    - n_normal: ndarray 1*3: normal vector to the plan selected
"""
def find_line_sphere(image):
    print "find_line_sphere"
    img = copy.deepcopy(img_as_float(image[::4, ::4]))

    # Segmentation of the photo
    # Computes Felsenszwalb’s efficient graph based image segmentation
    # segments_fz = felzenszwalb(img, scale=150, sigma=0.7, min_size=350)
    segments_fz = felzenszwalb(img, scale=150, sigma=0.7, min_size=100)

    img_seg = mark_boundaries(img, segments_fz)
    plt.figure()
    plt.imshow(segments_fz)
    plt.savefig('figures\\' + str(_id) + '_segments_felzenszwalb.png')
    plt.imshow(img_seg)
    plt.savefig('figures\\' + str(_id) + '_img_seg_felzenszwalb.png')
    # plt.show()
    # return image, np.asarray([0, 1, 0])

    # Extract inner boundaries of segments found
    boundaries = find_boundaries(segments_fz, mode='inner')

    # Convert points to cartesian coordinate then apply a RANSAC detection of line
    line, n_normal = line_estimation(get_contour_3d_pts(boundaries))

    # Plot the line on the initial image
    image = convert_3d_pts_to_sphere(copy.deepcopy(image), line)

    return image, n_normal


""" Extract points which define all contours
Input:
    - matrix: n*m
Output:
    ndarray: nb_pt*3
"""
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
                # Image to Spherical coordinates
                theta = i*theta_step
                phi = (j - m_d2)*phi_step

                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                cos_phi = math.cos(phi)
                sin_phi = math.sin(phi)

                # Spherical to cartesian coordinates
                x = cos_phi*sin_theta
                y = sin_phi*sin_theta
                z = cos_theta

                list_points.append(np.asarray([x, y, z]))

    return np.asarray(list_points)


""" Draw the line on the image, and return the image
Input:
    - img: ndarray n*m*d
    - line_pts: ndarray: l*3
Output:
    - ndarray: n*m*d
"""
def convert_3d_pts_to_sphere(img, line_pts):
    print "convert_3d_pts_to_sphere"
    n, m, d = img.shape
    nb_pt = len(line_pts)

    theta_step = math.pi/float(n)
    phi_step = 2*math.pi/float(m)
    x_step = 1/theta_step
    y_step = 1/phi_step
    m_d2 = m/2.0

    N = 5   # 2*D = size of the square to draw for each point of the line

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

        # Convert Spherical to image coordinate
        x_img = int(theta*x_step) % n
        y_img = int(phi*y_step + m_d2) % m

        # Update the new Image, with a square around the point
        if x_img > N and x_img < n - N and y_img > N and y_img < m - N:
            for u in range(-N, N, 1):
                for v in range(-N, N, 1):
                    img[x_img + u, y_img + v] = BORDER_COLOR

    # plt.imshow(img)
    # plt.show()
    return img


""" Hough Tranform to detect plan which have (0,0,0) as inner point
Input:
    - pts: ndarray: n*3
    - tolerance: float: distance between points and the plan selected
Output:
    - label: ndarray n*3, of boolean: True define point on the plan
    - n_max: normal vector to the plan selected
"""
def hough_plane(pts, tolerance=0.0001):
    print "hough_plane"
    nb_pt = len(pts)

    nb_error = 0
    error_mean = 0
    N = 200

    a_min = -2
    a_max = 2
    a_step = (a_max - a_min) / float(N)
    b_min = -2
    b_max = 2
    b_step = (b_max - b_min) / float(N)
    a_nb = N
    b_nb = N

    accumulator = np.zeros((a_nb, b_nb))

    # Fill in the accumulator
    for i in range(0, nb_pt):
        x = pts[i, 0]
        y = pts[i, 1]
        z = pts[i, 2]

        div = (y*b_step)
        if abs(y) > 0.001:
            for a in range(0, a_nb):
                b = int((((z - (a_min + (a*a_step))*x) / y) - b_min)/b_step)

                # If the b in on the scope of study
                if b < N and b >= 0:
                    accumulator[a, b] += 1
                else:
                    nb_error += 1
                    # error_mean += b
    # print "[hough_plane]Nb out of scope: ", nb_error, error_mean/nb_error

    plt.figure()
    plt.imshow(accumulator)
    # plt.show()
    plt.savefig('figures\\' + str(_id) + '_accumulator_hough_plane.png')

    # Select the maximum point of the accumulator
    max_accu = 0
    n_max = [0, 0]
    for a in range(0, N):
        for b in range(0, N):
            if accumulator[a, b] > max_accu:
                max_accu = accumulator[a, b]
                n_max = [a_min + a*a_step, b_min + b*b_step, -1]

    print 'Accumulator peak', n_max, max_accu

    # Find points which are closed to the plan
    n_max = np.asarray(n_max)
    label = np.asarray([False for i in range(0, nb_pt)])
    for i in range(0, nb_pt):
        if (abs(np.dot(n_max, pts[i])) < tolerance):
            label[i] = 1
        else:
            label[i] = 0

    return label, n_max


""" RANSAC Tranform to detect plan which have (0,0,0) as inner point
Input:
    - pts: ndarray: n*3
    - nb_attempt: flaot: number of plan tested
    - tolerance: float: distance between points and the plan selected
Output:
    - label: ndarray n*3, of boolean: True define point on the plan
    - n_max: normal vector to the plan selected
"""
def ransac_plane(pts, nb_attempt=200, tolerance=0.001):
    print "ransac_plane"
    nb_pt = len(pts)

    r = np.asarray([0, 0, 0])
    max_nb_pt = 0
    n_max = 0

    for i in range(0, nb_attempt):
        nb_pt_near = 0

        # Select points to define a plan
        p = random.randint(0, nb_pt)
        q = random.randint(0, nb_pt)
        while p == q:
            q = random.randint(0, nb_pt)

        p = pts[p]
        q = pts[q]

        n = np.cross(p - q, p - r)
        n /= np.linalg.norm(n)

        # Count points which are near to the selected plan
        for j in range(0, nb_pt):
            if np.dot(n, pts[j] - p) < tolerance:
                nb_pt_near += 1

        if max_nb_pt < nb_pt_near:
            max_nb_pt = nb_pt_near
            n_max = n

    print nb_pt, max_nb_pt

    # Find points which are closed to the plan
    label = np.asarray([False for i in range(0, nb_pt)])
    for i in range(0, nb_pt):
        if (np.dot(n_max, pts[i] - p) < tolerance):
            label[i] = 1
        else:
            label[i] = 0

    return label, n_max


""" Apply the tranform to detect plan, try Ransac and Hough
Input:
    - xyz: ndarray, n*3
Output:
    - point_plan: m*3: points on the plan
    - vector_n: ndarray 1*3: normal vector to the plan selected
"""
def line_estimation(xyz):
    print "line_estimation"
    # robustly fit line only using inlier data with RANSAC algorithm
    # model_robust, inliers = ransac(xyz, LineModelND, min_samples=500,
    #                                residual_threshold=0.01, max_trials=10000)
    # print type(inliers)
    # inliers, vector_n = ransac_plane(xyz, nb_attempt=200, tolerance=0.00001)
    inliers, vector_n = hough_plane(xyz, tolerance=0.05)

    outliers = inliers == False

    print "Nb points",len(xyz), 'Nb points on the plan', len(xyz[inliers])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[inliers][:, 0], xyz[inliers][:, 1], xyz[inliers][:, 2], c='b', s=200,
               marker='o', label='Inlier data')
    ax.scatter(xyz[outliers][::8, 0], xyz[outliers][::8, 1], xyz[outliers][::8, 2], c='r',
               marker='o', label='Outlier data')
    ax.legend(loc='lower left')
    # plt.show()
    plt.savefig('figures\\' + str(_id) + '_detection_plane.png')

    return xyz[inliers], vector_n


""" Main function to load the file and apply the algorithm
"""
def detect_line(filename):
    print 'detect_line: ', filename
    global _id
    _id = get_id()
    img = imread(filename)
    filename = '.'.join(filename.split('.')[:-1])
    image, v_normal = find_line_sphere(img)
    imsave(filename + "_line_detect.jpg", image)
    image = rotation_image(img, v_normal)
    imsave(filename + "_horizontal.jpg", del_black_pixel(image))


if __name__ == '__main__':
    print 'Start: Horizon detection and correction'
    t0 = time.time()

    path = 'C:\Users\Yann\Pictures\Data\\'
    path = 'Data\\'
    filename = "360_1079.jpg"
    filename = "360_0878.jpg"
    filename = "360_0871.jpg"
    filename = "360_0215.jpg"
    filename = "360_0215_rot_zx2.jpg"
    filename = "360_0871_rot_zx2.jpg"
    # detect_line(path + filename)

    filename_list = ["360_0085.jpg", "360_0047.jpg", "360_0121.jpg", "360_0134.jpg", "360_0156.jpg"]
    for filename in filename_list:
        detect_line(path + filename)


    print("done in %0.3fs" % (time.time() - t0))
    print 'End'