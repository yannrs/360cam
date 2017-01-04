# coding=utf-8

import numpy as np
import math
from time import time
from scipy.misc import imread,imsave

############################################################################
#                               Projection
############################################################################

def projection_simple(img):
    print 'Projection: simple'
    n, m, d = img.shape
    image = np.zeros([n, m, 3])

    theta_step = math.pi/float(n)
    phi_step = 2*math.pi/float(m)
    x_step = 1/theta_step
    y_step = 1/phi_step

    nb_error = 0

    for i in range(0, n):
        for j in range(0, m):
            theta = i*theta_step
            phi = (j - (m/2))*phi_step

            # Projection
            X = theta * math.cos(phi)
            Y = theta

            x = int(X*x_step)
            y = int(Y*y_step + (m/2))

            image[x % n, y % m] = img[i, j]

    return image

def projection_cylindrique_classique(img):
    print 'Projection: projection_cylindrique_classique'
    n, m, d = img.shape
    image = np.zeros([n, m, 3])

    theta_step = math.pi/float(n)
    phi_step = 2*math.pi/float(m)
    x_step = 1/theta_step
    y_step = 1/phi_step

    nb_error = 0
    pi_d6 = math.pi /6.0
    pi_5d6 = math.pi*5.0 /6.0
    n_d2 = n/2.0
    m_d2 = m/2.0
    pi_d2 = math.pi/2.0

    for i in range(0, n):
        theta = i*theta_step
        if theta > pi_d6 and theta < pi_5d6:
            X = math.tan(theta - pi_d2)
            for j in range(0, m):
                phi = (j - m_d2)*phi_step

                # Projection
                Y = phi

                x = int(X*x_step + n_d2)
                y = int(Y*y_step + m_d2)

                image[x % n, y % m] = img[i, j]

    return image


def projection_lambert(img):
    print 'Projection: Lambert'
    n, m, d = img.shape
    image = np.zeros([n,m,3])

    theta_step = math.pi/float(n/2)
    phi_step = math.pi/float(m/2)
    x_step = 1/float(theta_step)
    y_step = float(m/2)

    nb_error = 0

    for i in range(0, n):
        for j in range(0, m):
            theta = (i - (n/2))*theta_step
            phi = (j - (m/2))*phi_step

            # Projection
            X = theta
            Y = math.sin(phi)

            x = math.floor(X*x_step + (n/2.0))
            y = math.floor(Y*y_step + (m/2.0))

            if x >= 0 and x < n and y >= 0 and y < m:
                image[x, y] = img[i, j]
            else:
                nb_error += 1

    print 'Nb errors: ', nb_error
    return image


def projection_lambert2(img):
    print 'Projection: Lambert2'
    n, m, d = img.shape
    image = np.zeros([n, m, 3])

    theta_step = math.pi/float(n)
    phi_step = 2*math.pi/float(m)
    x_step = 1/float(theta_step)
    y_step = m/2.0

    nb_error = 0

    for i in range(0, n):
        for j in range(0, m):
            theta = i*theta_step
            phi = (j - m/2.0)*phi_step

            # Projection
            X = theta
            Y = math.sin(phi)

            x = int(X*x_step)
            y = int(Y*y_step)

            if x >= 0 and x < n and y >= 0 and y < m:
                image[x, y] = img[i, j]
            else:
                nb_error += 1

    print 'Nb errors: ', nb_error
    return image


def projection_hammer_aitoff(img):
    print 'Projection: Hammer-Aïtoff'
    n, m, d = img.shape
    image = np.zeros([n, m, 3])

    theta_step = math.pi/float(n)
    phi_step = math.pi/float(m)
    x_step = float(n)/2.0
    y_step = float(m)

    const = 2 * math.sqrt(2)
    nb_error = 0

    for i in range(0, n):
        for j in range(0, m):
            theta = (i - (n/2))*theta_step
            phi = (j - (m/2))*phi_step

            ## Projection Hammer-Aïtoff
            aux = math.sqrt(1 + math.cos(phi)*math.cos(theta/2.0))
            if aux == 0:
                aux = 1
            X = const * (math.cos(phi) * math.sin(theta/2.0)) / aux
            Y = math.sqrt(2) * math.sin(phi) / aux

            x = int(X*x_step + (n/2))
            y = int(Y*y_step + (m/2))

            if x > 0 and x < n and y > 0 and y < m:
                image[x, y] = img[i, j]
            else:
                nb_error += 1

    print 'Nb errors: ', nb_error
    return image


def projection_braun(img):
    print 'Projection: Braun'
    n, m, d = img.shape
    image = np.zeros([n, m, 3])

    theta_step = math.pi/float(n)
    phi_step = math.pi/float(m)
    x_step = 1/float(theta_step)
    y_step = float(m)

    nb_error = 0

    for i in range(0, n):
        for j in range(0, m):
            theta = (i - (n/2))*theta_step
            phi = (j - (m/2))*phi_step

            ## Projection de Braun
            X = theta
            Y = 2 * math.tan(phi/2.0)

            x = int(X*x_step + (n/2))
            y = int(Y*y_step + (m/2))

            if x > 0 and x < n and y > 0 and y < m:
                image[x, y] = img[i, j]
            else:
                nb_error += 1

    print 'Nb errors: ', nb_error
    return image


def projection_mercator(img):
    print 'Projection: Mercator'
    n, m, d = img.shape
    image = np.zeros([n, m, 3])

    theta_step = math.pi/float(n)
    phi_step = math.pi/float(m)
    x_step = 1/float(theta_step)
    y_step = 1/float(phi_step)

    nb_error = 0

    for i in range(0, n):
        for j in range(0, m):
            theta = (i - (n/2))*theta_step
            phi = (j - (m/2))*phi_step

            ## Projection de Mercator
            X = theta
            Y = math.log(math.tan(math.pi/4.0 + phi/2.0))

            x = int(X*x_step + (n/2))
            y = int(Y*y_step + (m/2))

            if x > 0 and x < n and y > 0 and y < m:
                image[x, y] = img[i, j]
            else:
                nb_error += 1

    print 'Nb errors: ', nb_error
    return image


def projection_conique(img):
    print 'Projection: conique'
    n, m, d = img.shape
    image = np.zeros([n, m, 3])

    theta_step = 2*math.pi/float(n)
    phi_step = math.pi/float(m)
    x_step = float(n)/(2*float(m))
    y_step = 1/2

    nb_error = 0

    for i in range(0, n):
        for j in range(0, m):
            theta = i*theta_step
            rho = j

            ## Projection de Mercator
            X = rho * math.sin(theta)
            Y = rho - rho * math.cos(theta)

            x = int(X*x_step)
            y = int(Y*y_step)

            if x > 0 and x < n and y > 0 and y < m:
                image[x, y] = img[i, j]
            else:
                # print x,y
                # print i,j
                nb_error += 1

    print 'Nb errors: ', nb_error
    return image


def projection(filename):
    img = imread(filename)
    filename = '.'.join(filename.split('.')[:-1])

    # image = projection_simple(img)
    # imsave(filename + "_proj_simple.jpg", image)
    #
    # image = projection_lambert(img)
    # imsave(filename + "_proj_lambert.jpg", image)
    #
    # image = projection_hammer_aitoff(img)
    # imsave(filename + "_proj_hammer_aitoff.jpg", image)
    #
    # image = projection_braun(img)
    # imsave(filename + "_proj_braun.jpg", image)
    # #
    # # image = projection_mercator(img)
    # # imsave(filename + "_proj_mercator.jpg", image)
    #
    # image = projection_lambert2(img)
    # imsave(filename + "_proj_lambert2.jpg", image)
    #
    # # image = projection_conique(img)
    # # imsave(filename + "_proj_conique.jpg", image)

    image = projection_cylindrique_classique(img)
    # imsave(filename + "_proj_cylindrique_classique.jpg", image)
    imsave(filename + "_proj_cylindrique_classique.jpg", del_black_pixel(image))


############################################################################
#                               ROTATION
############################################################################

def rotation_image(img):
    print 'rotation_image'
    n, m, d = img.shape
    image = np.zeros([n, m, 3])

    theta_step = math.pi/float(n)
    phi_step = 2*math.pi/float(m)
    x_step = 1/theta_step
    y_step = 1/phi_step

    angle_rotation = math.pi/4.0
    cos_angle = math.cos(angle_rotation)
    sin_angle = math.sin(angle_rotation)
    cos_angle2 = math.cos(2*angle_rotation)
    sin_angle2 = math.sin(2*angle_rotation)
    m_d2 = m/2.0

    for i in range(0, n):
        theta = i*theta_step            # [0, Pi]
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for j in range(0, m):
            # Compute coordinate of a point with spherical coordinate
            phi = (j - m_d2)*phi_step      # [-Pi, Pi]

            # Cartesian coordinate of the point
            x = math.cos(phi) * sin_theta
            y = math.sin(phi) * sin_theta
            z = cos_theta

            # Rotation
            # x2 = x
            # y2 = y * cos_theta - z * sin_theta
            # z2 = y * sin_theta + z * cos_theta
            x2, y2, z2 = rotation_3d_z(x, y, z, cos_angle2, -sin_angle2)
            x2, y2, z2 = rotation_3d_x(x2, y2, z2, cos_angle, sin_angle)

            # Convert to Spherical coordinate
            if z2 <= -1:
                theta2 = 0
            elif z2 >= 1:
                theta2 = math.pi
            else:
                theta2 = math.acos(z2)
            phi2 = math.atan2(y2, x2)

            # Update the new Image
            x_img = int(theta2*x_step)
            y_img = int(phi2*y_step + m_d2)

            image[x_img % n, y_img % m] = img[i, j]

    return image


def rotation_3d_x(x, y, z, cos_r, sin_r):
    x2 = x
    y2 = y * cos_r - z * sin_r
    z2 = y * sin_r + z * cos_r
    return x2, y2, z2


def rotation_3d_y(x, y, z, cos_r, sin_r):
    x2 = x * cos_r + z * sin_r
    y2 = y
    z2 = -x * sin_r + z * cos_r
    return x2, y2, z2


def rotation_3d_z(x, y, z, cos_r, sin_r):
    x2 = x * cos_r - y * sin_r
    y2 = x * sin_r + y * cos_r
    z2 = z
    return x2, y2, z2


def test_rotation(filename):
    img = imread(filename)
    filename = '.'.join(filename.split('.')[:-1])
    image = rotation_image(img)
    # imsave(filename + "_rot_x.jpg", image)
    imsave(filename + "_rot_zx2.jpg", del_black_pixel(image))

############################################################################
#                               Clean Image
############################################################################


def del_black_pixel(img):
    print 'del_black_pixel'
    n, m, d = img.shape

    for i in range(0, n):
        for j in range(0, m):
            if img[i, j, 0] == 0 and img[i, j, 1] == 0 and img[i, j, 2] == 0:
                img[i, j] = new_pixel2(img, i, j)

    return img


def new_pixel(img, i, j):
    n, m, d = img.shape
    pixel = img[i, j]
    nb_pixel = 0

    if i > 0:
        if j > 0:
            pixel += img[i-1, j-1]
            pixel += img[i, j-1]
            pixel += img[i-1, j]
            nb_pixel += 3
        else:
            pixel += img[i-1, j]
            nb_pixel += 1
    else:
        if j > 0:
            pixel += img[i, j-1]
            nb_pixel += 1

    if i < n-1:
        if j < m-1:
            pixel += img[i+1, j+1]
            pixel += img[i, j+1]
            pixel += img[i+1, j]
            nb_pixel += 3
        else:
            pixel += img[i+1, j]
            nb_pixel += 1
    else:
        if j < m-1:
            pixel += img[i, j+1]
            nb_pixel += 1

    if i > 0 and j < m - 1:
        pixel += img[i-1, j+1]
        nb_pixel += 1
    if i < n - 1 and j > 0:
        pixel += img[i+1, j-1]
        nb_pixel += 1

    pixel /= nb_pixel

    return pixel


def new_pixel2(img, i, j):
    n, m, d = img.shape
    pixel = img[i, j]
    nb_pixel = 0

    if i > 0:
        if j > 0:
            pixel, nb_pixel = check_color(pixel, nb_pixel, img, i-1, j-1)
            pixel, nb_pixel = check_color(pixel, nb_pixel, img, i, j-1)
            pixel, nb_pixel = check_color(pixel, nb_pixel, img, i-1, j)
        else:
            pixel, nb_pixel = check_color(pixel, nb_pixel, img, i-1, j)
    else:
        if j > 0:
            pixel, nb_pixel = check_color(pixel, nb_pixel, img, i, j-1)

    if i < n-1:
        if j < m-1:
            pixel, nb_pixel = check_color(pixel, nb_pixel, img, i+1, j+1)
            pixel, nb_pixel = check_color(pixel, nb_pixel, img, i, j+1)
            pixel, nb_pixel = check_color(pixel, nb_pixel, img, i+1, j)
        else:
            pixel, nb_pixel = check_color(pixel, nb_pixel, img, i+1, j)
    else:
        if j < m-1:
            pixel, nb_pixel = check_color(pixel, nb_pixel, img, i, j+1)

    if i > 0 and j < m - 1:
        pixel, nb_pixel = check_color(pixel, nb_pixel, img, i-1, j+1)
    if i < n - 1 and j > 0:
        pixel, nb_pixel = check_color(pixel, nb_pixel, img, i+1, j-1)

    if nb_pixel != 0:
        pixel /= nb_pixel

    return pixel


def check_color(pixel, nb, img, i, j):
    if not (img[i, j, 0] == 0 and img[i, j, 1] == 0 and img[i, j, 2] == 0):
        return pixel + img[i, j], nb + 1
    return pixel, nb

if __name__ == '__main__':
    print 'start: Projection'
    t0 = time()

    path = 'C:\Users\Yann\Pictures\Data\\'
    path = 'Data\\'
    filename = "360_1079.jpg"
    filename = "360_0878.jpg"
    filename = "360_0871.jpg"
    filename = "360_0215.jpg"
    projection(path + filename)
    # test_rotation(path + filename)

    print("done in %0.3fs" % (time() - t0))
    print 'End'