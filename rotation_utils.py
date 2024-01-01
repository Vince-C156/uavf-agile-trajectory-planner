import numpy as np

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])

def quaternion_to_dcm(q):
    w, x, y, z = q
    
    # Prepare the components for the matrix
    x2, y2, z2, w2 = 2 * x, 2 * y, 2 * z, 2 * w
    xx, yy, zz = x2 * x, y2 * y, z2 * z
    xy, xz, yz = x2 * y, x2 * z, y2 * z
    xw, yw, zw = w2 * x, w2 * y, w2 * z

    # Construct the DCM using numpy arrays
    R = np.array([[1 - yy - zz, xy - zw, xz + yw],
                  [xy + zw, 1 - xx - zz, yz - xw],
                  [xz - yw, yz + xw, 1 - xx - yy]])
    
    return R