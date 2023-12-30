import numpy as np

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])

def quaternion_to_dcm(q):
    w, x, y, z = q
    
    # Calculate individual elements of the DCM
    xx = 2 * x * x
    xy = 2 * x * y
    xz = 2 * x * z
    xw = 2 * x * w
    
    yy = 2 * y * y
    yz = 2 * y * z
    yw = 2 * y * w
    
    zz = 2 * z * z
    zw = 2 * z * w
    
    # Construct the DCM
    R = np.array([[1 - yy - zz, xy - zw, xz + yw],
                  [xy + zw, 1 - xx - zz, yz - xw],
                  [xz - yw, yz + xw, 1 - xx - yy]])
    
    return R