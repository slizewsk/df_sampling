from df_sampling.core_imports import np

R = np.array([  [-0.05487395617553902, -0.8734371822248346,-0.48383503143198114],
                [0.4941107627040048, -0.4448286178025452,0.7469819642829028],
                [-0.8676654903323697, -0.1980782408317943,0.4559842183620723]]) 

H = np.array([  [0.9999967207734917, 0.0, 0.002560945579906427],
                [0.0, 1.0, 0.0],
                [-0.002560945579906427, 0.0, 0.9999967207734917]])

offsett = np.array([8.112,0,0.0208])
solarmotion=np.array([12.9/100, 245.6/100, 7.78/100])

def cart2sph_pos(xyz):
    """Convert Cartesian coordinates to spherical coordinates."""
    x, y, z = xyz
    D = np.sqrt(x**2 + y**2)  # Projected distance
    r = np.sqrt(x**2 + y**2 + z**2)  # Radial distance
    phi = np.arctan2(y, x)  # Azimuthal angle
    theta = np.arctan(z / D)  # Polar angle
    return np.array([r, theta, phi])

def sph2cart_pos(rtp):
    """Convert spherical coordinates to Cartesian coordinates."""
    r, theta, phi = rtp
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return np.array([x, y, z])

def cart2sph_vel(vel_cart, xyz):
    """Convert velocity from Cartesian to spherical coordinates."""
    x, y, z = xyz
    vx, vy, vz = vel_cart
    dist, theta, phi = cart2sph_pos(xyz)
    proj_dist = np.sqrt(x**2 + y**2) 
    vr = np.dot(xyz, vel_cart) / dist
    mu_theta = ((z * (x * vx + y * vy) - proj_dist**2 * vz)) / (dist**2 * proj_dist)
    vtheta = -mu_theta * dist
    mu_phi = (x * vy - y * vx) / proj_dist**2
    vphi = mu_phi * dist * np.cos(theta)
    return np.array([vr, vtheta, vphi])

def sph2cart_vel(vel_sph, rtp): 
    """Convert velocity from spherical to Cartesian coordinates."""
    r, theta, phi = rtp
    sph_mat = np.array([
        [np.cos(phi) * np.cos(theta), -np.cos(phi) * np.sin(theta), -np.sin(phi)],
        [np.sin(phi) * np.cos(theta), -np.sin(phi) * np.sin(theta), np.cos(phi)],
        [np.sin(theta), np.cos(theta),  0] 
    ])
    return np.dot(sph_mat, vel_sph)

def eq2gc_pos(r_eq, R=R, H=H, offsett=offsett, return_cart=False):
    """Convert equatorial to galactic coordinates."""
    dist, ra, dec = r_eq
    r_icrs = sph2cart_pos(np.array([dist, dec, ra]))
    r_gal = np.dot(R, r_icrs) - offsett
    r_gal = np.dot(H, r_gal)
    return r_gal if return_cart else cart2sph_pos(r_gal)

def gc2eq_pos(xyz, R=R, H=H, offsett=offsett, return_cart=False):
    """Convert galactic to equatorial coordinates."""
    r_gal = np.dot(np.linalg.inv(H), xyz) + offsett
    r_icrs = np.dot(np.linalg.inv(R), r_gal)
    return r_icrs if return_cart else cart2sph_pos(r_icrs)

def eq2gc_vel(vel_eq, pos_eq, R=R, H=H, offsett=offsett, solarmotion=solarmotion, return_cart=False):
    """Convert velocity from equatorial to galactic frame."""
    dist, ra, dec = pos_eq
    vlos, pmra, pmdec = vel_eq
    conversion_factor = 4.740470463533349
    dist_with_conversion = dist * conversion_factor
    vra = (dist_with_conversion * pmra) / 100
    vdec = (dist_with_conversion * pmdec) / 100
    r_gal = eq2gc_pos(pos_eq, R, H, offsett, return_cart=True)
    v_icrs = sph2cart_vel(np.array([vlos, vdec, vra]), np.array([dist, dec, ra]))
    v_gal = np.dot(H @ R, v_icrs) + solarmotion
    return v_gal if return_cart else cart2sph_vel(v_gal, r_gal)

def gc2eq_vel(vel_gc, pos_gc, R=R, H=H, offsett=offsett, solarmotion=solarmotion, in_cart=False, return_cart=False):
    """Convert velocity from galactic to equatorial frame."""
    if not in_cart:
        pos_gc = sph2cart_pos(pos_gc)
        vel_gc = sph2cart_vel(vel_gc, pos_gc)
    dist, ra, dec = gc2eq_pos(pos_gc, R, H, offsett, return_cart=False)
    xyz_helio = sph2cart_pos(np.array([dist, dec, ra]))
    v_gal = vel_gc - solarmotion
    v_icrs = np.dot(np.linalg.inv(H @ R), v_gal)
    return v_icrs * 100 if return_cart else cart2sph_vel(v_icrs, xyz_helio)
