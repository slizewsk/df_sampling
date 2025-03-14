from .core_imports import np

def cart2sph_pos(xyz):
    x,y,z = xyz
    D = np.sqrt(x**2+y**2) # projected dist
    r = np.sqrt(x**2 + y**2 + z**2)  #radial dist
    phi = np.arctan2(y, x) #azimuthal angle
    theta =  np.arctan(z / D) #polar angle
#     if phi < 0:
#         phi += 2 * np.pi
#     same as 
#     theta =  np.arcsin(z / r) #polar angle
    return np.array([r,theta,phi])
def sph2cart_pos(rtp):
    r,theta,phi = rtp
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return np.array([x, y, z])

def cart2sph_vel(vel_cart,xyz):
    x,y,z = xyz
    vx,vy,vz = vel_cart
    dist,theta,phi = cart2sph_pos(xyz);
    proj_dist = np.sqrt(x**2 + y**2) 
    vr = np.dot(xyz,vel_cart)/dist
    mu_theta = ((z * (x * vx + y * vy) - np.square(proj_dist) * vz)) / np.square(dist) / proj_dist
    vtheta = -mu_theta * dist

    mu_phi = (x * vy - y * vx) / np.square(proj_dist)
    vphi = mu_phi * dist * np.cos(theta)
    
    vel_sph = np.array([vr,vtheta,vphi])
    return vel_sph
def sph2cart_vel(vel_sph,rtp): 
    r, theta,phi = rtp
    sph_mat = np.array([[np.cos(phi) * np.cos(theta), -np.cos(phi) * np.sin(theta), -np.sin(phi)],
                        [np.sin(phi) * np.cos(theta), -np.sin(phi) * np.sin(theta), np.cos(phi)],
                        [np.sin(theta), np.cos(theta),  0] ])
    vel_cart = np.dot(sph_mat,vel_sph)
    return vel_cart

def eq2gc_pos(r_eq,
                R =np.array([[-0.05487395617553902, -0.8734371822248346,-0.48383503143198114],
                                [0.4941107627040048, -0.4448286178025452,0.7469819642829028],
                                [-0.8676654903323697, -0.1980782408317943,0.4559842183620723]]) , 
                H = np.array([[0.9999967207734917, 0.0, 0.002560945579906427],
                                [0.0, 1.0, 0.0],
                                [-0.002560945579906427, 0.0, 0.9999967207734917]]), 
                offsett = np.array([8.112,0,0.0208]),return_cart=False):
        dist,ra,dec = r_eq
        # print('pos_eq',r_eq)
        r_icrs = sph2cart_pos(np.array([dist,dec,ra]))
        # print('xyz_helio',r_icrs)   
        r_gal = np.dot(R,r_icrs)
        # print('rot1',r_gal)
        r_gal -= offsett
        r_gal = np.dot(H ,r_gal)
        if return_cart: 
                return r_gal
        # print('gc xyz',r_gal)
        else: 
                r,theta,phi = cart2sph_pos(r_gal)
                return np.array([r,theta,phi])    
def gc2eq_pos(xyz,
                        R = np.array([[-0.05487395617553902, -0.8734371822248346,-0.48383503143198114],
                                [0.4941107627040048, -0.4448286178025452,0.7469819642829028],
                                [-0.8676654903323697, -0.1980782408317943,0.4559842183620723]]) , 
                        H = np.array([[0.9999967207734917, 0.0, 0.002560945579906427],
                                [0.0, 1.0, 0.0],
                                [-0.002560945579906427, 0.0, 0.9999967207734917]]), 
                        offsett = np.array([8.112,0,0.0208]),return_cart=False):
        # print('gc xyz',xyz)
        H_inv = np.linalg.inv(H)
        R_inv = np.linalg.inv(R)
        # H_inv =  np.dot(np.array([[-1, 0, 0],[0, 1, 0],[0, 0, -1]]),H)
        # R_inv =  np.dot(np.array([[-1, 0, 0],[0, 1, 0],[0, 0, -1]]),R)
        r_gal = np.dot(H_inv,xyz)
        r_gal += offsett
        # print('rot1',r_gal)
        r_icrs = np.dot(R_inv, r_gal)
        if return_cart: 
                return r_icrs
        else:
                r,dec,ra = cart2sph_pos(r_icrs)
                return np.array([r,ra,dec])    

def eq2gc_vel(vel_eq,pos_eq,
                        R =np.array([[-0.05487395617553902, -0.8734371822248346,-0.48383503143198114],
                                [0.4941107627040048, -0.4448286178025452,0.7469819642829028],
                                [-0.8676654903323697, -0.1980782408317943,0.4559842183620723]]) , 
                        H = np.array([[0.9999967207734917, 0.0, 0.002560945579906427],
                                [0.0, 1.0, 0.0],
                                [-0.002560945579906427, 0.0, 0.9999967207734917]]), 
                        offsett = np.array([8.112,0,0.0208]),
                        solarmotion=np.array([12.9/100, 245.6/100, 7.78/100]),
                        return_cart=False):
    dist,ra,dec = pos_eq
    vlos,pmra,pmdec = vel_eq
    vlos /= 100
    conversion_factor = 4.740470463533349
    dist_with_conversion = dist * conversion_factor
    
    vra  = (dist_with_conversion * pmra ) / 100
    vdec = (dist_with_conversion * pmdec) / 100

    r_gal = eq2gc_pos(pos_eq,return_cart=True)
    v_icrs = sph2cart_vel(rtp=np.array([dist,dec,ra ]), vel_sph=np.array([vlos, vdec, vra]))
    A = H @ R
    v_gal = A @ v_icrs
    v_gal += solarmotion
    v_gal_sph = cart2sph_vel(xyz=r_gal,vel_cart=v_gal)
    if return_cart:
        return v_gal
    else:
        return v_gal_sph
def gc2eq_vel(vel_gc,pos_gc,
                        R =np.array([[-0.05487395617553902, -0.8734371822248346,-0.48383503143198114],
                                [0.4941107627040048, -0.4448286178025452,0.7469819642829028],
                                [-0.8676654903323697, -0.1980782408317943,0.4559842183620723]]) , 
                        H = np.array([[0.9999967207734917, 0.0, 0.002560945579906427],
                                [0.0, 1.0, 0.0],
                                [-0.002560945579906427, 0.0, 0.9999967207734917]]), 
                        offsett = np.array([8.112,0,0.0208]),solarmotion= np.array([12.9/100, 245.6/100, 7.78/100]),
                        in_cart=False,return_cart=False):
        if in_cart:
                ricrs = pos_gc
                v_gal = vel_gc
        else: 
                ricrs = sph2cart_pos(pos_gc)
                v_gal = sph2cart_vel(vel_sph=vel_gc,rtp=pos_gc)
        # 
        dist,ra,dec = gc2eq_pos(ricrs, return_cart=False)
        xyz_helio = sph2cart_pos(np.array([dist,dec,ra]))
        # print('distance',dist)
        H_inv = np.linalg.inv(H)
        R_inv = np.linalg.inv(R)
        # input spherical velocity and position
        # print('v_gal',v_gal)
        v_gal -= solarmotion
        A = np.dot(R_inv,H_inv)
        # v_icrs = np.linalg.inv(A) @ v_gal
        v_icrs = A @ v_gal
        # print('v_icrs',v_icrs)
        if return_cart:
                return v_icrs*100
        else:
                vlos,vtheta,vphi = cart2sph_vel(vel_cart = v_icrs, xyz = xyz_helio)
                # vlos,vtheta,vphi = cart2sph_vel(vel_cart = np.array([v_icrs[0],v_icrs[1],v_icrs[2]]),
                                                # xyz = np.array([xyz_helio[0],xyz_helio[1],xyz_helio[2]]))
                
                conversion_factor = 4.740470463533349
                dist_with_conversion = dist * conversion_factor
                # print('vra',vphi)
                vtheta *= 100
                vphi *= 100
                vlos *= 100
                pmra = (vphi) / dist_with_conversion  
                pmdec = (vtheta) / dist_with_conversion  
                vel_eq = np.array([vlos,pmra,pmdec])
                # print("v_eq:", vel_eq)
                return vel_eq



# # defining transformation matrices to ref/call
# R = np.array([[-0.05487395617553902, -0.8734371822248346, -0.48383503143198114],
#           [0.4941107627040048,   -0.4448286178025452,  0.7469819642829028],
#           [-0.8676654903323697,  -0.1980782408317943,  0.4559842183620723]])
# H = np.array([[0.9999967207734917, 0.0, 0.002560945579906427],
#               [0.0, 1.0, 0.0],
#               [-0.002560945579906427, 0.0, 0.9999967207734917]])
# H_inv =  np.dot(np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]),H)
# R_inv =  np.dot(np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]),R)

# these four are from jeffs code
# transforms cartesian <-> spherical positions or velocities
# def c2s_pos(x, y, z):
#     dist = np.sqrt(np.dot(np.array([x, y, z]),np.array([x, y, z])))
#     phi = np.arctan2(y, x)
#     theta = np.arcsin(z / dist)
#     return np.array([dist, theta, phi])
# def s2c_pos(r, theta, phi):
#     x = r * np.cos(phi) * np.cos(theta)
#     y = r * np.sin(phi) * np.cos(theta)
#     z = r * np.sin(theta)
#     return np.array([x, y, z])

# def c2s_vel(x, y, z, vx, vy, vz):
#     sph_pos = c2s_pos(x, y, z);
#     dist = sph_pos[0]
#     lat = sph_pos[1]
#     lon = sph_pos[2]

#     proj_dist = np.sqrt(np.square(x) + np.square(y))
    
#     vr = np.dot(np.array([x, y, z]), np.array([vx, vy, vz])) / dist
    
#     mu_theta = ((z * (x * vx + y * vy) -
#                   np.square(proj_dist) * vz)
#                   ) / np.square(dist) / proj_dist
#     vtheta = -mu_theta * dist
    
#     mu_phi = (x * vy - y * vx) / np.square(proj_dist)
#     vphi = mu_phi * dist * np.cos(lat)
    
#     return np.array([vr, vtheta, vphi])
# def s2c_vel(r, theta, phi, vr, vtheta,vphi):
#     vx = vr * np.cos(phi) * np.cos(theta) - vphi * np.sin(phi)- vtheta * np.cos(phi) * np.sin(theta);
#     vy = vr * np.sin(phi) * np.cos(theta) + vphi * np.cos(phi)- vtheta * np.sin(phi) * np.sin(theta);
#     vz = vr * np.sin(theta) + vtheta * np.cos(theta);
#     return np.array([vx, vy, vz])

# # two more from jeff's code, plus my reverse of his 
# def jeff_transform_pos(ra, dec, dist, 
#                         R =np.array([[-0.05487395617553902, -0.8734371822248346,-0.48383503143198114],
#                                 [0.4941107627040048, -0.4448286178025452,0.7469819642829028],
#                                 [-0.8676654903323697, -0.1980782408317943,0.4559842183620723]]),
#                         H =np.array([[0.9999967207734917, 0.0, 0.002560945579906427],
#                                 [0.0, 1.0, 0.0],
#                                 [-0.002560945579906427, 0.0, 0.9999967207734917]]), 
#                         offsett = np.array([8.112,0,0.0208])):
#     r_icrs = s2c_pos(dist,dec,ra)
#     r_gal = R @ r_icrs
#     r_gal -= offsett
#     r_gal = H @ r_gal
#     return r_gal
# def jeff_reverse_transform_pos(r_gal, 
#                                R=np.array([[-0.05487395617553902, -0.8734371822248346,-0.48383503143198114],
#                                 [0.4941107627040048, -0.4448286178025452,0.7469819642829028],
#                                 [-0.8676654903323697, -0.1980782408317943,0.4559842183620723]]), 
#                                 H=np.array([[0.9999967207734917, 0.0, 0.002560945579906427],
#                                 [0.0, 1.0, 0.0],
#                                 [-0.002560945579906427, 0.0, 0.9999967207734917]]), 
#                                 offsett=np.array([8.112, 0, 0.0208]),return_cart=False):
#     r_helio = np.linalg.inv(H) @ r_gal
#     r_helio += offsett
#     r_icrs = np.linalg.inv(R) @ r_helio
#     if return_cart:
#         return r_icrs
#     else:
#         dist, dec, ra = c2s_pos(*r_icrs)  # assumes `c2s_pos` converts Cartesian to spherical
#         return np.array([ra,dec,dist])

# def jeff_transform_vel(ra, dec, dist, pmra, pmdec, vlos, 
#                         R = np.array([[-0.05487395617553902, -0.8734371822248346,-0.48383503143198114],
#                             [0.4941107627040048, -0.4448286178025452,0.7469819642829028],
#                             [-0.8676654903323697, -0.1980782408317943,0.4559842183620723]]), 
#                         H = np.array([[0.9999967207734917, 0.0, 0.002560945579906427],
#                             [0.0, 1.0, 0.0],
#                             [-0.002560945579906427, 0.0, 0.9999967207734917]]), 
#                         offsett=np.array([8.112, 0, 0.0208]), solarmotion=np.array([12.9/100, 245.6/100, 7.78/100]),return_cart=True):
#         vlos /= 100
#         conversion_factor = 4.740470463533349
#         dist_with_conversion = dist * conversion_factor
        
#         vra = dist_with_conversion * pmra / 100
#         vdec = dist_with_conversion * pmdec / 100

#         r_gal = jeff_transform_pos(ra, dec, dist, R, H, offsett)
#         v_icrs = s2c_vel(dist, dec, ra, vlos, vdec, vra)
#         v_gal = H @ R @ v_icrs
#         v_gal += solarmotion
#         if return_cart:return v_gal
#         else:
#                 v_sph = c2s_vel(r_gal[0], r_gal[1], r_gal[2],v_gal[0], v_gal[1], v_gal[2])
#                 return v_sph
# def jeff_reverse_transform_vel(r_gal, v_gal, R=np.array([[-0.05487395617553902, -0.8734371822248346,-0.48383503143198114],
#                                 [0.4941107627040048, -0.4448286178025452,0.7469819642829028],
#                                 [-0.8676654903323697, -0.1980782408317943,0.4559842183620723]]),
#                                 H=np.array([[0.9999967207734917, 0.0, 0.002560945579906427],
#                                 [0.0, 1.0, 0.0],
#                                 [-0.002560945579906427, 0.0, 0.9999967207734917]]),
#                                   offsett=np.array([8.112, 0, 0.0208]), solarmotion=np.array([12.9/100, 245.6/100, 7.78/100])):
#     v_gal -= solarmotion
#     v_icrs = np.linalg.inv(H @ R) @ v_gal
#     xyz_helio = jeff_reverse_transform_pos(r_gal,return_cart=True) 
#     ra,dec,dist = jeff_reverse_transform_pos(r_gal,return_cart=False) 
#     vlos, vdec, vra = c2s_vel(*xyz_helio, *v_icrs)  
#     pmra = vra / (dist * 4.740470463533349) * 100
#     pmdec = vdec / (dist * 4.740470463533349) * 100
#     vlos *= 100
#     return np.array([vlos, pmra, pmdec])
