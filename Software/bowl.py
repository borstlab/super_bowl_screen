import numpy as np
import cv2
import jax.numpy as jnp
import jax 
from jax import jit , lax

import pylab as plt


@jit
def rotate(src,dest,spher_coords,roll,pitch,yaw,width,height):
    pi = jnp.pi
    roll = jnp.deg2rad(roll)
    pitch = jnp.deg2rad(pitch)
    yaw = jnp.deg2rad(yaw)
    # perform the rotations using 3D rotation matrices
    R_x = jnp.array([[1, 0, 0], [0, jnp.cos(roll), -jnp.sin(roll)], [0, jnp.sin(roll), jnp.cos(roll)]])
    R_y = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)], [0, 1, 0], [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
    R_z = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0], [jnp.sin(yaw), jnp.cos(yaw), 0], [0, 0, 1]])
    R = jnp.dot(jnp.dot(R_x, R_y), R_z)
    
    spherical_coords = jnp.moveaxis(spher_coords, 0, -1)
    spherical_coords = spherical_coords.reshape(-1, 3)
    spherical_coords = jnp.dot(spherical_coords, R)
    spherical_coords = spherical_coords.reshape(*spher_coords[0].shape, 3)
    
    
    # convert spherical coordinates back to equirectangular coordinates
    lng = jnp.arctan2(spherical_coords[:,:,1], spherical_coords[:,:,0])
    lat = jnp.arctan2(spherical_coords[:,:,2], jnp.sqrt(spherical_coords[:,:,0]**2 + spherical_coords[:,:,1]**2))

    ix = (0.5 * lng / pi + 0.5) * width - 0.5
    iy = (lat / pi + 0.5) * height - 0.5
    dest = src[jnp.round(iy).astype(int), jnp.round((ix ) % width).astype(int),:]#+ width // 4
    return dest
    

class Stimulus():
    def __init__(self,img_size, fov_azi=0, fov_ele=0):
        
        self.width= img_size[1]
        self.height= img_size[0]
        self.fov_azi = fov_azi
        self.fov_ele = fov_ele
        pi = jnp.pi
        
        x, y = jnp.meshgrid(jnp.arange(self.width), jnp.arange(self.height))
        xx = 2 * (x + 0.5) / self.width - 1
        yy = 2 * (y + 0.5) / self.height - 1
        lng = pi * xx
        lat = 0.5 * pi * yy
    
        # calculate the spherical coordinates
        X = jnp.cos(lat) * jnp.cos(lng)
        Y = jnp.cos(lat) * jnp.sin(lng)
        Z = jnp.sin(lat)
        
        self.spher_coords = jnp.array([X, Y, Z])
        
    def rot_equi_img(self,src, dest, roll, pitch, yaw):
        return rotate(src,dest,self.spher_coords,roll,pitch,yaw,self.width,self.height)



def projection(frame,rhos,phis):
    return frame[rhos,phis,:]

projection_jit = jax.jit(projection)

@jit
def select_fov(image):
        return image[0:280,180:540,:]
    
@jit
def write_fov(image,insertion):
        return image.at[0:280,180:540,:].set(insertion)


@jit
def apply_mask(img, mask):
    return img * jnp.bitwise_and(mask[..., jnp.newaxis], 1)
        

@jit
def insert_image(large_img, small_img, position):
    y, x = position
    h, w = small_img.shape[:2]
    mask = jnp.zeros_like(large_img)
    mask = lax.dynamic_update_slice(mask, small_img, (y, x, 0))
    return mask#jnp.where(mask == 0, large_img, mask)


class Projector():
    def __init__(self, res_x=1280, res_y=720, proj_x=1240, proj_y=620):
        
        self.stim_x= 360
        self.stim_y= 180
        self.fov_azi = (0,180)
        self.fov_ele = (15,140)
        self.resolution = (res_x, res_y)
        self.projected_area = (proj_x, proj_y)
        self.blank_screen = np.zeros([self.resolution[1],self.resolution[0],3],dtype = "uint8")
        self.mask_screen = jnp.zeros([self.resolution[1],self.resolution[0],3],dtype = "uint8")
        self.border = int((self.resolution[0]-self.projected_area[0])/2)
        
        #WINDOW_NAME = 'Full Integration'
        #cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        #cv2.moveWindow(WINDOW_NAME, pos_x, pos_y)
        #cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    def initialize_projection_matrix(self,stim_size, fov_azi, fov_ele):
        
        self.stim_x= stim_size[1]
        self.stim_y= stim_size[0]
        self.fov_azi = fov_azi
        self.fov_ele = fov_ele
        xdim = self.projected_area[0]
        ydim = self.projected_area[1]
        xcenter = int(self.projected_area[0]/2)
        positiony = 0
        
        #reverse transform
        x_ones = np.ones(xdim)
        y_ones = np.ones(ydim).T
        x_vec = np.linspace(1,xdim,xdim)
        y_vec = (np.linspace(1,ydim,ydim)).T

        ymat = np.outer(y_vec,x_ones)
        xmat = np.outer(y_ones,x_vec)

        ymat = np.outer(y_vec,x_ones)
        xmat = np.outer(y_ones,x_vec)

        #up down angle rho
        rhos =   (np.around((np.sqrt((xmat - xcenter)**2 + (ymat - positiony)**2))/xcenter*self.stim_y)).astype(int)
        #left right phi
        phis = (np.around((np.arctan2((ymat - positiony), (xmat - xcenter)))/np.pi*self.stim_x)).astype(int)

        mask = np.zeros([ydim,xdim])
        inner = self.stim_y/self.fov_ele[1]*self.fov_ele[0]
        mask[np.where((rhos <=self.stim_y) &(rhos >=inner))] = 255
        self.mask= np.asarray( mask, dtype="uint8" )
        #phis[np.where(rhos >=140)] = 0
        #rhos[np.where(rhos >=140)] = 0
        self.phis = np.clip(phis,0,self.stim_x)
        self.rhos = np.clip(rhos,0,self.stim_y)
        
        
    
    def project_image(self, image):
        return projection_jit(image,self.rhos,self.phis)  
    
    
    #def mask_image_cv(self, image):
    #    projektor = np.asarray(image,dtype = "uint8")
    #   projektor = cv2.bitwise_and(projektor,projektor,mask = self.mask)
        
    #    self.mask_screen[:self.projected_area[1],self.border:-self.border,:]=projektor
        
    #    return self.mask_screen
    
    def mask_image(self, image):
        
        projektor = apply_mask(image,self.mask)
        self.mask_screen = insert_image(self.mask_screen, projektor, (0,self.border))
        return  np.asarray(self.mask_screen,dtype = "uint8")