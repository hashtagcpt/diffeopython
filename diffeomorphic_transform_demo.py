import numpy as np
import os
import random
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

from tkinter import filedialog

def diffeomorphic():
    maxdistortion = 80  # changes max amount of distortion
    nsteps = 20  # number of steps
    imsz = 1024
      # size of output images (bigger or equal to 2 x input image)

    picpath = filedialog.askdirectory(title="Choose a directory with base images")
    print(picpath)
    outpicpath = filedialog.askdirectory(title="Choose a directory to save warped images")
    print(outpicpath)

    # create output directory if it does not exist
    if not os.path.exists(outpicpath):
        os.makedirs(outpicpath)

    imgtype = 'jpg'  # file type
    saveimgtype = 'jpeg'  # file type to save as bc jpeg is the correct string for saving as jpg

    bg_fill = 128 # 128 for grey background; 255 for white; 0 for black
    resize_picture = True  # True if you want to resize the picture
    re_sz = 512  # size of resized picture only used if the above is true

    # Get list of files in directory
    fns = [fn for fn in os.listdir(picpath) if fn.endswith('.' + imgtype)]

    # Create meshgrid for interpolation
    YI, XI = np.meshgrid(np.arange(1, imsz + 1), np.arange(1, imsz + 1))

    phase_offset_max = 90  # Maximum phase offset
     # Randomize the phase offset        
    phaseoffset = random.randint(0, phase_offset_max)
    
    for i, fn in enumerate(fns):

        # Load the image
        Im = np.uint8(np.ones((imsz, imsz, 3)) * bg_fill)  
        P = np.array(Image.open(os.path.join(picpath, fn)))

        if resize_picture:
            P = cv2.resize(P, (re_sz, re_sz))

        # Remove alpha channel if necessary
        P = P[:, :, :3]
        Psz = P.shape

        # Upsample by factor of 2 in two dimensions
        P2 = np.zeros((2 * Psz[0], 2 * Psz[1], Psz[2]), dtype=P.dtype)
        P2[::2, ::2, :] = P
        P2[1::2, ::2, :] = P
        P2[1::2, 1::2, :] = P
        P2[::2, 1::2, :] = P
        P = P2
        Psz = P.shape

        # Pad image if necessary
        x1 = round((imsz - Psz[0]) / 2)
        y1 = round((imsz - Psz[1]) / 2)

        # Add fourth plane if necessary
        if Psz[2] == 4:
            Im = np.dstack((Im, np.zeros((imsz, imsz))))

        Im[x1:x1 + Psz[0], y1:y1 + Psz[1], :] = P

        cxA, cyA = getdiffeo(imsz, maxdistortion, nsteps)
        cxB, cyB = getdiffeo(imsz, maxdistortion, nsteps)
        cxF, cyF = getdiffeo(imsz, maxdistortion, nsteps)

        interpIm = Im.copy()

        for quadrant in range(1, 5):
            if quadrant == 1:
                cx, cy = cxA, cyA
                ind, indstep = 1, 1
            elif quadrant == 2:
                cx, cy = cxF - cxA, cyF - cyA
            elif quadrant == 3:
                ind, indstep = 4 * nsteps, -1
                interpIm = Im.copy()
                cx, cy = cxB, cyB
            elif quadrant == 4:
                cx, cy = cxF - cxB, cyF - cyB

            cy = YI + cy
            cx = XI + cx
            mask = (cx < 1) | (cx > imsz) | (cy < 1) | (cy > imsz)
            cx[mask] = 1
            cy[mask] = 1

            # Interpolate the image
            w = 0.1
            for _ in range(nsteps):  # This is the number of steps - Total number of warps is nsteps * quadrant
                centrex = 0.5 + (0.5 - w / 2) * np.cos((phaseoffset + ind) * 2 * np.pi / (4 * nsteps))
                centrey = 0.5 + (0.5 - w / 2) * np.sin((phaseoffset + ind) * 2 * np.pi / (4 * nsteps))

                randfn = os.path.join(outpicpath, f'Im_{i:02d}_{ind:02d}.{imgtype}')
                Image.fromarray(np.uint8(interpIm)).save(randfn, saveimgtype)

                # Define the regular grid (assuming imsz x imsz grid)
                x = np.arange(imsz)
                y = np.arange(imsz)
                grid = np.meshgrid(x, y, indexing='ij')

                # Define the interpolating function for each channel
                interp_func_r = RegularGridInterpolator((x, y), interpIm[:, :, 0], bounds_error=False, fill_value=bg_fill)
                interp_func_g = RegularGridInterpolator((x, y), interpIm[:, :, 1], bounds_error=False, fill_value=bg_fill)
                interp_func_b = RegularGridInterpolator((x, y), interpIm[:, :, 2], bounds_error=False, fill_value=bg_fill)

                # Create a list of (x, y) coordinates for which we want interpolations
                coords = np.column_stack((cx.flatten(), cy.flatten()))

                # Perform the interpolation and reshape back to the image size
                interpIm[:, :, 0] = interp_func_r(coords).reshape(imsz, imsz)
                interpIm[:, :, 1] = interp_func_g(coords).reshape(imsz, imsz)
                interpIm[:, :, 2] = interp_func_b(coords).reshape(imsz, imsz)

                # Update index
                ind += indstep


def getdiffeo(imsz, maxdistortion, nsteps):
    ncomp = 6

    # Precompute the DCT-II bases
    t = np.arange(imsz)
    dct_base = np.cos(np.pi * t[:, None] * np.arange(ncomp) / imsz)

    # Initialize arrays for the transform
    Xn = np.zeros((imsz, imsz))
    Yn = np.zeros((imsz, imsz))

    # Coefficients and phase for the transforms
    a = np.random.rand(ncomp, ncomp) * 2 * np.pi
    ph = np.random.rand(ncomp, ncomp, 4) * 2 * np.pi

    for xc in range(ncomp):
        for yc in range(ncomp):
            # Directly use DCT bases for transformation
            Xn += a[xc, yc] * np.outer(dct_base[:, xc], dct_base[:, yc] * np.cos(ph[xc, yc, 0])) 
            Yn += a[xc, yc] * np.outer(dct_base[:, xc], dct_base[:, yc] * np.cos(ph[xc, yc, 1])) 

    # Normalizing the transformations
    Xn = Xn / np.sqrt(np.mean(Xn ** 2))
    Yn = Yn / np.sqrt(np.mean(Yn ** 2))

    YIn = maxdistortion * Yn / nsteps
    XIn = maxdistortion * Xn / nsteps

    return XIn, YIn


# Run the function
diffeomorphic()
