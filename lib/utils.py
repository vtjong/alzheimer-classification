import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def visualize(PATH, View = "Axial_View",cmap = None):
    '''
    Visualize Image

    Parameters
    ----------
    PATH --- Path to *nii.gz image or image as numpy array
    View --- Sagittal_View or Axial_View or Coronal_View
    cmap

    '''
    plt.ion()
    view = View.lower()
    v = {
        "sagittal_view":0,
        "axial_view":2,
        "coronal_view":1,
    }
    if view not in ["sagittal_view","axial_view","coronal_view"]:
        print("Enter Valid View")
        return
    if(type(PATH) == np.str_ or type(PATH) == str):
        img = nib.load(PATH)
        img1  = img.get_fdata()
    elif(type(PATH) == np.ndarray and len(PATH.shape) == 3):
        img1 = PATH
    else:
        print("ERROR: Input valid type 'PATH'")
        return
    for i in range(img1.shape[v[view]]):
        plt.cla()
        if v[view] == 0:
            plt.imshow(img1[i,:,:], cmap = cmap)
        elif v[view] == 1:
            plt.imshow(img1[:,i,:], cmap = cmap)
        else:
            plt.imshow(img1[:,:,i], cmap = cmap)
        plt.show()
        plt.pause(0.1)
    print(f"Shape of image:{img1.shape}")
