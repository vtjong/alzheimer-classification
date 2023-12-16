import cv2 as cv
from PIL import Image
import nibabel as nib
import numpy as np
import pandas as pd
import sys
import os
# from deepbrain import Extractor

def get_data(path):
    return nib.load(path).get_data()

def get_data_with_skull_scraping(path, PROB = 0.5):
    arr = nib.load(path).get_data()
    ext = Extractor()
    prob = ext.run(arr)
    mask = prob > PROB
    arr = arr*mask
    return arr

def histeq(data):
    for slice_index in range(data.shape[2]):
        data[:,:,slice_index]=cv.equalizeHist(data[:,:,slice_index])
    return data

def to_uint8(data):
    data=data.astype(np.float)
    data[data<0]=0
    return ((data-data.min())*255.0/data.max()).astype(np.uint8)

def convert_gif_to_jpg(gif_image_path, converted_image_path):
    with Image.open(gif_image_path) as img:
        img.convert('RGB').save(converted_image_path, 'JPEG')

def crop_black_boundary(mri_image_path, output_path):
    image = cv.imread(mri_image_path, cv.IMREAD_GRAYSCALE)
    _, thresh = cv.threshold(image, 1, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    cv.imwrite(output_path, cropped_image)

# convert_gif_to_jpg('/Users/valenetjong/alzheimer-classification/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg_tra_90.gif', 'test.png')
crop_black_boundary('test.png', 'cropped_test.png')

""" Pre-processing Functions """

DEMENTIA_MAP = {
    '0.0': "nondemented",
    '0.5': "mildly demented",
    '1.0': 'moderately demented',
    '2.0': 'severely demented'
}

def extract_files(base_dir, target_dir, oasis_csv_path):
    oasis_df = pd.read_csv(oasis_csv_path)
    scan_types = ["cor_110", "sag_95", "tra_90"]

    for subdir in filter(lambda d: d != '.DS_Store', os.listdir(base_dir)):
        source_dir = os.path.join(base_dir, subdir, "PROCESSED", "MPRAGE", 
                                  "T88_111")
        num = int(subdir.split('_')[1])
        dementia_type = oasis_df.iloc[num]['CDR']
        if pd.isna(dementia_type):
            continue

        for scan_type in scan_types:
            for n_suffix in ['n3', 'n4']:
                fn = os.path.join(source_dir, f"{subdir}_mpr_{n_suffix}_anon_"
                                  f"111_t88_gfc_{scan_type}.gif")
                if os.path.exists(fn):
                    process_image(fn, target_dir, dementia_type, num, 
                                  scan_type)

def process_image(fn, target_dir, dementia_type, num, scan_type):
    with Image.open(fn) as img:
        img = skull_strip(img)
        target_subdir = os.path.join(target_dir, DEMENTIA_MAP[str(dementia_type)], str(num))
        os.makedirs(target_subdir, exist_ok=True)
        target_path = os.path.join(target_subdir, 
                                   f"{scan_type}.png")
        img.convert('RGB').save(target_path)

def skull_strip(img):
    # Convert to grayscale for skull stripping
    img_gray = img.convert('L')
    img_np = np.array(img_gray)

    # Basic skull stripping using Otsu's method
    threshold = skimage.filters.threshold_otsu(img_np)
    mask = img_np > threshold
    mask = skimage.morphology.remove_small_objects(mask, min_size=100)
    img_np[~mask] = 0

    # Intensity normalization
    img_normalized = img_np / 255.0

    # Convert back to PIL Image
    img_processed = Image.fromarray((img_normalized * 255).astype(np.uint8))
    return img_processed

# Replace 'path_to_disc1' with the actual path to your 'disc1' directory
path_to_disc1 = '/Users/valenetjong/aml_final/disc1'
oasis_csv_path = '/Users/valenetjong/aml_final/datacsv/oasis_cross-sectional.csv'
extracted_files = extract_files(path_to_disc1, '/Users/valenetjong/aml_final/data', oasis_csv_path)