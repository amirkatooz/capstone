import numpy as np
import pandas as pd
from scipy.misc import imrotate



def rotate_image(image, angle):
    
    rotated_image_str = ' '.join(
        [str(n) for n in imrotate(np.fromstring(image['Image'], sep=' ').reshape(96, 96), -angle).reshape(96*96,)]
    )
    
    angle_rad = np.radians(angle)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    
    keypoints = image.drop('Image').values - 48
    rotated_keypoints = np.zeros(30)
    
    num_of_keypoints = (len(image) - 1) / 2
    for i in range(num_of_keypoints):
        rotated_keypoints[2*i] = cos*keypoints[2*i] - sin*keypoints[2*i+1]
        rotated_keypoints[2*i+1] = sin*keypoints[2*i] + cos*keypoints[2*i+1]
    
    rotated_image_keypoints = image.drop('Image')
    rotated_image_keypoints.iloc[:] = rotated_keypoints + 48
    rotated_image_keypoints['Image'] = rotated_image_str
    
    return rotated_image_keypoints



def rotate_images(image_df, angle):
    
    rotated_image_series = image_df['Image'].map(
        lambda image: ' '.join(
            [str(n) for n in imrotate(np.fromstring(image, sep=' ').reshape(96, 96), -angle).reshape(96*96,)]
        )
    )
    
    angle_rad = np.radians(angle)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    
    keypoints_matrix = image_df.drop('Image', axis=1).values - 48
    rotated_keypoints = np.zeros_like(keypoints_matrix)

    num_of_keypoints = (image_df.shape[1] - 1) / 2
    for i in range(num_of_keypoints):
        rotated_keypoints[:, 2*i] = cos*keypoints_matrix[:, 2*i] - sin*keypoints_matrix[:, 2*i+1]
        rotated_keypoints[:, 2*i+1] = sin*keypoints_matrix[:, 2*i] + cos*keypoints_matrix[:, 2*i+1]
    
    rotated_images_df = pd.DataFrame(rotated_keypoints+48, columns=image_df.drop('Image', axis=1).columns)
    rotated_images_df['Image'] = rotated_image_series
        
    return rotated_images_df



def flip_images(image_df):
    
    flipped_image_series = image_df['Image'].map(
        lambda image: ' '.join(
            [str(n) for n in np.fliplr(np.fromstring(image, sep=' ').reshape(96, 96)).reshape(96*96,)]
        )
    )
    
    flipped_image_df = image_df.drop('Image', axis=1)
    
    num_of_keypoints = (image_df.shape[1] - 1) / 2
    for i in range(num_of_keypoints):
        flipped_image_df.iloc[:, 2*i] = 96 - flipped_image_df.iloc[:, 2*i]
        
    colname_list = [colname.replace('right', 'TEMP') for colname in flipped_image_df.columns.tolist()]
    colname_list = [colname.replace('left', 'right') for colname in colname_list]
    colname_list = [colname.replace('TEMP', 'left') for colname in colname_list]
    
    flipped_image_df.columns = colname_list
    flipped_image_df = flipped_image_df[image_df.drop('Image', axis=1).columns.tolist()]
    flipped_image_df['Image'] = flipped_image_series
    
    return flipped_image_df



def adjust_contrast(image_df, degree=.8):
    
    image_list = [np.fromstring(image, sep=' ') for image in image_df['Image']]
    adjusted_contrast_images = pd.Series([degree*image + (1-degree)*image.mean() for image in image_list])
    adjusted_image_str_list = adjusted_contrast_images.map(lambda image: ' '.join([str(round(n)) for n in image]))
    
    adjusted_contrast_df = image_df.drop('Image', axis=1)
    adjusted_contrast_df['Image'] = adjusted_image_str_list
    
    return adjusted_contrast_df