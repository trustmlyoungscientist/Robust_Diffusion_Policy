import numpy as np
import cv2
import zarr
import os
import cv2
import click
from typing import Callable, Optional

def add_hardware_execution_error(actions, max_offset=10):
    """
    Simulate hardware execution errors and add random offsets to action information.
    : paramactions: The input action information is in the shape of (num_stamples, 2)
    : parameter max_offset: maximum offset, default is 0.1
    :return: Action information after adding hardware execution errors
    """
    # generate offset randomly
    offsets = np.random.uniform(-max_offset, max_offset, size=actions.shape[-1])
    noisy_actions = actions + offsets
    return noisy_actions

def add_camera_calibration_error(images, translation_range=(-10, 10), distortion_coefficients=None):
    """
    Simulate camera calibration errors, including image translation and distortion.
    : paramimages: The input image information has a shape of (num_stamples, height, width, 3)
    : paramtranslation_range: translation range, default is (-5,5)
    : paramdistortion-coefficients: Distortion coefficients, default to None, randomly generated if None
    : return: Image information after adding camera calibration error
    """
    num_samples, height, width, _ = images.shape
    
    # generate offset randomly
    translation_x = np.random.uniform(translation_range[0], translation_range[1])
    translation_y = np.random.uniform(translation_range[0], translation_range[1])
    
    # generate distortion coefficient
    if distortion_coefficients is None:
        k1_param = 0.2
        k2_param = 0.0
        p1_param = 0.1
        p2_param = 0.1
        k1 = np.random.uniform(0, k1_param)
        k2 = np.random.uniform(0, k2_param)
        p1 = np.random.uniform(-p1_param, p1_param)
        p2 = np.random.uniform(-p2_param, p2_param)
        distortion_coefficients = (k1, k2, p1, p2)
    
    # calculate the distorted coordinates
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = map_x + translation_x
    map_y = map_y + translation_y
    x = (map_x - width / 2) / (width / 2)
    y = (map_y - height / 2) / (height / 2)
    r = np.sqrt(x**2 + y**2)
    x_distorted = x * (1 + distortion_coefficients[0]*r**2 + distortion_coefficients[1]*r**4) + distortion_coefficients[2]*(r**2 + 2*x**2) + 2*distortion_coefficients[3]*x*y
    y_distorted = y * (1 + distortion_coefficients[0]*r**2 + distortion_coefficients[1]*r**4) + distortion_coefficients[3]*(r**2 + 2*y**2) + 2*distortion_coefficients[2]*x*y
    map_x_distorted = x_distorted * (width / 2) + width / 2
    map_y_distorted = y_distorted * (height / 2) + height / 2
    
    # remapping with bilinear interpolation 
    noisy_images = np.zeros_like(images)
    for i in range(num_samples):
        noisy_images[i] = cv2.remap(images[i], map_x_distorted.astype(np.float32), map_y_distorted.astype(np.float32), cv2.INTER_LINEAR)
    
    return noisy_images



def add_noise_to_sequences(zarr_file, output_file, eta=0.3):
    # load raw dataset
    data = zarr.open(zarr_file, mode='r')
    actions = data['data']['action'][:]
    images = data['data']['img'][:]
    states = data['data']['state'][:]
    episode_ends = data['meta']['episode_ends'][:]
    
    output = zarr.open(output_file, mode='w')
    output.create_group('data')
    output.create_group('meta')
    output['data'].create_dataset('action', data=actions)
    output['data'].create_dataset('img', data=images)
    output['data'].create_dataset('state', data=states)
    output['meta'].create_dataset('episode_ends', data=episode_ends)
    
    num_episodes = len(episode_ends)
    
    # randomly select sequences to add noise
    selected_indices = np.random.choice(num_episodes, size=int(eta * num_episodes), replace=False)
    
    for idx in selected_indices:
        if idx == 0:
            start = 0
        else:
            start = episode_ends[idx - 1]
        end = episode_ends[idx]
        
        subseq_start = start
        subseq_end = end
        
        # add noise
        sub_images = images[subseq_start:subseq_end]
        noisy_images = add_camera_calibration_error(sub_images)
        output['data']['img'][subseq_start:subseq_end] = noisy_images

@click.command()
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--noise_rate',  help='Noise rate', metavar='FLOAT',)

def main(
    source: str,
    dest: str,
    noise_rate: Optional[float],
):
    # dest = f"{os.path.dirname(source)}/pusht_cchi_v7_replay_camera_noise_eta_{str(int(eta*100))}.zarr"
    add_noise_to_sequences(source, dest, eta=noise_rate)