import cv2
import os
import numpy as np


def gaussianSmooth(kernel_size,sigma,image):
    # Apply Gaussian smooth
    smoothed = cv2.GaussianBlur(image, kernel_size, sigma)
    return smoothed

def doGaussianTests(images, output_dir):
    # Assign test_parameter values with (kernel size, sigma) tuples
    test_parameters = [
        (3, 3, 0.5),(3, 3, 1.0),(3, 3, 1.5),(3, 3, 2.0),(3, 3, 2.5),
        (5, 5, 0.5),(5, 5, 1.0),(5, 5, 1.5),(5, 5, 2.0),(5, 5, 2.5),
        (7, 7, 0.5),(7, 7, 1.0),(7, 7, 1.5),(7, 7, 2.0),(7, 7, 2.5),
        (9, 9, 0.5),(9, 9, 1.0),(9, 9, 1.5),(9, 9, 2.0),(9, 9, 2.5)
    ]
    imagesToDog = []
    # Do Gaussina smoothing on images and store them in directory
    for i, image in enumerate(images):
        for kernel_width, kernel_height, sigma in test_parameters:
            kernel_size = (kernel_width, kernel_height)
            smoothed_image = gaussianSmooth(kernel_size, sigma, image)
            filename = f"gaussianImage{i+1}_k{kernel_size[0]}x{kernel_size[1]}_s{sigma}.jpg"
            full_path = os.path.join(output_dir, filename)
            cv2.imwrite(full_path, smoothed_image)
            if (kernel_width, kernel_height, sigma) == (7, 7, 2.0):
                imagesToDog.append(smoothed_image)
    return imagesToDog

def medianSmooth(kernel_size, image):
    # Apply median smooth
    smoothed = cv2.medianBlur(image, kernel_size)
    return smoothed

def doMedianTests(images, output_dir):
    # Assign kernel sizes to test
    test_kernel_sizes = [3, 5, 7, 9]
    # Do median smoothing on images and store them in directory
    for i, image in enumerate(images):
        for kernel_size in test_kernel_sizes:
            smoothed_image = medianSmooth(kernel_size, image)
            filename = f"medianImage{i+1}_k{kernel_size}.jpg"
            full_path = os.path.join(output_dir, filename)
            cv2.imwrite(full_path, smoothed_image)

def difference_of_gaussian(image, k=1.6, epsilon=0.1):
    # Apply Gaussian blur with the original sigma
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(gray_image, (0, 0), 2.0)
    # Apply Gaussian blur with the multiplied sigma
    g2 = cv2.GaussianBlur(gray_image, (0, 0), k*2.0)
    # Compute the difference of the Gaussians
    dog = g1 - g2
    # Find the minimum and maximum values of the DoG image
    dog_min = np.min(dog)
    dog_max = np.max(dog)
    # Normalize to range of [0,1]
    dog_norm = (dog - dog_min) / (dog_max - dog_min)
    # Threshold the image
    # Since we've normalized to [0, 1], we don't need to multiply by 255 in the threshold
    edge_image = np.where(dog_norm >= epsilon, 1, 0)
    edge_image_uint8 = (edge_image * 255).astype(np.uint8)
    return edge_image_uint8

def dogTests(images,output_dir):
    #Assign kernel size and epsilon values
    k_and_epsilons = [
        (1.2,0.1),(1.6,0.1),(2.0,0.1),(2.4,0.1),(2.8,0.1),
        (1.2,0.3),(1.6,0.3),(2.0,0.3),(2.4,0.3),(2.8,0.3),
        (1.2,0.5),(1.6,0.5),(2.0,0.5),(2.4,0.5),(2.8,0.5),
        (1.2,0.7),(1.6,0.7),(2.0,0.7),(2.4,0.7),(2.8,0.7),
    ]
    imagesToCombine= []
    # Find dog and save images and store specific images to test in combine
    for i, image in enumerate(images):
        for k, epsilon in k_and_epsilons:
            dog = difference_of_gaussian(image, k, epsilon)
            filename = f"dogImage{i+1}_k{k}_eps{epsilon}.jpg"
            full_path = os.path.join(output_dir, filename)
            cv2.imwrite(full_path, dog)
            if ((k, epsilon )== (1.2,0.1)) or ((k, epsilon) == (1.2,0.7)) or ((k, epsilon )== (1.6,0.1)) or ((k, epsilon) == (1.6,0.7)) or ((k, epsilon) == (2.8,0.7)):
                imagesToCombine.append(dog)
    return imagesToCombine

def quantizeValueChannel(image, levels):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Split the channels
    h, s, v = cv2.split(hsv)
    # Calculate the height and width of the image
    height, width = v.shape
    # Create an empty array for the quantized image
    quantized_image = np.zeros((height, width, 3), dtype=np.uint8)
    quantized_image[:, :, 0] = h
    quantized_image[:, :, 1] = s
    # Quantize the V channel
    max_val = 255
    quantization_interval = max_val // levels
    # Loop through each pixel in the V channel and quantize it
    for i in range(height):
        for j in range(width):
            v_quantized = (v[i, j] // quantization_interval) * quantization_interval
            quantized_image[i, j, 2] = v_quantized
    # Convert the quantized HSV image back to BGR
    bgr_quantized = cv2.cvtColor(quantized_image, cv2.COLOR_HSV2BGR)
    return bgr_quantized

def quantizeTests(images, output_dir):
    # Assign levels to test
    levels = [3,4,6,9,14]
    imagesToCombine= []
    # Quantize and save images and store specific images to test in combine
    for i, image in enumerate(images):
        for level in levels:
            quantized = quantizeValueChannel(image, level)
            filename = f"quantImage{i+1}_lev{level}.jpg"
            full_path = os.path.join(output_dir, filename)
            cv2.imwrite(full_path, quantized)
            imagesToCombine.append(quantized)
    return imagesToCombine

def combinedTests(fromDog, fromQuantize, output_dir):
    #Combine images from Dog and quantized
    for i, edgeIm in enumerate(fromDog):
        for j,quantIm in enumerate(fromQuantize):
            if i == j:
                combined = combineEdgeQuant(edgeIm, quantIm)
                filename = f"combinedImage{i+1}.jpg"
                full_path = os.path.join(output_dir, filename)
                cv2.imwrite(full_path, combined)

def combineEdgeQuant(edge_image, quantized_image):
    inverted_edges = invert_image(edge_image)
    normalized_inverted_edges = inverted_edges / 255.0
    # Create an empty image to store the result
    combined_image = np.zeros_like(quantized_image)
    # Loop over each pixel in the quantized image and the inverted edge image
    for i in range(edge_image.shape[0]):
        for j in range(edge_image.shape[1]):
            combined_image[i, j] = quantized_image[i, j] * normalized_inverted_edges[i, j]
    return combined_image        

def invert_image(edge_image):
    # Create a new image array filled with zeros (black)
    inverted_image = np.zeros_like(edge_image)
    # Loop over each pixel in the edge image
    for i in range(edge_image.shape[0]):
        for j in range(edge_image.shape[1]):
            # Invert the pixel value
            # If the pixel is black (0), make it white (255) opposite if otherwise
            inverted_image[i, j] = 255 - edge_image[i, j]
    return inverted_image

images = [cv2.imread(f"report/data/image{i}.jpg") for i in range(1, 9)]
output_directory = 'report/result'
imagesToDog = doGaussianTests(images, output_directory)
doMedianTests(images, output_directory)
fromDog = dogTests(imagesToDog , output_directory)
fromQuantize = quantizeTests(imagesToDog,output_directory )
combinedTests(fromDog, fromQuantize,output_directory)