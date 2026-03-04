import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion

# Convert image to binary format
def binarize(img):
    return (img > 0).astype(np.uint8)

# Threshold method
def threshold(img, thresh):
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if img[y, x] > thresh - 50:
                img[y, x] = 255
            else:
                img[y, x] = 0

# Histogram method
def hist(img):
    h = np.zeros(256)
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            h[img[y, x]] += 1
    return h

# Find peak method
def find_peak(h):
    peak_intensity = np.argmax(h)
    peak_value = np.max(h)
    return peak_intensity, peak_value

# Dilation
def dilation(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    return binary_dilation(binarize(img), structure=kernel).astype(np.uint8) * 255

# Erosion
def erosion(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    return binary_erosion(binarize(img), structure=kernel).astype(np.uint8) * 255

# Connected Component Labeling using Flood Fill
def connected_components(img):
    labels = np.zeros_like(img, dtype=int)
    label = 1
    height, width = img.shape

    # Flood fill method
    def flood_fill(x, y, label):
        stack = [(x, y)]
        while stack:
            px, py = stack.pop()
            if labels[py, px] == 0 and img[py, px] == 255:
                labels[py, px] = label
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < width and 0 <= ny < height and labels[ny, nx] == 0:
                        stack.append((nx, ny))
    
    for y in range(height):
        for x in range(width):
            if img[y, x] == 255 and labels[y, x] == 0:
                flood_fill(x, y, label)
                label += 1
    
    return labels, label - 1

# Classify O-ring as pass or fail
def classify_orings(labels, num_regions, min_size=500):
    region_sizes = np.bincount(labels.flatten())[1:]
    hole_count = np.sum(region_sizes < min_size)
    
    return "FAIL" if hole_count > 2 else "PASS"

# Read in an image into memory 
i = 1
process = True
while process:
    index = i % 16
    before = time.time()
    if index != 0:
        print(index)
        img = cv.imread('C:/Users/yuiku/OneDrive/Documents/College Notes/Year 4/Semester 2/Computer Visions - Simon/Orings/Oring' + str(index) + '.jpg', 0)
        
        # Start timer
        start_time = time.time()

        h = hist(img)
        peak_intensity, peak_value = find_peak(h)
        print(f"Peak Intensity: {peak_intensity}, Peak Value: {peak_value}")

        # Plot histogram with peak marker
        plt.plot(h, label="Histogram")
        plt.axvline(x=peak_intensity, color='r', linestyle='--', label=f'Peak: {peak_intensity}')
        plt.legend()
        plt.show()

        # Threshold the image
        threshold(img, peak_intensity)
        cv.imshow('Thresholded Image', img)

        # Dilate the image
        dilated_img = dilation(img)
        cv.imshow('Dilated Image', dilated_img)
        
        # Erode the image
        eroded_img = erosion(img)
        cv.imshow('Eroded Image', eroded_img)
        
        # Connected Component Labeling
        labels, num_regions = connected_components(dilated_img)
        print(f"Number of regions detected: {num_regions}")
        
        # Classify O-rings
        result = classify_orings(labels, num_regions)
        print(f"O-ring classification: {result}")
        

        # Convert grayscale to color for annotation
        output_img = cv.cvtColor(dilated_img, cv.COLOR_GRAY2BGR)

        # Add classification text to the image
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(output_img, f' {result}', (10, 30), font, 1, (0, 0, 255), 2, cv.LINE_AA)

        # Add time taken to the image
        after = time.time()
        time_taken = after - start_time
        cv.putText(output_img, f'Time: {time_taken:.2f}s', (10, 70), font, 1, (0, 255, 0), 2, cv.LINE_AA)

        # To quit the loop
        k = cv.waitKey(0)
        if k == ord('q'):
            process = False
    i += 1