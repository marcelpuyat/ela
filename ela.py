import numpy as np
import sys
import os
from PIL import Image, ImageEnhance, ImageChops

ELA_QUALITY = 95
NUM_HORIZ_REGIONS = 8
NUM_VERT_REGIONS = 8
FAKE_THRESHOLD_DIFF = 1.2 # Still experimenting with this value.

FAKE_FILES_TEST_DIR = "fake_test"
REAL_FILES_TEST_DIR = "real_test"
MISCLASSIFIED_FAKES_DIR = "misclassified_fake"
MISCLASSIFIED_REAL_DIR = "misclassified"
CORRECT_FAKES_DIR = "fake"
CORRECT_REAL_DIR = "not_fake"

# Returns True if real, False if fake
def classify_image(image_file_name, biggest_diff_filename=None):

	orig_file = image_file_name
	new_file = 'tmp.jpg'

	# Resave the image at 95% quality.
	orig_im = Image.open(orig_file)
	orig_im.save(new_file, quality=ELA_QUALITY)
	new_im = Image.open(new_file)

	# Get pixels into an np array so we can easily compute their abs diff.
	orig_pixels = np.array(list(orig_im.getdata()))
	new_pixels = np.array(list(new_im.getdata()))

	# Compute abs diff between orig and resaved image. We could also use ImageChops.difference,
	# here we just do it manually.
	# diff_im = ImageChops.difference(orig_im, new_im)
	diff_pixels = np.absolute(orig_pixels - new_pixels)
	diff_im = Image.new(orig_im.mode, orig_im.size)
	diff_pixels_list = list(map(tuple, diff_pixels))
	diff_im.putdata(diff_pixels_list)

	# Try to see if any region in the image has been modified. We do this by doing the following:
	# Divide pic into 25 blocks of size width/5,height/5 and compute mean of pixel diffs inside
	# each block. Then get the max mean diffs of all the blocks and compare it to our threshold.
	region_width = orig_im.size[0]/NUM_HORIZ_REGIONS
	region_height = orig_im.size[1]/NUM_VERT_REGIONS
	reshaped_diff = diff_pixels.reshape((orig_im.size[1], orig_im.size[0], 3))
	reshaped_real = orig_pixels.reshape((orig_im.size[1], orig_im.size[0], 3))
	biggest_mean_diff = 0
	biggest_mean_diff_im = None
	mean_diffs = np.zeros(NUM_VERT_REGIONS*NUM_HORIZ_REGIONS)
	for row in xrange(NUM_VERT_REGIONS):
		for col in xrange(NUM_HORIZ_REGIONS):
			diff_cell = reshaped_diff[row*region_height:row*region_height+region_height, col*region_width:col*region_width+region_width]
			mean_diff = np.mean(diff_cell)

			real_cell = reshaped_real[row*region_height:row*region_height+region_height, col*region_width:col*region_width+region_width]
			real_cell = real_cell.reshape((region_width * region_height, 3))
			real_cell_pixels = list(map(tuple, real_cell))
			cell_im = Image.new(orig_im.mode, (region_width, region_height))
			cell_im.putdata(real_cell_pixels)

			# Remember region that had the biggest diff.
			if mean_diff > biggest_mean_diff:
				biggest_mean_diff = mean_diff
				biggest_mean_diff_im = cell_im
			mean_diffs[col + row*NUM_HORIZ_REGIONS] = mean_diff
			# cell_im.show(title=str((row,col)))

	if biggest_diff_filename is not None:
		biggest_mean_diff_im.save(biggest_diff_filename)

	# The rest of the code here is to just display the ELA diff image.

	# Compute min and max pixel values. This is in the format:
	# ((min_r_diff, max_r_diff), (min_g_diff, max_g_diff), (min_b_diff, max_b_diff))

	# extrema = diff_im.getextrema()

	# # Compute max among all band maxes.
	# max_diff = max([ex[1] for ex in extrema])

	# # Make things easier to visualize by scaling by max. TODO(marcelpuyat) read up more on how this works...
	# scale = 255.0/max_diff
	# diff_im = ImageEnhance.Brightness(diff_im).enhance(scale)

	# diff_im.show()

	# Diff between biggest diff region and 10th percentile must be greater than this
	# threshold value for us to consider it fake.
	diff = biggest_mean_diff - np.percentile(mean_diffs, 10)
	return diff < FAKE_THRESHOLD_DIFF, diff

def main():
	# Go over fakes
	for f in os.listdir(FAKE_FILES_TEST_DIR):
		is_real, ela_diff = classify_image(FAKE_FILES_TEST_DIR + "/" + f)

		print("Classifying fake image (" + f + ") as " + ("real" if is_real else "fake") + " ("+ str(ela_diff) +")")
		if is_real:
			os.rename(FAKE_FILES_TEST_DIR + "/" + f, MISCLASSIFIED_FAKES_DIR + "/" + f)
		else:
			os.rename(FAKE_FILES_TEST_DIR + "/" + f, CORRECT_FAKES_DIR + "/" + f)

	# Go over reals
	for f in os.listdir(REAL_FILES_TEST_DIR):
		is_real, ela_diff = classify_image(REAL_FILES_TEST_DIR + "/" + f, "biggest_diff.jpg")
		print("Classifying real image (" + f + ") as " + ("real" if is_real else "fake") + " ("+ str(ela_diff) +")")

		if is_real:
			os.rename(REAL_FILES_TEST_DIR + "/" + f, CORRECT_REAL_DIR + "/" + f)
		else:
			os.rename(REAL_FILES_TEST_DIR + "/" + f, MISCLASSIFIED_REAL_DIR + "/" + f)

main()