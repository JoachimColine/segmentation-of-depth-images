import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

	width = 224
	height = 171
	id = 11 # choose sample id to visualize
	scan = np.load("./dataset_100_000/scan_"+str(id)+".npy")
	depth_data = scan[0,:]
	segmentation = scan[1,:]
	plt.imshow(np.transpose(depth_data.reshape((width,height))))
	plt.show()
	plt.imshow(np.transpose(segmentation.reshape((width,height))))
	plt.show()
