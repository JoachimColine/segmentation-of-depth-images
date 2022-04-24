
--- INFO8010: Semantic Segmentation of Depth Images for Robotic Grasping ---
                              Joachim Coline

Some information on how the code is organized:

- DIRECTORIES:

./dataset_100_000: contains the dataset samples. 
                   Only 50 samples are provided because the files are too heavy. 
                   Full dataset is available upon request (or can be generated 
                   with Blender using datagen.py).
./real_data: contains data collected from sensor
./results: contains assessment results + weights of the final model (FCN-8s with pretrained VGG16).
           The weights for the other trained models are available upon request.

- FILES:

assessment.py: script for assessing a trained model on test data. It prints the scores.
CustomDatasets.py: dataset class specific to Blensor scans contained in ./dataset_100_000.
CustomModels.py: class for defining the neural net. 
                 If using vgg11 or vgg19, it must be manually changed in the code.
datagen.py: script to be run in Blender Python API. Blensor package must be installed. 
            See https://www.blensor.org/pages/downloads.html
            For using the script, 3D object models must be stored in ./objects.
forward_time.py: script for calculating mean time required for a forward pass.
main.py: script for running the training algorithm.
real_images.py: script for visualizing predictions on real data.
visualize.py: script for visualizing a sample from the synthetic dataset.