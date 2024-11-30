Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.

.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html

[Unreleased]
------------

Added
^^^^^
- Reduce dataset size function added.

Fixed
^^^^^
- Backward compatibility with torch and torchvision.

[0.1.16] - 2024-11-14
---------------------

Added
^^^^^
- Verbose to indicate device used.

[0.1.15] - 2024-11-03
---------------------

Added
^^^^^
- Create train directory with sequential numbering.
- Improved verbose metrics display during training.
- Resume training function.
- Example with command line variables for training and resume training.

Changed
^^^^^^^
- Saving loss csv file for each epoch in training model.
- Training model with control flow of resume training.
- Plot loss graphs from dataframe function.

[0.1.14] - 2024-10-16
---------------------

Fixed
^^^^^
- Config file input in the train network function.

[0.1.13] - 2024-10-14
---------------------

Added
^^^^^
- MSE loss for classes.
- Euclidean distance for keypoints.
- Using ``figsize`` as parameter in ``display_images_in_grid()`` function.
- Updated keypoint loss function.
- Options to use the exponent of sum of all the mse for keypoints.
- Detect function added.
- Render detection function added.
- CSV files saving for losses added in training.
- Options to include the loss types in training.
- Saving sample rendered images for truth and prediction.
- Average Precision calculation function.
- Precision and Recall curve function.
- Mean average precision function also returns precision, recall, and F score for class.
- Function to setup the job and train network from YAML file added.
- Example code added to train the overall network with job setup from config file.

Fixed
^^^^^
- Mean Average Precision calculation fixed.

Removed
^^^^^^^
- Saving dictionary of results.
- Confusion matrix for each batch size.

[0.1.11] - 2024-08-04
---------------------

Fixed 
^^^^^
- Calculating loss over batch size in all the loss functions.


[0.1.10] - 2024-07-11
---------------------

Added
^^^^^
- Average box ratio of the image added.


[0.1.9] - 2024-06-14
--------------------

Added 
^^^^^
- Progress bar added for each batch.

Fixed
^^^^^
- Problems with tensors not assigned to device.

[0.1.0] - 2024-06-13
--------------------

Added
^^^^^
- Function to load variables from a yaml file.
- Read labels from text file function.
- Read files from directory function.
- Split label tensor function.
- Label to tensor conversion for ground truth.
- Cartesian to polar relative to center coordinates.
- Truth head function.
- ``HandDataset`` class for loading the data.
- Rendering function added to render hand bounding box, pose, labels.
- Displaying grid of images function added that plots images together in a grid.
- Network loading using transfer learning added.
- Network head added that uses the prediction tensor from network and stores them into annotation dictionary.
- Conversion from xywh to xyxy format added.
- Intersection over union function added to metrics.
- Best box head selection function based on maximum confidence added.
- Head extraction function that extracts the bounding box and keypoint coordinates added.
- Activation functions for prediction head added.
- Loss function added.
- Optimizer function added.
- Non-Max suppression (NMS) function added.
- Mean average precision calculation added.
- Loss function returns a dictionary of all losses.
- Added Scheduler class for flexible use of schedulers.
- Train function added to train the model.
- Example file for training added.