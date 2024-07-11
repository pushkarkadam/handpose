Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.

.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html

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