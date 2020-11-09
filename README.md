# Angle Estimation

Different RNNs were experimented with for analysing PMU data for estimating voltage angles. The most accurate architechture found was a CNN-GRU with attention.

`ML/` implements the PyTorch model, datasets, transforms and training code.

`Tester.ipynb` runs the tests.

`simulations/` implements the MATLAB Simulink simulation and script to generate the test data.
