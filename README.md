# Multi-Agent SLAM

## Main Files
* nclt_distributed_time_varying.py: the main file for running distributed sparse-gaussian mapping on NCLT dataset, with time-varying graph 
* agent.py: classes that define agent of different types (e.g. distributed vs. centralized); 
* quad_tree.py: sparse-gp and quad-tree (2D) implementations;
* tsdf.py: helpers for transforming sensor observations into SPGP pseudo-points;
* message_manager.py: module for managing (in-memory) message passing between agents
* data_loader.py: contains classes that load sensor data from NCLT dataset

## Auxilary files
* lidar_2d_distributed_time_varying.py: file for running distributed sparse-gaussian mapping on the Uni-bonn 2D Lidar dataset, with time-varying graph
* plot_prediction.py: functions for plotting metrics

## Config Files
* nclt: contains config for running agents on the NCLT dataset. In particular, distributed_time_varying.yaml defines the config for running distributed-mapping with time-varying graph
* uni-bonn: contains config for running agents on the uni-bonn datasets.