# Federated Learning For Wireless Network

Client Scheduling in Wireless Federated Learning Based on Channel and Learning Qualities

*By Peng Wang*

## Project structure and outline
#### Wireless Channel Simulation Module
*Simulate the wireless channel state changes between multiple users and the base station when they move within a certain range。*
- /channels - Wireless Environment Simulation
  - /Random PathLoss
  - /Spatial Channel Model
  - /Winner2 and Random Waypoint Model

#### Federated learning training module
*Simulate the federated learning performance of multiple users under the condition of limited channel resources。*
- /data - Dataset storage directory
- /dictionaries - Scene preset file directory
- /utilities - Simulation configuration directory
- /models - Training function directory
- main_fed.py - Main function


## User selection method
- rand - Random selection
- chan - Channel quality only
- imp  - Gradient entropy \* loss
- nor  - Gradient norm
- ent  - Gradient entropy
- bal  - Combined Quality
- grad - Gradient divergence
- cadf - Category difference
- cdls - Category diff \* loss
- los  - Loss

