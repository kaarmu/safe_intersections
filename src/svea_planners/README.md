# svea_planners

This repository contains various path planners that can be immediately used with
[svea](https://github.com/kth-sml/svea).

## Installation

Please refer to [svea](https://github.com/KTH-SML/svea) on how to install and
run SVEA software.

## Usage
Currently only the A* planner is developed and fully functional. To 'rosify' its behaviour, please refer to `planner_interface.py` in the [main svea repo](https://github.com/KTH-SML/svea). A graphical representation of the workflow is given in the following:
![uml_svea_planners](https://github.com/KTH-SML/svea_planners/assets/33842599/dea00719-b503-4e57-992d-899e55b61229)

## Roadmap

- [ ] Implement a few (2-3) different basic path planners
  - RRT
  - Reeds-Shepp
  - ?
- [ ] Helper functions, e.g. different kind of path smoothing, frenet frame

