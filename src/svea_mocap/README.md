# svea_mocap

This repository contains convenience launch files and python class for
integrating the Qualisys Motion Capture (mocap) System into the SVEA software
stack. The repository has two main examples to demonstrate:
1. Using mocap as localization for experiments
2. Using mocap to evaluate new localization methods (i.e. slam or via aruco markers)

## Installation

Please refer to [svea](https://github.com/KTH-SML/svea) on how to install and
run SVEA software.

Once you have set up your workspace based on [svea](https://github.com/KTH-SML/svea),
in addition to cloning this package, you also need to clone the
[motion\_capture\_system](https://github.com/KTH-SML/motion_capture_system) repository,
which is the driver for ROS-ifying data from the motion capture system.

## Usage

For setting up the motion capture system, contact one of the lab's
research engineers who works with the system.

Once you have gotten help to set up your environment on the motion
capture system, you can proceed to one of the following examples.

**Note:** For these examples to work, you need to be on the same network as
the motion capture system.

### Testing connection to mocap

To test if you can connect to the mocap system in the first place, try
running

```bash
roslaunch mocap_qualisys qualisys.launch
```

If no errors or warnings pop-up, you can then try calling `rostopic list`
and you should see all of the subjects that are published by the mocap
system.

### Example: mocap as localization

This is useful when running an experiment that requires perfect localization
can be run within the confines of the motion capture area.

To use mocap as localization, try running the following launch
file with the `<subject name>` replaced:

```bash
roslaunch svea_mocap mocap_only mocap_name:=<subject name>
```

### Example: evaluating new localization approaches

This is useful when new localization approaches need to be evaluated against
the ground truth of the motion capture system.

To compare indoor localization with mocap, try running the following
launch file with the `<subject name>` replaced:

```bash
roslaunch svea_mocap localization_comparison mocap_name:=<subject name>
```

In order to visualize data produced by the localization node running onboard the svea platform and compare them to the
pose returned by the mocap, launch the *localization_comparison_ui.launch*. This node will take measurements from both sources
and generates relevant plots about the RMSE related to x and y coordinates, as well as the yaw angle.
Furthermore, a live representation of the estimated and real velocity is provided, coupled with the current trajectory.

```bash
roslaunch svea_mocap localization_comparison_ui mocap_name:=<subject name> localization_topic:=<topic name> ground_truth_topic:=<mocap topic name>
```

For a detailed description of how the node works, see the documentation embedded in the script.

**Note:** By default the example supports the use of the RC remote for driving the
vehicle around while recording the inputs from the RC remote.
