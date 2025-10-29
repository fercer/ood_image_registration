# Batch Connect - Example using containerized Xpra app (QuPath)

This is a template repo and example adapted from:
https://github.com/OSC/bc_example_jupyter

The goal is to make it easy to access containerized applications that use [Xpra](https://github.com/Xpra-org/xpra/) (specifically [the web client](https://github.com/Xpra-org/xpra-html5)) for GUI access. For an example apptainer/singularity definition, see: https://github.com/TheJacksonLaboratory/rit-imageanalysis-containers/blob/main/qupath_xpra.def
For a docker definition, see: https://github.com/napari/napari/blob/main/dockerfile

The idea is that in this template repository, in [template/before.sh.erb](https://github.com/psobolewskiPhD/bc_example_qupath/blob/main/template/before.sh.erb) the port, password, and DISPLAY are already set up for a minimal Xpra session. 
So to implement an app, only [the container path](https://github.com/psobolewskiPhD/bc_example_qupath/blob/be88adf8636531c6078c783e7b55d85512c82830/template/before.sh.erb#L54) needs to be modified, as well as the [actual start executable](https://github.com/psobolewskiPhD/bc_example_qupath/blob/be88adf8636531c6078c783e7b55d85512c82830/template/script.sh.erb#L36). Beyond that, you can update the icon.png to for the app, as well as [view.html.erb](https://github.com/psobolewskiPhD/bc_example_qupath/blob/main/view.html.erb) which defines the connect button.
In terms of the SLURM parameters and the submission form, a minimal set is already provided in [form.yml](https://github.com/psobolewskiPhD/bc_example_qupath/blob/main/form.yml), but you can edit that along side [submit.yml.erb](https://github.com/psobolewskiPhD/bc_example_qupath/blob/main/submit.yml.erb).

## Prerequisites

This Batch Connect app requires the following software be installed on the
**compute nodes** that the batch job is intended to run on (**NOT** the
OnDemand node):

- [OpenSSL](https://www.openssl.org/) 1.0.1+ (used to hash the password, but this is not required)
- apptainer/singularity (as a module)

## Install

Click the `Use this template` button and create a new repository from this template repository using the GitHub interface.

Clone your new repository to your sandbox environment (`~/ondemand/dev`).

Place your container with the application and xpra in a location accessible to your user (for development) and update [the container path](https://github.com/psobolewskiPhD/bc_example_qupath/blob/be88adf8636531c6078c783e7b55d85512c82830/template/before.sh.erb#L54). For making public, move the container to a globally accessible location (`/cm/shared`) and update the path accordingly.

## Make your changes

You can edit the files directly in your sandbox environment or have a local clone for development and push/pull to sync changes to the sandbox. Note that the dev OpenOnDemand will use whatever branch you have checked out in the sandbox repository, so you can easily test different changes by working in branches.
