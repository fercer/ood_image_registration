# Image Registration app

This was created from the https://github.com/psobolewskiPhD/bc_example_qupath template.

## Prerequisites

This Batch Connect app requires the following software be installed on the
**compute nodes** that the batch job is intended to run on (**NOT** the
OnDemand node):

- [OpenSSL](https://www.openssl.org/) 1.0.1+ (used to hash the password, but this is not required)
- apptainer/singularity (as a module)

## Install

Clone this repository to your sandbox environment (`~/ondemand/dev`).

Build the Apptainer [valis contianer](containers/valis-wsi_1.2.0_patched.def) and [bftools container  ](containers/bftools.def) and move them into a location accessible to your user (for development) and update [the container path](https://github.com/fercer/ood_image_registration/blob/96c04577d7e8acc2644127d045999c4fdf7267cb/template/before.sh.erb). For making public, move the container to a globally accessible location (`/cm/shared`) and update the path accordingly.

## Make your changes

You can edit the files directly in your sandbox environment or have a local clone for development and push/pull to sync changes to the sandbox. Note that the dev OpenOnDemand will use whatever branch you have checked out in the sandbox repository, so you can easily test different changes by working in branches.
