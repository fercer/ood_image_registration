# Container definition file

This folder contains the definition file to build the container required by the Open OnDemand **Image Registration** application.

## valis-wsi_1.2.0_patched.def

This container has the [VALIS](https://valis.readthedocs.io/en/latest/) library installed, which is used for registration of the input image to the reference image.
The code of VALIS version `1.2.0` has some issues that have been identified but not yet corrected.
For this reason, the container applies a couple of patches on the code to be fully functional.

Additionally, a set of [Python scripts](../scripts) are downloaded during the container building process to provide a Command Line Interface to use VALIS registration functionality as a background process with minimal user interaction.