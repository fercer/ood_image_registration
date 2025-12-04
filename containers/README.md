# Container definition files

This folder contains the definition files required to build the containers required by the Open OnDemand **Image Registration** application.

## bftools.def

This container is required to use the `showinf` to get information about the reference and input images, such as the number of color channels, and number of slices in multi-focus images.

## valis-wsi_1.2.0_patched.def

This container has the [VALIS](https://valis.readthedocs.io/en/latest/) library installed, which is used for registration of the input image to the reference image.
The code of VALIS version `1.2.0` has some issues that have been identified but not yet corrected.
For this reason, the container applies a couple of patches on the code to be fully functional.

Additionally, a [Python script](https://gist.github.com/fercer/21913333808a2dad2e5e1644714a4a1e) is downloaded during the container building process to provide a Command Line Interface to use VALIS registration functionality as a background process with minimal user interaction.