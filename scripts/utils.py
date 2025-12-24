import os
import json
import sys
import argparse

from itertools import product

from pathlib import Path

import numpy as np
import itk

from valis import registration, non_rigid_registrars
import valis.slide_io
import valis.preprocessing


class BioFormatsSlideReaderZ(valis.slide_io.BioFormatsSlideReader):
    def __init__(self, src_f, series=None, z_slice=0, *args, **kwargs):
        super(BioFormatsSlideReaderZ, self).__init__(src_f, series=series, *args, **kwargs)
        self._z_slice = z_slice

    def slide2vips(self, level, series=None, xywh=None, tile_wh=None, z=0, t=0, *args, **kwargs):
        return super(BioFormatsSlideReaderZ, self).slide2vips(level, series=series, xywh=xywh, tile_wh=tile_wh, z=self._z_slice, t=t, *args, **kwargs)


# Load the available preprocessors in VALIS from a JSON file.
parent_dir = Path(__file__).parent.resolve()
with open(parent_dir / "preprocessors.json", 'r') as f:
    PREPROCESSORS = json.load(f)

    for prep in PREPROCESSORS.keys():
        PREPROCESSORS[prep]["class"] = getattr(valis.preprocessing, prep)

# Load the available readers in VALIS from a JSON file.
with open(parent_dir / "readers.json", 'r') as f:
    IMAGE_READERS = json.load(f)

    for prep in IMAGE_READERS.keys():
        if prep == "BioFormatsSlideReaderZ":
            IMAGE_READERS[prep]["class"] = BioFormatsSlideReaderZ
        else:
            IMAGE_READERS[prep]["class"] = getattr(valis.slide_io, prep)


ARGUMENTS_TYPES = {
    "bool": bool,
    "str": str,
    "int": int,
    "float": float,
    "str_int": lambda arg: int(arg) if arg.isnumeric() else arg
}


class SimpleElastixWarper2(non_rigid_registrars.NonRigidRegistrar):
    """
    From https://github.com/MathOnco/valis/issues/205#issuecomment-3192079394
    This class allows to use SimpleElastix for non-rigid registration after the original package used for this (SimpleElastix) was deprecated and repalced by SimpleITK.

    Uses SimpleElastix to register images

    """
    def __init__(self, params=None, elastix_params={}):
        """
        Parameters
        ----------
        """
        super().__init__(params=params)
        self.elastix_params = elastix_params

    @staticmethod
    def get_default_params(img_shape, grid_spacing_ratio=0.025):
        """
        Get default parameters for registration with sitk.ElastixImageFilter

        See https://simpleelastix.readthedocs.io/Introduction.html
        for advice on parameter selection
        """
        parameter_object = itk.ParameterObject()
        p = parameter_object.GetDefaultParameterMap("bspline")
        p["Metric"] = ['AdvancedMattesMutualInformation', 'TransformBendingEnergyPenalty']
        p["MaximumNumberOfIterations"] = ['1500']  # Can try up to 2000
        p['FixedImagePyramid'] = ["FixedRecursiveImagePyramid"]
        p['MovingImagePyramid'] = ["MovingRecursiveImagePyramid"]
        p['Interpolator'] = ["BSplineInterpolator"]
        p["ImageSampler"] = ["RandomCoordinate"]
        p["MetricSamplingStrategy"] = ["None"]  # Use all points
        p["UseRandomSampleRegion"] = ["true"]
        p["ErodeMask"] = ["true"]
        p["NumberOfHistogramBins"] = ["32"]
        p["NumberOfSpatialSamples"] = ["3000"]
        p["NewSamplesEveryIteration"] = ["true"]
        p["SampleRegionSize"] = [str(min([img_shape[1] // 3, img_shape[0] // 3]))]
        p["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
        p["ASGDParameterEstimationMethod"] = ["DisplacementDistribution"]
        p["HowToCombineTransforms"] = ["Compose"]
        grid_spacing_x = img_shape[1]*grid_spacing_ratio
        grid_spacing_y = img_shape[0]*grid_spacing_ratio
        grid_spacing = str(int(np.mean([grid_spacing_x, grid_spacing_y])))
        p["FinalGridSpacingInPhysicalUnits"] = [grid_spacing]
        p["WriteResultImage"] = ["false"]

        return p

    def calc(self, moving_img, fixed_img, mask=None, *args, **kwargs):
        """Perform non-rigid registration using SimpleElastix.
        """

        assert moving_img.shape == fixed_img.shape,\
            print("Images have different shapes")

        itk_fixed_image = itk.GetImageFromArray(fixed_img)
        itk_moving_image = itk.GetImageFromArray(moving_img)

        params = self.get_default_params(img_shape=moving_img.shape[0:2])
        if self.elastix_params is not None:
            for k, v in self.elastix_params:
                params.update({k:v})

        parameter_object = itk.ParameterObject()
        parameter_object.AddParameterMap(params)

        result_image, elastix_params = itk.elastix_registration_method(itk_fixed_image,
                                                                    itk_moving_image,
                                                                    parameter_object=parameter_object,
                                                                    log_to_console=False)

        backward_deformation = itk.transformix_deformation_field(itk_moving_image, elastix_params)
        backward_deformation = itk.GetArrayFromImage(backward_deformation)

        self.params = None  # Can't pickle SimpleITK.ParameterMap
        self.elastix_params = None  # Can't pickle SimpleITK.ParameterMap

        dxdy = np.array([backward_deformation[..., 0], backward_deformation[..., 1]])

        return dxdy


def same_name_check(out_basename, out_extension):
    if not os.path.isfile(f"{out_basename}.{out_extension}"):
        return f"{out_basename}.{out_extension}"

    same_name_count = 1
    while os.path.isfile(f"{out_basename}_({same_name_count}).{out_extension}"):
        same_name_count += 1

    return f"{out_basename}_({same_name_count}).{out_extension}"


def basic_arguments():
    base_args_parser = argparse.ArgumentParser("VALIS registration", add_help=False)

    group_parser = base_args_parser.add_argument_group("Input Options")

    group_parser.add_argument("-r", "--reference-image", dest="reference_image", type=str, help="Reference image to register towards into.", required=True)
    group_parser.add_argument("-i", "--input-image", dest="input_image", type=str, help="Image to register towards the reference image.", required=True)

    group_parser.add_argument("-rr", "--reference-image-reader", dest="reference_image_reader", type=str, help="Reference image reader class.", choices=list(IMAGE_READERS.keys()), required=True)
    group_parser.add_argument("-rp", "--reference-image-preprocessor", dest="reference_image_preprocessor", type=str, help="Reference image preprocessor class.", choices=list(PREPROCESSORS.keys()), required=True)

    group_parser.add_argument("-ir", "--input-image-reader", dest="input_image_reader", type=str, help="Input image reader class.", choices=list(IMAGE_READERS.keys()), required=True)
    group_parser.add_argument("-ip", "--input-image-preprocessor", dest="input_image_preprocessor", type=str, help="Input image preprocessor class.", choices=list(PREPROCESSORS.keys()), required=True)

    group_parser.add_argument("-f", "--registrar-file", dest="registrar_file", type=str, help="A pre-computed registrar operation", default=None)

    group_parser = base_args_parser.add_argument_group("Output options")

    group_parser.add_argument("-s", "--dst-dir-name", dest="slide_src_dir", help="Name of the output directory", required=True)
    group_parser.add_argument("-t", "--dst-temp-dir", dest="results_dst_dir", help="Destination directory where intermediate files are stored", required=True)

    group_parser.add_argument("-o", "--dst-reg-dir", dest="registered_slide_dst_dir", help="Destination directory where the outputs files are stored", required=True)

    group_parser.add_argument("-c", "--codec", dest="compression", type=str, choices=["none", "jpeg", "deflate", "packbits", "ccittfax4", "lzw", "webp", "zstd", "jp2k"], help="Compression codec", default="deflate")
    group_parser.add_argument("-cq", "--compression-quality", dest="Q", type=int, help="Compression quality 1-100", default=100)

    group_parser.add_argument("-y", "--apply-registration", dest="apply_registration", action="store_true", help="Whether apply registration to the high-resolution image or not. If not required, this will only generate the transformation matrix as a .csv file.", default=False)

    group_parser = base_args_parser.add_argument_group("Registration options")

    group_parser.add_argument("-n", "--non-rigid", dest="non_rigid", action="store_true", help="Whether apply non-rigid registration or not", default=False)
    group_parser.add_argument("-m", "--micro", dest="micro", action="store_true", help="Whether apply micro rigid registration or not", default=False)

    group_parser.add_argument("-mp", "--max-processed-size", dest="max_processed_image_dim_px", type=int, help="Maximum width or height of processed images", default=registration.DEFAULT_MAX_PROCESSED_IMG_SIZE)
    group_parser.add_argument("-mn", "--max-nonrigid-size", dest="max_non_rigid_registration_dim_px", type=int, help="Maximum width or height of images used for non-rigid registration", default=registration.DEFAULT_MAX_NON_RIGID_REG_SIZE)
    group_parser.add_argument("-mm", "--max-micro-size", dest="max_micro_reg_size", type=int, help="Maximum width or height of images used for further micro rigid registration", default=registration.DEFAULT_MAX_MICRO_REG_SIZE)

    group_parser.add_argument("-mt", "--metric-type", dest="metric_type", type=str, choices=["distance", "similarity"], help="Metric used for registration optimization process", default="distance")

    group_parser.add_argument("-mc", "--matcher-class", dest="matcher_cls", type=str, choices=["Vgg", "default"], help="Matcher used for rigid registration", default="default")
    group_parser.add_argument("-nrc", "--non-rigid-class", dest="non_rigid_registrar_cls", type=str, choices=["SimpleElastix", "default"], help="Algorithm used for non-rigid registration", default="default")

    return base_args_parser


def add_advanced_arguments(dynamic_parser, selected_option, image_mode="", option_name="", prefix="", options_dict=None):
    """
    This function adds advanced arguments to the argument parser from a JSON file.

    :param dynamic_parser: An ArgumentParser used for an specific group of options.
    :param selected_option: The option that was selected by the user for this group of options.
    :param image_mode: Whether it is used for the `reference`, or `input` image.
    :param option_name: The name of the option, such as `preprocessor`, or `reader`.
    :param prefix: A prefix added to the option flag.
    :param json_filename: The name of the JSON file from where these advanced arguments are loaded.
    """
    group = dynamic_parser.add_argument_group(f"Advanced options for {image_mode} {option_name} [{selected_option}]")

    option_args = options_dict[selected_option]["arguments"]
    for param, param_props in option_args.items():
        kwargs = {
            "dest": prefix + "_" + param,
            "help": param_props.get("label", "") + ". " + param_props.get("help", "")
        }

        if param_props["type"] == "bool":
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = ARGUMENTS_TYPES.get(param_props["type"])

        if "options" in param_props:
            kwargs["choices"] = param_props["options"]

        group.add_argument("--" + prefix + "-" + param.lower(), **kwargs)


def preprocessors_arguments(base_parser):
    """
    This function generates the argument parser for options of methods such as image preprocessors and image readers.

    :param base_parser: Base argument parser with the general options for executing this program.
    :param args: Parsed arguments from the general options.
    """
    args, _ = base_parser.parse_known_args()

    image_modes = [
        ("reference", "r"),
        ("input", "i")
    ]
    option_classes = [
        ("preprocessor", "p", PREPROCESSORS, None),
        ("reader", "r", IMAGE_READERS, {})
    ]

    parser = base_parser
    modes_count = len(image_modes) * len(option_classes)
    for m_idx, ((mode_name, mode_prefix), (opt_name, opt_prefix, opt_dict, _)) in enumerate(product(image_modes, option_classes)):
        parser = argparse.ArgumentParser(
            parents=[parser],
            add_help=m_idx == (modes_count - 1)
        )

        if args.__dict__[mode_name + "_image_" + opt_name]:
            add_advanced_arguments(
                parser,
                args.__dict__[mode_name + "_image_" + opt_name],
                image_mode=mode_name,
                option_name=opt_name,
                prefix=mode_prefix + opt_prefix,
                options_dict=opt_dict
            )

    base_args = parser.parse_args()

    if '-h' in sys.argv or '--help' in sys.argv:
        # Create a parser just for display purposes that INCLUDES -h
        help_parser = argparse.ArgumentParser(
            parents=[base_parser],
            description="Main script help. Select a --reference-image-preprocessor, --reference-image-reader, --input-image-preprocessor, and a --input-image-reader to see advanced options."
        )
        help_parser.print_help()
        sys.exit(0)

    base_args_dict = dict(base_args.__dict__)

    option_prefixes = []
    for (mode_name, mode_prefix), (opt_name, opt_prefix, _, opt_default) in product(image_modes, option_classes):
        mode_opt_kwargs = {
            k.split(mode_prefix + opt_prefix + "_")[1]: v
            for k, v in base_args_dict.items()
            if k.startswith(mode_prefix + opt_prefix + "_")
        }
        if not len(mode_opt_kwargs):
            mode_opt_kwargs = opt_default

        option_prefixes.append(mode_prefix + opt_prefix + "_")

        base_args_dict[mode_name + "_image_" + opt_name + "_kwargs"] = mode_opt_kwargs

    base_args_dict = {
        k: v
        for k, v in base_args_dict.items()
        if not k.startswith(tuple(option_prefixes))
    }

    base_args = argparse.Namespace(**base_args_dict)

    return base_args
