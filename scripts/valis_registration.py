# Copyright The Jackson Laboratory, September 2025
# Contact: Fernando Cervantes (fernando.cervantes@jax.org)

import os
import argparse

from valis import registration, non_rigid_registrars
from valis.feature_matcher import Matcher
from valis.feature_detectors import VggFD
from valis.preprocessing import (ChannelGetter,
                                 ColorfulStandardizer,
                                 JCDist,
                                 OD,
                                 ColorDeconvolver,
                                 Luminosity,
                                 BgColorDistance,
                                 StainFlattener,
                                 Gray,
                                 HEDeconvolution)
from valis.slide_io import VipsSlideReader, BioFormatsSlideReader

import numpy as np
import itk


class BioFormatsSlideReaderZ(BioFormatsSlideReader):
    def __init__(self, src_f, series=None, z_slice=0, *args, **kwargs):
        super(BioFormatsSlideReaderZ, self).__init__(src_f, series=series, *args, **kwargs)
        self._z_slice = z_slice

    def slide2vips(self, level, series=None, xywh=None, tile_wh=None, z=0, t=0, *args, **kwargs):
        return super(BioFormatsSlideReaderZ, self).slide2vips(level, series=series, xywh=xywh, tile_wh=tile_wh, z=self._z_slice, t=t, *args, **kwargs)


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


PREPROCESSORS = {
    "ChannelGetter": [ChannelGetter, {"channel": "dapi", "adaptive_eq": True, "invert": False}],
    "HEDeconvolution": HEDeconvolution
}


IMAGE_READERS = {
    "bioformats": BioFormatsSlideReader,
    "bioformats-Z": BioFormatsSlideReaderZ,
    "vips": VipsSlideReader
}


def same_name_check(out_basename, out_extension):
    if not os.path.isfile(f"{out_basename}.{out_extension}"):
        return f"{out_basename}.{out_extension}"

    same_name_count = 1
    while os.path.isfile(f"{out_basename}_({same_name_count}).{out_extension}"):
        same_name_count += 1

    return f"{out_basename}_({same_name_count}).{out_extension}"


def main(slide_src_dir, results_dst_dir, registered_slide_dst_dir, img_list,
         apply_registration=False,
         registrar_file=None,
         non_rigid=False,
         micro=False,
         metric_type="distance",
         matcher_cls="default",
         non_rigid_registrar_cls="default",
         max_processed_image_dim_px=registration.DEFAULT_THUMBNAIL_SIZE,
         max_non_rigid_registration_dim_px=registration.DEFAULT_MAX_NON_RIGID_REG_SIZE,
         max_micro_reg_size=registration.DEFAULT_MAX_MICRO_REG_SIZE,
         compression="deflate",
         Q=100):
    processor_dict = None

    img_list = map(lambda img_fn: img_fn.split(":"), img_list)

    img_list_preprocessors = [
         (*img_fn, * [None] * (4 - len(img_fn)))
         for img_fn in img_list
    ]
    img_list_preprocessors = [
        (img_fn, preprocessor, img_reader, {"z_slice": int(img_reader_arg)} if img_reader_arg is not None else {})
        for img_fn, preprocessor, img_reader, img_reader_arg in img_list_preprocessors
    ]

    (img_list,
     preprocessors,
     image_readers,
     image_readers_kwargs) = list(zip(*img_list_preprocessors))

    processor_dict = {
        fn: PREPROCESSORS.get(func_name, None)
        for fn, func_name in zip(img_list, preprocessors)
    }
    reader_dict = {
        fn: [IMAGE_READERS.get(func_name, IMAGE_READERS["vips"]), func_kwargs]
        for fn, func_name, func_kwargs in zip(img_list, image_readers,
                                              image_readers_kwargs)
    }

    print("Image processors:", processor_dict)
    print("Image readers:", reader_dict)

    # By default the first image passed in the image list is used as reference
    # to align the rest of the images to it.
    reference_img_f = img_list[0]

    if registrar_file is not None:
        registrar = registration.load_registrar(registrar_file)

    else:
        if matcher_cls is not None and matcher_cls == "Vgg":
            matcher = Matcher(feature_detector=VggFD(),
                              metric_type=metric_type)
        else:
            matcher = registration.DEFAULT_MATCHER

        if non_rigid:
            if (non_rigid_registrar_cls is not None
               and non_rigid_registrar_cls == "SimpleElastix"):
                non_rigid_registrar_cls = SimpleElastixWarper2
                non_rigid_reg_params = {}
            else:
                # Use default
                non_rigid_registrar_cls = registration.DEFAULT_NON_RIGID_CLASS
                non_rigid_reg_params = registration.DEFAULT_NON_RIGID_KWARGS

        else:
            non_rigid_registrar_cls = None
            non_rigid_reg_params = None

        # Create a Valis object and use it to register the slides inside
        # the slide_src_dir directory
        registrar = registration.Valis(
            slide_src_dir,
            results_dst_dir,
            img_list=img_list,
            reference_img_f=reference_img_f,
            align_to_reference=reference_img_f is not None,
            matcher=matcher,
            matcher_for_sorting=Matcher(feature_detector=VggFD(),
                                        metric_type=metric_type),
            non_rigid_registrar_cls=non_rigid_registrar_cls,
            non_rigid_reg_params=non_rigid_reg_params,
            max_processed_image_dim_px=max_processed_image_dim_px,
            max_non_rigid_registration_dim_px=max_non_rigid_registration_dim_px,
        )

        (rigid_registrar,
         non_rigid_registrar,
         error_df) = registrar.register(processor_dict=processor_dict,
                                        reader_dict=reader_dict)

        if micro:
            registrar.register_micro(
                max_non_rigid_registration_dim_px=max_micro_reg_size,
                reference_img_f=reference_img_f,
                align_to_reference=reference_img_f is not None,
                processor_dict=processor_dict,
                non_rigid_registrar_cls=non_rigid_registrar_cls,
                non_rigid_reg_params=non_rigid_reg_params
            )

    # Save registration transform matrices.
    # If requested by the user, save the transformed slides as ome.tiff too.
    ref_slide = registrar.get_slide(reference_img_f)
    for img_name in registrar.original_img_list:
        if img_name == reference_img_f:
            continue

        non_ref_slide = registrar.get_slide(img_name)
        dst_M_fn = same_name_check(
            os.path.join(registered_slide_dst_dir,
                         f"{non_ref_slide.name}_to_{ref_slide.name}_transformation_matrix"),
            "csv"
        )

        # Save transformation matrix
        np.savetxt(dst_M_fn, non_ref_slide.M, delimiter=",", fmt="%0.18f")

        if apply_registration:
            dst_img_fn = same_name_check(
                os.path.join(registered_slide_dst_dir,
                             f"{non_ref_slide.name}_to_{ref_slide.name}"),
                "ome.tiff"
            )

            non_ref_slide.warp_and_save_slide(
                dst_img_fn,
                non_rigid=non_rigid,
                crop=registration.CROP_REF,
                compression=compression,
                Q=Q
            )

    # Kill the JVM
    registration.kill_jvm()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser("VALIS registration", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args_parser.add_argument("-s", "--dst-dir-name", dest="slide_src_dir", help="Name of the output directory")
    args_parser.add_argument("-i", "--images", dest="img_list", type=str, nargs="+", help="Images to register. If not specified, all supported images from the source directory are used. Processing functions can be passed after each image (e.g. he_img:HEDeconvolution multichannel_img:ChannelGetter)")

    args_parser.add_argument("-o", "--dst-dir", dest="results_dst_dir", help="Destination parent directory")

    args_parser.add_argument("-d", "--dst-reg-dir", dest="registered_slide_dst_dir", help="Destination directory where the ome-tiff files are stored")

    args_parser.add_argument("-f", "--registrar-file", dest="registrar_file", type=str, help="A pre-computed registrar operation", default=None)

    args_parser.add_argument("-n", "--non-rigid", dest="non_rigid", action="store_true", help="Whether apply non-rigid registration or not", default=False)
    args_parser.add_argument("-m", "--micro", dest="micro", action="store_true", help="Whether apply micro rigid registration or not", default=False)

    args_parser.add_argument("-y", "--apply-registration", dest="apply_registration", action="store_true", help="Whether apply registration to the high-resolution image or not", default=False)

    args_parser.add_argument("-mp", "--max-processed-size", dest="max_processed_image_dim_px", type=int, help="Maximum width or height of processed images", default=registration.DEFAULT_MAX_PROCESSED_IMG_SIZE)
    args_parser.add_argument("-mn", "--max-nonrigid-size", dest="max_non_rigid_registration_dim_px", type=int, help="Maximum width or height of images used for non-rigid registration", default=registration.DEFAULT_MAX_NON_RIGID_REG_SIZE)
    args_parser.add_argument("-mm", "--max-micro-size", dest="max_micro_reg_size", type=int, help="Maximum width or height of images used for further micro rigid registration", default=registration.DEFAULT_MAX_MICRO_REG_SIZE)

    args_parser.add_argument("-t", "--metric-type", dest="metric_type", type=str, choices=["distance", "similarity"], help="Metric used for registration optimization process", default="distance")

    args_parser.add_argument("-mc", "--matcher-class", dest="matcher_cls", type=str, choices=["Vgg", "default"], help="Matcher used for rigid registration", default="default")
    args_parser.add_argument("-nrc", "--non-rigid-class", dest="non_rigid_registrar_cls", type=str, choices=["SimpleElastix", "default"], help="Algorithm used for non-rigid registration", default="default")

    args_parser.add_argument("-c", "--codec", dest="compression", type=str, choices=["none", "jpeg", "deflate", "packbits", "ccittfax4", "lzw", "webp", "zstd", "jp2k"], help="Compression codec", default="deflate")
    args_parser.add_argument("-cq", "--compression-quality", dest="Q", type=int, help="Compression quality 1-100", default=100)

    args = args_parser.parse_args()

    main(**args.__dict__)
