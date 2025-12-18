# Copyright The Jackson Laboratory, September 2025
# Contact: Fernando Cervantes (fernando.cervantes@jax.org)
import os

import numpy as np

from valis import registration
from valis.feature_matcher import Matcher
from valis.feature_detectors import VggFD

import utils


def init_non_rigid_parameters(non_rigid_registrar_cls, non_rigid=False):
    if non_rigid:
        if (non_rigid_registrar_cls is not None
           and non_rigid_registrar_cls == "SimpleElastix"):
            non_rigid_registrar_cls = utils.SimpleElastixWarper2
            non_rigid_reg_params = {}
        else:
            # Use default
            non_rigid_registrar_cls = registration.DEFAULT_NON_RIGID_CLASS
            non_rigid_reg_params = registration.DEFAULT_NON_RIGID_KWARGS

    else:
        non_rigid_registrar_cls = None
        non_rigid_reg_params = None

    return non_rigid_registrar_cls, non_rigid_reg_params


def init_registrar(
        slide_src_dir,
        results_dst_dir,
        reference_image,
        input_image,
        metric_type="distance",
        matcher_cls="default",
        non_rigid_registrar_cls=None,
        non_rigid_reg_params=None,
        max_processed_image_dim_px=registration.DEFAULT_THUMBNAIL_SIZE,
        max_non_rigid_registration_dim_px=registration.DEFAULT_MAX_NON_RIGID_REG_SIZE):
    """
    Initializes a registrar object for image registration.

    :param slide_src_dir: Name of the directory where the intermediate outputs are stored.
    :param results_dst_dir: Directory where the intermediate outputs are stored.
    :param reference_image: Reference image to register towards into.
    :param input_image: Image to register towards the reference image.
    :param non_rigid: Whether apply non-rigid registration or not.
    :param metric_type: String describing what the custom metric function returns, e.g. 'similarity' or 'distance'.
    :param matcher_cls: Matcher used to identify features in both images for registration.
    :param non_rigid_registrar_cls: Uninstantiated NonRigidRegistrar class that will be used by `non_rigid_registrar` to calculate the deformation fields between images.
    :param max_processed_image_dim_px: Maximum width or height of processed images.
    :param max_non_rigid_registration_dim_px: Maximum width or height of images used for non-rigid registration.
    """
    if matcher_cls is not None and matcher_cls == "Vgg":
        matcher = Matcher(feature_detector=VggFD(),
                          metric_type=metric_type)
    else:
        matcher = registration.DEFAULT_MATCHER

    # Create a Valis object and use it to register the slides inside
    # the slide_src_dir directory
    registrar = registration.Valis(
        slide_src_dir,
        results_dst_dir,
        img_list=[reference_image, input_image],
        reference_img_f=reference_image,
        align_to_reference=reference_image is not None,
        matcher=matcher,
        matcher_for_sorting=Matcher(feature_detector=VggFD(),
                                    metric_type=metric_type),
        non_rigid_registrar_cls=non_rigid_registrar_cls,
        non_rigid_reg_params=non_rigid_reg_params,
        max_processed_image_dim_px=max_processed_image_dim_px,
        max_non_rigid_registration_dim_px=max_non_rigid_registration_dim_px,
    )

    return registrar


def run_registration(
        registrar,
        reference_image,
        input_image,
        reference_image_reader=None,
        reference_image_reader_kwargs=None,
        reference_image_preprocessor=None,
        reference_image_preprocessor_kwargs=None,
        input_image_reader=None,
        input_image_reader_kwargs=None,
        input_image_preprocessor=None,
        input_image_preprocessor_kwargs=None,
        non_rigid_reg_params=None,
        micro=False,
        non_rigid_registrar_cls="default",
        max_micro_reg_size=registration.DEFAULT_MAX_MICRO_REG_SIZE):

    processor_dict = {
        reference_image: [utils.PREPROCESSORS[reference_image_preprocessor]["class"],
                          reference_image_preprocessor_kwargs],
        input_image: [utils.PREPROCESSORS[input_image_preprocessor]["class"],
                      input_image_preprocessor_kwargs],
    }
    reader_dict = {
        reference_image: [utils.IMAGE_READERS[reference_image_reader]["class"],
                          reference_image_reader_kwargs],
        input_image: [utils.IMAGE_READERS[input_image_reader]["class"],
                      input_image_reader_kwargs]
    }

    _ = registrar.register(processor_dict=processor_dict,
                           reader_dict=reader_dict)

    if micro:
        registrar.register_micro(
            max_non_rigid_registration_dim_px=max_micro_reg_size,
            reference_img_f=reference_image,
            align_to_reference=reference_image is not None,
            processor_dict=processor_dict,
            non_rigid_registrar_cls=non_rigid_registrar_cls,
            non_rigid_reg_params=non_rigid_reg_params
        )


def save_outputs(
        registrar,
        reference_image,
        input_image,
        registered_slide_dst_dir,
        apply_registration=False,
        non_rigid=False,
        compression="deflate",
        Q=100):
    # Save registration transform matrices.
    # If requested by the user, save the transformed slides as ome.tiff too.
    ref_slide = registrar.get_slide(reference_image)

    non_ref_slide = registrar.get_slide(input_image)
    dst_M_fn = utils.same_name_check(
        os.path.join(registered_slide_dst_dir,
                     f"{non_ref_slide.name}_to_{ref_slide.name}_transformation_matrix"),
        "csv"
    )

    # Save transformation matrix
    np.savetxt(dst_M_fn, non_ref_slide.M, delimiter=",", fmt="%0.18f")

    if apply_registration:
        dst_img_fn = utils.same_name_check(
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
    base_parser = utils.basic_arguments()
    args = utils.preprocessors_arguments(base_parser)

    (non_rigid_registrar_cls,
     non_rigid_reg_params) = init_non_rigid_parameters(
         args.non_rigid_registrar_cls,
         non_rigid=args.non_rigid)

    if args.registrar_file is not None:
        registrar = registration.load_registrar(args.registrar_file)

    else:
        registrar = init_registrar(
            args.slide_src_dir,
            args.results_dst_dir,
            args.reference_image,
            args.input_image,
            metric_type=args.metric_type,
            matcher_cls=args.matcher_cls,
            non_rigid_registrar_cls=non_rigid_registrar_cls,
            non_rigid_reg_params=non_rigid_reg_params,
            max_processed_image_dim_px=args.max_processed_image_dim_px,
            max_non_rigid_registration_dim_px=args.max_non_rigid_registration_dim_px
        )

    run_registration(
        registrar,
        args.reference_image,
        args.input_image,
        reference_image_reader=args.reference_image_reader,
        reference_image_reader_kwargs=args.reference_image_reader_kwargs,
        reference_image_preprocessor=args.reference_image_preprocessor,
        reference_image_preprocessor_kwargs=args.reference_image_preprocessor_kwargs,
        input_image_reader=args.input_image_reader,
        input_image_reader_kwargs=args.input_image_reader_kwargs,
        input_image_preprocessor=args.input_image_preprocessor,
        input_image_preprocessor_kwargs=args.input_image_preprocessor_kwargs,
        non_rigid=args.non_rigid,
        non_rigid_reg_params=non_rigid_reg_params,
        micro=args.micro,
        non_rigid_registrar_cls=non_rigid_registrar_cls,
        max_micro_reg_size=args.max_micro_reg_size
    )

    save_outputs(
        registrar,
        args.reference_image,
        args.input_image,
        args.registered_slide_dst_dir,
        apply_registration=args.apply_registration,
        non_rigid_registrar_cls=non_rigid_registrar_cls,
        compression=args.compression,
        Q=args.Q
    )