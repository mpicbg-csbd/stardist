from __future__ import absolute_import, print_function
import warnings
import pathlib
def format_Warning(message, category, filename, lineno, line=''):
    return f"{pathlib.Path(filename).name} ({lineno}): {message}\n"
warnings.formatwarning = format_Warning

from .version import __version__

# TODO: which functions to expose here? all?
from .nms import non_maximum_suppression, non_maximum_suppression_3d, non_maximum_suppression_3d_sparse
from .utils import edt_prob, fill_label_holes, sample_points, calculate_extents, export_imagej_rois, gputools_available
from .geometry import star_dist,   polygons_to_label,   relabel_image_stardist, ray_angles, dist_to_coord
from .geometry import star_dist3D, polyhedron_to_label, relabel_image_stardist3D
from .plot.plot import random_label_cmap, draw_polygons, _draw_polygons
from .plot.render import render_label, render_label_pred
from .rays3d import rays_from_json, Rays_Cartesian, Rays_SubDivide, Rays_Tetra, Rays_Octo, Rays_GoldenSpiral, Rays_Explicit
from .sample_patches import sample_patches
from .export_bioimageio import export_bioimageio
