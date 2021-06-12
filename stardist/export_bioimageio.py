import numpy as np
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import hashlib
import shutil
import tempfile
from ruamel.yaml import YAML

from csbdeep.utils import axes_check_and_normalize, axes_dict, move_image_axes


def _get_stardist_metadata():
    # from importlib.metadata import metadata
    from importlib_metadata import metadata
    from pkg_resources import get_distribution
    
    package_data = metadata('stardist')
    pkg_info = get_distribution('stardist')
    
    data = SimpleNamespace(
        description=package_data['Summary'],
        authors = list(dict(name=name.strip()) for name in package_data['Author'].split(',')),
        git_repo=package_data['Home-Page'],
        license=package_data['License'],
        version=package_data['Version'],
        requirements=('tensorflow', ) + tuple(map(str, pkg_info.requires()))
    )
    return data


def _get_stardist_dependencies():
    from pkg_resources import get_distribution
    
    pkg_info = get_distribution('stardist')
    reqs = ('tensorflow', ) + tuple(map(str, pkg_info.requires()))
    return reqs
    

def _get_weights_name(model, prefer="best"):
    #TODO factor that out (its the same as csbdeep.base_model)
    from itertools import chain
    # get all weight files and sort by modification time descending (newest first)
    weights_ext   = ('*.h5','*.hdf5')
    weights_files = chain(*(model.logdir.glob(ext) for ext in weights_ext))
    weights_files = reversed(sorted(weights_files, key=lambda f: f.stat().st_mtime))
    weights_files = list(weights_files)
    if len(weights_files) == 0:
        raise ValueError("Couldn't find any network weights (%s) to load." % ', '.join(weights_ext))
    weights_preferred = list(filter(lambda f: prefer in f.name, weights_files))
    weights_chosen = weights_preferred[0] if len(weights_preferred)>0 else weights_files[0]
    return weights_chosen.name


# def export_TF(model, fname=None):
#     from csbdeep.utils.tf import keras_import, export_SavedModel
#     K = keras_import('backend')
#     Model = keras_import('models', 'Model')
#     Input, Conv2D = keras_import('layers', 'Input', 'Conv2D')

#     x = Input((None,None,1), name = 'input')
#     y = Conv2D(1, 1, padding="same", name = 'prob')(x)
#     csbdeep_model = Model(x,y)

#     # if model.basedir is None and fname is None:
#     #     raise ValueError("Need explicit 'fname', since model directory not available (basedir=None).")

#     # csbdeep_model = Model(model.keras_model.inputs[0], model.keras_model.outputs[0])
#     fname = (model.logdir / 'TF_SavedModel.zip') if fname is None else Path(fname)
#     export_SavedModel(csbdeep_model, str(fname))
#     return csbdeep_model


# TODO: this could be turned into a base class method
def _default_bioimageio_spec(self, mode="tensorflow_saved_model_bundle", prefer_weights='best'):
    """the model specific spec parameters (e.g. axes stride) etc."""
    
    spec = SimpleNamespace()

    package_data = _get_stardist_metadata()
    
    # metadata
    spec.format_version = '0.3.2'
    spec.name           = 'StarDist Model'
    spec.timestamp      = datetime.now().isoformat()
    spec.description    = package_data.description
    spec.authors        = package_data.authors
    spec.cite           = [
            dict(text='Cell Detection with Star-Convex Polygons',
                 doi='https://doi.org/10.1007/978-3-030-00934-2_30'),
            dict(text='Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy',
                 doi='https://doi.org/10.1109/WACV45572.2020.9093435')
    ]

    spec.git_repo       = package_data.git_repo
    spec.tags           = ["stardist", "segmentation", "instance segmentation", "tensorflow"]
    spec.license        = package_data.license
    # spec.documentation  = 'https://github.com/stardist/stardist'

    spec.documentation  = 'README.md'
    # other stuff
    spec.covers = ["https://raw.githubusercontent.com/stardist/stardist/master/images/stardist_logo.jpg"] 

    spec.config = dict(stardist_version=package_data.version)

    spec.dependencies = 'pip:./requirements.txt'

    # get weights
    weights_name = _get_weights_name(self, prefer=prefer_weights)
    
    if mode=='keras_hdf5':
        fname_weights = self.logdir/weights_name
        with open(fname_weights, "rb") as f:
            bytes = f.read() # read entire file as bytes
            _hash = hashlib.sha256(bytes).hexdigest()
            sha256_weights  = hashlib.sha256(bytes).hexdigest()
        
        spec.weights = dict(keras_hdf5=dict(
        authors=["NN"],
        source=str(fname_weights),
        sha256=sha256_weights,
        ))
    elif mode=='tensorflow_saved_model_bundle':
        fname_bundle = self.logdir / 'TF_SavedModel.zip'
        self.load_weights(weights_name)
        model_csbdeep = self.export_TF(fname_bundle, single_output=True)
        # model_csbdeep = export_TF(self, fname_bundle)
        with open(fname_bundle, "rb") as f:
            bytes = f.read() # read entire file as bytes
            _hash = hashlib.sha256(bytes).hexdigest()
            sha256_bundle  = hashlib.sha256(bytes).hexdigest()
        
        spec.weights = dict(tensorflow_saved_model_bundle=dict(
            source=str(fname_bundle),
            authors=spec.authors,
            sha256=sha256_bundle,
        ))
    else:
        raise ValueError(f'unsupported mode {mode}')

    # TODO: this needs more attention, e.g. how axes are treated in a general way
    axes = self.config.axes.lower()
    img_axes_in = axes_check_and_normalize(axes,self.config.n_dim+1)
    net_axes_in = axes
    net_axes_out = axes_check_and_normalize(self._axes_out).lower()
    net_axes_lost = set(net_axes_in).difference(set(net_axes_out))
    img_axes_out = ''.join(a for a in img_axes_in if a not in net_axes_lost)

    ndim_tensor = self.config.n_dim + 2

    # input shape including batch size 
    div_by = list(self._axes_div_by(net_axes_in))
    input_shape = dict(min=[1]+div_by,step = [0]+div_by)

    if mode=='keras_hdf5':
        output_names = ("prob", "dist") +  (("class_prob",) if self._is_multiclass() else ())
        output_n_channels = (1, self.config.n_rays,) + ((1,) if self._is_multiclass() else ())
        output_scale = [1]+list(1/g for g in self.config.grid) + [0]
    elif mode=='tensorflow_saved_model_bundle':
        if self._is_multiclass():
            raise NotImplementedError('Tensorflow SaveModel not supported for multiclass models yet')
        # output_names = ("outputall",)
        # output_n_channels = (1 + self.config.n_rays,)
        # output_scale = [1]*(ndim_tensor-1) + [0]

        # output_names = ("prob",)
        # output_n_channels = (1,)
        # output_scale = [1]*(ndim_tensor-1) + [0]

        output_names = model_csbdeep.output_names
        output_n_channels = (1 + self.config.n_rays,)
        output_scale = [1]*(ndim_tensor-1) + [0]
        print(output_names)

    # input/output
    spec.inputs = [dict(name       = name,
                       data_type  = 'float32',
                       data_range = ['-inf', 'inf'],
                       axes       = 'b'+net_axes_in.lower(),
                       shape = input_shape,
                       preprocessing = [
                           dict(name="scale_range",
                                kwargs=dict(
                                    mode="per_sample",
                                    # TODO: might make it an option to normalize across channels...
                                    axes=net_axes_in.lower().replace('c', ''), 
                                    min_percentile=1,
                                    max_percentile=99.8,
                                ))]
                        ) for name in model_csbdeep.input_names]

    spec.outputs = [dict(name     = name,
                       data_type  = 'float32',
                       data_range = ['-inf', 'inf'],
                       axes       = 'b' + net_axes_out.lower(),
                       shape = dict(reference_input=model_csbdeep.inputs[0].name,
                                    scale=output_scale,
                                    offset=[1]*(ndim_tensor-1) + [n_channel]
                                    # scale=[1,1,1,1], 
                                    # offset=[0,0,0,0]
                                    )
                         ) for name, n_channel in zip(output_names, output_n_channels)]

    return spec


def export_bioimageio(model, outpath, test_inputs=[], test_outputs=[], output_format="dir", mode="tensorflow_saved_model_bundle", prefer_weights='best', validate=True, overwrite_spec_kwargs={}):
    """
    Export stardist model into bioimageio format, https://github.com/bioimage-io/spec-bioimage-io 
    
    Parameters
    ----------
    model : StarDist2D, StarDist3D
          the model to convert
    outpath : str, Path
          name of directory/zip file
    test_inputs : list of ndarray
          the list of test input images
    test_outputs : list of ndarray
          the list of test output images
    authors : list of str
          the list of model authors 
    output_format : str
          'dir' -> save as directory, 'zip' -> save and compress into zip file 
    mode : str
          either 'tensorflow_saved_model_bundle' or 'keras_hdf5'
    prefer_weights : str
          the weights to save (see model._find_and_load_weights)
    validate: bool
          if True, validates the spec (needs https://github.com/bioimage-io/python-bioimage-io)
    overwrite_spec_kwargs: dict
          a dictionary of spec parameters that will overwrite any default/inferred one 
          
    """

    # get default spec parameters
    spec = _default_bioimageio_spec(model, mode=mode, prefer_weights=prefer_weights)

    spec.__dict__.update(overwrite_spec_kwargs)
    # prepare demo input/outputs 
    if not len(test_inputs) == len(test_outputs):
        raise ValueError('test_inputs and test_outputs need to have same size')
    test_inputs  = tuple(map(np.asarray, test_inputs))
    test_outputs = tuple(map(np.asarray, test_outputs))


    def export_to_dir(spec, outdir):
        # create a local copy
        spec = SimpleNamespace(**vars(spec))
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        if not outdir.is_dir():
            raise ValueError(f'not a directory: {outdir}')

        # copy weights/bundle
        f_weights = Path(spec.weights[mode]['source'])
        shutil.copy(f_weights, outdir/f_weights.name)
        spec.weights[mode]['source'] = f'./{f_weights.name}'
        # copy other jsons
        for f in Path(model.logdir).glob('*.json'):
            shutil.copy(f, outdir/f.name)

        # create requirements
        with open(outdir/'requirements.txt', 'w', encoding='UTF-8') as f:
            f.write('\n'.join(_get_stardist_dependencies()))
        
        # copy test_inputs, outputs
        spec.test_inputs, spec.test_outputs = [], []
        for i, (inp, out) in enumerate(zip(test_inputs, test_outputs)):
            inp_name, out_name = f'./test_input_{i:04d}.npy', f'./test_output_{i:04d}.npy'
            np.save(outdir/inp_name, inp)
            np.save(outdir/out_name, out)
            spec.test_inputs.append(inp_name)
            spec.test_outputs.append(out_name)

        # write to file
        yaml = YAML(typ='rt')
        yaml.default_flow_style = False
        with open(outdir/'model.yaml', 'w', encoding='UTF-8') as f:
            yaml.dump(vars(spec), f)

        return spec

    outpath = Path(outpath)
    if output_format == 'dir':
        spec = export_to_dir(spec, outpath)
    elif output_format == 'zip':
        outpath = outpath.with_suffix('')
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = export_to_dir(spec, tmpdir)
            shutil.make_archive(outpath, output_format, tmpdir)
    else:
        raise ValueError("Unsupported output format '%s'" % output_format)

    if validate:        
        try:
            import bioimageio
        except ImportError as e:
            raise ImportError('Cannot find package `bioimageio` that is needed for validation. \nEither set validate=False or install with\n   pip install git+https://github.com/bioimage-io/python-bioimage-io')
        from bioimageio.spec.__main__ import verify_model_data, ValidationError
        from bioimageio.spec import maybe_convert_model
        spec_data =   maybe_convert_model(vars(spec))
        verify_model_data(spec_data)
    return spec
