import numpy as np

from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist import export_bioimageio
from csbdeep.utils import normalize


def test_pretrained():
    for t in ('2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018'):
        model = StarDist2D.from_pretrained(t)
        for output_format in ('zip', 'dir'):
            spec = export_bioimageio(model, f"modelzoo/{t}{'.zip' if output_format =='.zip' else ''}",
                                 output_format=output_format,
                                 test_inputs = [],
                                 test_outputs = [],
                                 validate=True)
    

if __name__ == '__main__':

    # basic example of converting some models int modelzoo format (zip compressed folder)

    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    test_inp   = test_image_nuclei_2d()
    test_out,_ = model.predict_instances(normalize(test_inp))

    spec = export_bioimageio(model, f"modelzoo/stardist_test",
                             output_format='dir',
                             test_inputs = [test_inp],
                             test_outputs = [test_out],
                             validate=False)
        
    spec = export_bioimageio(model, f"modelzoo/stardist_test.zip",
                             output_format='zip',
                             test_inputs = [test_inp],
                             test_outputs = [test_out],
                             validate=True)
        


