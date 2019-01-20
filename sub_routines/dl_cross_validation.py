import os
from collections import OrderedDict
from hgdecode.classes import CrossValidation

datetime_dir = '/Users/davidemiani/OneDrive - Alma Mater Studiorum ' \
               'Università di Bologna/TesiMagistrale_DavideMiani/' \
               'results/hgdecode/dl/DeepConvNet/2019-01-15_01-54-25'
subj_dirs = os.listdir(datetime_dir)
subj_dirs.sort()
subj_dirs.remove('model_report.txt')
subj_dirs.remove('statistics')
subj_dirs = [os.path.join(datetime_dir, x) for x in subj_dirs]

name_to_start_codes = OrderedDict([('Right Hand', [1]),
                                   ('Left Hand', [2]),
                                   ('Rest', [3]),
                                   ('Feet', [4])])

for subj_dir in subj_dirs:
    CrossValidation.cross_validate(subj_results_dir=subj_dir,
                                   label_names=name_to_start_codes)
