from hgdecode.classes import CrossValidation

subj_results_dir = '/Users/davidemiani/Desktop/results/hgdecode/dl' \
                   '/DeepConvNet/2019-01-16_09-57-27/subj01'
figures_dir = '/Users/davidemiani/Desktop/results/hgdecode/dl/DeepConvNet' \
              '/2019-01-16_09-57-27/statistics/figures/subj01'
tables_dir = '/Users/davidemiani/Desktop/results/hgdecode/dl/DeepConvNet' \
             '/2019-01-16_09-57-27/statistics/tables'

CrossValidation.cross_validate(
    subj_results_dir=subj_results_dir,
    figures_dir=figures_dir,
    tables_dir=tables_dir
)
