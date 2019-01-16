from hgdecode.classes import CrossValidation

subj_results_dir = '/Users/davidemiani/Desktop/test/subj01'
figures_dir = '/Users/davidemiani/Desktop/test/statistics/figures/subj01'
tables_dir = '/Users/davidemiani/Desktop/test/statistics/tables'

CrossValidation.cross_validate(
    subj_results_dir=subj_results_dir,
    figures_dir=figures_dir,
    tables_dir=tables_dir
)
