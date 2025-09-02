from json import dump
from sys import stdin, stdout

from scipy.stats import sem, ttest_rel
import pandas as pd


def main():
    df = pd.read_csv(stdin, index_col=0)
    df['error'] = df['target'] - df['output']
    df['squared_error'] = df['error'] ** 2
    baseline_squared_errors = (
        df.loc[df['algorithm'] == 'BASELINE']['squared_error']
    )
    icm_squared_errors = df.loc[df['algorithm'] == 'ICM']['squared_error']
    ebcm_squared_errors = df.loc[df['algorithm'] == 'EBCM']['squared_error']
    one_sided_paired_ttest_result = ttest_rel(
        ebcm_squared_errors,
        icm_squared_errors,
        alternative='less',
    )
    data = {
        'end_of_day_count': df['end_of_day'].nunique(),
        'player_count': len(df.groupby(['end_of_day', 'player'])),
        'baseline_squared_error': {
            'mean': baseline_squared_errors.mean(),
            'standard_error': sem(baseline_squared_errors),
        },
        'icm_squared_error': {
            'mean': icm_squared_errors.mean(),
            'standard_error': sem(icm_squared_errors),
        },
        'ebcm_squared_error': {
            'mean': ebcm_squared_errors.mean(),
            'standard_error': sem(ebcm_squared_errors),
        },
        'one_sided_paired_ttest': {
            'statistic': one_sided_paired_ttest_result.statistic,
            'pvalue': one_sided_paired_ttest_result.pvalue,
            'df': one_sided_paired_ttest_result.df.item(),
        },
    }

    dump(data, stdout)


if __name__ == '__main__':
    main()
