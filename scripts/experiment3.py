from json import dump
from sys import stdin, stdout

from scipy.stats import sem, ttest_1samp
import pandas as pd


def main():
    df = pd.read_csv(stdin, index_col=0)
    df = df.loc[(df['count'] <= 10) & (df['algorithm'] == 'EBCM')]
    df['error'] = df['target'] - df['output']

    df.sort_values('chip_percentage', ascending=False, inplace=True)

    quartile_count = len(df) // 4
    dfs = {
        'large_stacks': df.iloc[:quartile_count],
        'medium_stacks': df.iloc[quartile_count:-quartile_count],
        'small_stacks': df.iloc[-quartile_count:],
    }
    data = {
        'end_of_day_count': df['end_of_day'].nunique(),
        'player_count': len(df.groupby(['end_of_day', 'player'])),
    }

    for key, df in dfs.items():
        one_sided_paired_ttest_result = ttest_1samp(df['error'], 0)

        data[key] = {
            'error': {
                'mean': df['error'].mean(),
                'standard_error': sem(df['error']),
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
