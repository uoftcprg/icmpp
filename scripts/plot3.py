from sys import argv, stdin

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_PATH = argv[1]


def main():
    df = pd.read_csv(stdin, index_col=0)
    df = df.loc[(df['count'] <= 10) & (df['algorithm'] == 'EBCM')]
    df['error'] = df['target'] - df['output']

    df.sort_values('chip_percentage', ascending=False, inplace=True)

    quartile_count = len(df) // 4
    df['group'] = (
        ['Large stacks'] * quartile_count
        + ['Medium stacks'] * (len(df) - 2 * quartile_count)
        + ['Small stacks'] * quartile_count
    )

    plt.figure()
    sns.barplot(df, x='group', y='error', hue='group')
    plt.xlabel('Group')
    plt.ylabel('Mean residual')
    plt.title('The mean residuals of the EBCM')
    plt.tight_layout()
    plt.savefig(RESULTS_PATH)


if __name__ == '__main__':
    main()
