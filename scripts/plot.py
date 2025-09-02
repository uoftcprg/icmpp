from sys import argv, stdin

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_PATH = argv[1]


def main():
    df = pd.read_csv(stdin, index_col=0)
    df['error'] = df['target'] - df['output']
    df['squared_error'] = df['error'] ** 2

    plt.figure()

    ax = sns.barplot(df, x='algorithm', y='squared_error', hue='algorithm')

    ax.set_xticklabels(['Baseline', 'ICM', 'EBCM'])
    plt.xlabel('Algorithm')
    plt.ylabel('MSE')
    plt.title('The MSEs of the baseline, ICM, and EBCM')
    plt.tight_layout()
    plt.savefig(RESULTS_PATH)


if __name__ == '__main__':
    main()
