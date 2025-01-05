from json import load
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASELINES_DATA_PATH = argv[1]
ICMPP_DATA_PATH = argv[2]
ICMPP_NAME = 'ICM++'
BASELINE_NAMES = ('ICM',)
LIMITS = 0.065, 0.070


def main():
    with open(BASELINES_DATA_PATH) as file:
        baseline_data = load(file)

    icmpp_data = np.load(ICMPP_DATA_PATH)
    data = []

    for name in BASELINE_NAMES:
        data.append({'algorithm': name, 'error': baseline_data[name]})

    for i, error in enumerate(icmpp_data['errors']):
        data.append({'algorithm': ICMPP_NAME, 'fold': i + 1, 'error': error})

    df = pd.DataFrame(data)

    plt.figure()
    sns.barplot(df, x='algorithm', y='error', hue='algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('RMSE')
    plt.title('RMSEs of Poker Tournament Equity Calculations')
    plt.ylim(LIMITS)
    plt.show()


if __name__ == '__main__':
    main()
