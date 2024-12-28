import matplotlib.pyplot as plt
import seaborn as sns

class Chart:
    def __init__(self, data):
        self.data = data

    def ScatterPlot(self, x, y):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x=x, y=y, alpha=0.7)

        # plt.xticks(rotation=45)
        # plt.yticks(fontsize=10)
        plt.title(f'Relationship between {x} and {y}', fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        # plt.tight_layout()
        plt.grid()
        plt.savefig(f'res/charts/scatterplot_{x}_{y}.png', dpi=300, bbox_inches='tight')

    def PairPlot(self):
        sns.pairplot(self.data[['DATE', 'PRICE', 'OPEN', 'HIGH', 'LOW', 'VOL', 'CHANGE']])
        plt.savefig(f'res/charts/pairplot.png', dpi=300, bbox_inches='tight')
