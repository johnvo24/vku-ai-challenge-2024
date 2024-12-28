from vnindex.visualize.read_data import *
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Analysis():
  def __init__(self):
    self.data = read_data('res/dataset.csv')
    self.data = execute(self.data)
    print(self.data.head(5))

    self.analysis_dir = 'res/analysis'
    if not os.path.exists(self.analysis_dir):
        os.makedirs(self.analysis_dir)

  # Khám phá phân bố dữ liệu
  def plot_histograms(self):
    """Khám phá phân bố dữ liệu cho các cột PRICE, OPEN, HIGH, LOW, VOL, CHANGE."""
    columns = ['PRICE', 'OPEN', 'HIGH', 'LOW', 'VOL', 'CHANGE']
    plt.figure(figsize=(15, 10))

    for i, col in enumerate(columns, 1):
        plt.subplot(2, 3, i)
        sns.histplot(self.data[col], kde=True)
        plt.title(f'Histogram of {col}')

    plt.tight_layout()
    self.save_plot(plt, 'histograms.png')

  def save_plot(self, plot, filename):
    """Lưu plot vào thư mục 'res/analysis'."""
    plot_path = os.path.join(self.analysis_dir, filename)
    plot.savefig(plot_path)
    plt.close()

analysis = Analysis()
analysis.plot_histograms()
