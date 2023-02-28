import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

class DataPlot:

    def __init__(self):
        self.fig_width = 20
        self.fig_height = 10

    def histogram(self, dataset, plot_name, ncolumns):
        """Plot histogram based on input dataset"""
        fig, axes = plt.subplots(math.ceil(dataset.shape[1] / ncolumns), ncolumns,
                                 figsize=(self.fig_width, self.fig_height))
        spare_axes = ncolumns - dataset.shape[1] % ncolumns
        if spare_axes == ncolumns:
            spare_axes = 0
        for axis in range(ncolumns - 1,  ncolumns - 1 - spare_axes, -1):
            fig.delaxes(axes[math.ceil(dataset.shape[1] / ncolumns) - 1, axis])
        ax = axes.ravel()
        for i in range(dataset.shape[1]):
            ax[i].hist(dataset.iloc[:, i], histtype='stepfilled', bins=25, alpha=0.25, color="#0000FF", lw=0)
            ax[i].set_title(dataset.keys()[i], fontsize=10, y=1.0, pad=-14, fontweight='bold')
            ax[i].grid(visible=True)
            ax[i].tick_params(axis='both', labelsize=8)
            ax[i].set_ylabel('Frequency', fontsize=8)
            ax[i].set_xlabel('Feature magnitude', fontsize=8)
        fig.suptitle('Histogram for life expectancy dataset features', fontsize=18, fontweight='bold')
        plt.subplots_adjust(top=0.85)
        fig.tight_layout()
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.close()

    def target_vs_feature(self, dataset, target, plot_name, ncolumns):
        """Plot the target vs each feature"""
        fig, axes = plt.subplots(math.ceil(dataset.shape[1] / ncolumns), ncolumns,
                                 figsize=(self.fig_width, self.fig_height))
        spare_axes = ncolumns - dataset.shape[1] % ncolumns
        if spare_axes == ncolumns:
            spare_axes = 0
        for axis in range(ncolumns - 1,  ncolumns - 1 - spare_axes, -1):
            fig.delaxes(axes[math.ceil(dataset.shape[1] / ncolumns) - 1, axis])
        ax = axes.ravel()
        for i in range(dataset.shape[1]):
            ax[i].scatter(dataset[target], dataset.iloc[:, i], s=10, marker='o', c='blue')
            ax[i].grid(visible=True)
            ax[i].tick_params(axis='both', labelsize=8)
            ax[i].set_ylabel(dataset.keys()[i], fontsize=10, fontweight='bold')
            ax[i].set_xlabel(target, fontsize=10, fontweight='bold')
        fig.suptitle('Assessment between life expectancy and each feature', fontsize=24, fontweight='bold')
        plt.subplots_adjust(top=0.85)
        fig.tight_layout()
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.close()

    def plot_results(self, loss, val_loss, description, tag):
        """Plot the accuracy results for train and validation sets per each epoch"""
        cmap = cm.get_cmap('tab10')
        colors = cmap.colors
        fig, axes = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
        text_str = ''
        max_val = 0
        min_val = 100
        for i in range(len(loss)):
            plt.plot(range(1, len(loss[i]) + 1), loss[i], ls='-', lw=2, color=colors[i % len(colors)],
                     label='Training loss: {}'.format(description[i]))
            plt.plot(range(1, len(val_loss[i]) + 1), val_loss[i], ls='--', lw=2, color=colors[i % len(colors)],
                     label='Validation loss: {}'.format(description[i]))
            plt.xlabel('EPOCHS', fontweight='bold', fontsize=14)
            plt.ylabel('LOSS', fontweight='bold', fontsize=14)
            text_str += '\nTRAINING LOSS {} = {}'.format(description[i], round(loss[i][-1], 4))
            text_str += '\nVALIDATION LOSS {} = {}'.format(description[i], round(val_loss[i][-1], 4))
            max_val = max(max_val, loss[i][-1], val_loss[i][-1])
            min_val = min(min_val, loss[i][-1], val_loss[i][-1])
        plt.ylim(min_val * 0.8, max_val * 1.2)
        plt.title('LOSS RESULTS ' + tag.upper() + text_str, fontweight='bold', fontsize=18)
        plt.legend()
        plt.grid()
        fig.tight_layout()
        plt.savefig('Loss results ' + tag + '.png', bbox_inches='tight')
        plt.close()

    def plot_predictions(self, preds, target, models_to_load):
        cmap = cm.get_cmap('tab10')
        colors = cmap.colors
        fig, axes = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
        for i in range(len(models_to_load)):
            if len(models_to_load) == 1:
                plt.plot(preds, ls='-', lw=2, color=colors[i % len(colors)],
                         label='Prediction ({})'.format(models_to_load))
            else:
                plt.plot(preds[i], ls='-', lw=2, color=colors[i % len(colors)],
                         label='Prediction ({})'.format(models_to_load[i]))
        plt.plot(target, ls='--', lw=2, color='blue', label='Target')
        plt.xlabel('TIME', fontweight='bold', fontsize=14)
        plt.ylabel('TEMPERATURE [C]', fontweight='bold', fontsize=14)
        plt.title('TESTING SET PREDICTION', fontweight='bold', fontsize=18)
        plt.legend()
        plt.grid()
        fig.tight_layout()
        plt.savefig('Testing set prediction.png', bbox_inches='tight')
        plt.close()
