import os
import random
import re
import time
import click
import matplotlib.pyplot as plt
import progressbar
from utils import ft_abs, ft_sqrt, ft_power, ft_sum

class Trainer(object):

    def __init__(self, data_file, sep, plot, model_file, epochs, batch_size, learning_rate):
        self.data_file = data_file
        self.sep = sep
        self.plot = plot
        self.model_file = model_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = [0.0, 0.0]
        self.x_data = []
        self.y_data = []
        self.labels = []
        # Read data
        self.read_data()
        if len(self.x_data) != len(self.y_data) or len(self.x_data) == 0:
            print("Error : no valid data found in %s" % self.data_file)
            return
        # Read model
        if len(self.model_file):
            self.read_model()
        # Plot data
        if self.plot:
            self.plot_data()

    def read_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file) as f:
                for line in f:
                    line = line.replace('\n', '')
                    line_data = line.split(self.sep)
                    if len(line_data) == 2 and all([value.isdigit() for value in line_data]):
                        self.x_data.append(int(line_data[0]))
                        self.y_data.append(int(line_data[1]))
                    elif len(line_data) == 2:
                        self.labels.append(line_data[0])
                        self.labels.append(line_data[1])

    def read_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, "r") as f:
                for line in f:
                    line = line.replace('\n', '')
                    line_data = line.split(self.sep)
                    r = re.compile('\d+(\.\d+)?')
                    if len(line_data) == 2 and all([r.match(value) for value in line_data]):
                        self.model[0] = float(line_data[0])
                        self.model[1] = float(line_data[1])

    def plot_data(self):
        #sort
        x_data, y_data = [list(t) for t in zip(*sorted(zip(self.x_data, self.y_data)))]
        #plot
        plt.plot(x_data, y_data)
        if len(self.labels):
            plt.xlabel(self.labels[0])
            plt.ylabel(self.labels[1])
        plt.show(block=True)

    def save_model(self):
        if not os.path.exists(self.model_file):
            mode = "w+"
        else:
            mode = "a"
        with open(self.model_file, mode) as f:
            f.write(str(self.model[0]) + self.sep + str(self.model[1]) + "\n")

    def train(self):
        theta_0 = 0.0
        theta_1 = 0.0
        # read model
        if self.model[0] != theta_0 or self.model[1] != theta_1:
            theta_0 = self.model[0]
            theta_1 = self.model[1]
        # process train
        self.train_loop()
        # write model file
        self.save_model()

    def train_loop(self):
        # shuffle datas
        l = list(zip(self.x_data, self.y_data))
        random.shuffle(l)
        x_data, y_data = zip(*l)
        # loop on epochs / batches / data_points
        for epoch in range(self.epochs):
            print("Training... Epoch : %s" % str(epoch + 1))
            for (batch_x, batch_y) in self.batches_generator(x_data, y_data):
                self.training_batch(batch_x, batch_y)
                
    def training_batch(self, batch_x, batch_y):
        matrix_0 = []
        for i, x_val in enumerate(batch_x):
            matrix_0.append(ft_abs(self.estimate(x_val) - batch_y[i]))
        self.model[0] = self.learning_rate * (1 / len(batch_x)) * ft_sum(matrix_0)
        matrix_1 = []
        for i, elem in enumerate(matrix_0):
            matrix_1.append(elem * batch_y[i])
        self.model[1] = self.learning_rate * (1 / len(batch_x)) * ft_sum(matrix_1)
        print()
    
    def estimate(self, x):
        y = self.model[0] + self.model[1] * x 
        return y
    
    def batches_generator(self, x_data, y_data):
        for i in progressbar.progressbar(range(len(x_data) // self.batch_size)):
            start = i * self.batch_size
            end = i * self.batch_size + self.batch_size
            batch_x = x_data[start:end]
            batch_y = y_data[start:end]
            yield batch_x, batch_y
    
@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("model_file", default="model.csv")
@click.option("-sep", "sep", default=",", help="csv separator")
@click.option("-p", "plot", is_flag=True, help="plot data")
@click.option("-e", "epochs", default=1, help="epochs to train")
@click.option("-b", "batch_size", default=1, help="batch size")
@click.option("-l", "learning_rate", default=0.01, help="learning rate")

def main(data_file, sep, plot, model_file, epochs, batch_size, learning_rate):
    trainer = Trainer(data_file, sep, plot, model_file, epochs, batch_size, learning_rate)
    trainer.train()

if __name__ == "__main__":
    main()