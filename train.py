import os
import random
import json
import re
import time
import click
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from threading import Thread

class Trainer(object):

    def __init__(self, data_file, sep, plot, model_file, epochs, batch_size, learning_rate):
        self.data_file = data_file
        self.sep = sep
        self.plot = plot
        self.model_file = model_file
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model =   {"theta_0": 0.0, 
                        "theta_1": 0.0, 
                        "x_min": 0, 
                        "x_max": 0, 
                        "y_min": 0, 
                        "y_max": 0}
        self.x_data = []
        self.y_data = []
        self.labels = []
        self.acc = []
        self.loss = []
        # Read data
        self.read_data()
        if len(self.x_data) != len(self.y_data) or len(self.x_data) == 0:
            print("Error : no valid data found in %s" % self.data_file)
            return
        if batch_size > 0 and batch_size <= len(self.x_data):
            self.batch_size = batch_size
        else:
            self.batch_size = len(self.x_data)
        # Read model
        if len(self.model_file):
            self.read_model()
            

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
            self.normalise()

    def normalise(self):
        x_min = min(self.x_data)
        x_max = max(self.x_data) - x_min
        y_min = min(self.y_data)
        y_max = max(self.y_data) - y_min
        for i, _ in enumerate(self.x_data):
            self.x_data[i] -= x_min
            self.x_data[i] /= x_max
            self.y_data[i] -= y_min
            self.y_data[i] /= y_max
        self.model["x_min"] = x_min
        self.model["x_max"] = x_max
        self.model["y_min"] = y_min
        self.model["y_max"] = y_max
            
    def read_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, "r") as f:
                check = f.read(2)
                f.seek(0)
                if len(check) != 0 and check[0] != "\n" and check != "{}":
                    data = json.load(f)
                    self.model["theta_0"] = data["theta_0"]
                    self.model["theta_1"] = data["theta_1"]
                    self.model["x_min"] = data["x_min"]
                    self.model["x_max"] = data["x_max"]
                    self.model["y_min"] = data["y_min"]
                    self.model["y_max"] = data["y_max"]

    def save_model(self):
        if not os.path.exists(self.model_file):
            mode = "w+"
        else:
            mode = "w"
        with open(self.model_file, mode) as f:
            json.dump(self.model, f)

    def animate(self):
        plt.clf()
        x_data, y_data = [list(t) for t in zip(*sorted(zip(self.x_data, self.y_data)))]
        plt.scatter(x_data, y_data)
        if len(self.labels):
            plt.xlabel(self.labels[0])
            plt.ylabel(self.labels[1])
        # result
        x1 = min(x_data)
        y1 = self.estimate(x1)
        x2 = max(x_data)
        y2 = self.estimate(x2)
        plt.plot([x1, x2], [y1, y2], c="r")
        plt.draw()
        plt.pause(10/self.epochs)

    def train(self):
        theta_0 = 0.0
        theta_1 = 0.0
        # read model
        if self.model["theta_0"] != theta_0 or self.model["theta_1"] != theta_1:
            theta_0 = self.model["theta_0"]
            theta_1 = self.model["theta_1"]
        # process train
        self.train_loop()
        # write model file
        self.save_model()
        # plot result
        if self.plot:
            plt.figure("Train history")
            plt.plot(self.acc, label="acc")
            plt.plot(self.loss, label="loss")
            plt.legend()
            plt.show(block=True)

    def train_loop(self):
        # shuffle datas
        l = list(zip(self.x_data, self.y_data))
        random.shuffle(l)
        x_data, y_data = zip(*l)
        # loop on epochs / batches / data_points
        for epoch in range(self.epochs):
            print("Training... Epoch : %d" % (epoch + 1))
            for (batch_x, batch_y) in self.batches_generator(x_data, y_data):
                loss, acc = self.training_batch(batch_x, batch_y)
                self.acc.append(acc)
                self.loss.append(loss)
                # print
                print("loss : %f ; acc : %f" % (round(loss, 2), round(acc, 2)))
                if self.plot:
                    self.animate()
                
    def batches_generator(self, x_data, y_data):
        for i in range(len(x_data) // self.batch_size):
            start = i * self.batch_size
            end = i * self.batch_size + self.batch_size
            batch_x = x_data[start:end]
            batch_y = y_data[start:end]
            yield batch_x, batch_y

    def training_batch(self, batch_x, batch_y):
        n = float(len(batch_x))
        b_vect = []
        a_vect = []
        for i, _ in enumerate(batch_x):
            error_b = self.estimate(batch_x[i]) - batch_y[i]
            b_vect.append(error_b)
            error_a = error_b * batch_x[i]
            a_vect.append(error_a)
        loss_b_prime = sum(b_vect)
        loss_a_prime = sum(a_vect)

        # process train
        tmp_theta_0 = self.learning_rate * loss_b_prime / n
        tmp_theta_1 = self.learning_rate * loss_a_prime / n
        self.model["theta_0"] -= tmp_theta_0
        self.model["theta_1"] -= tmp_theta_1

        # metrics
        new_loss_tab = []
        acc_tab = []
        for i, _ in enumerate(batch_x):
            error = self.estimate(batch_x[i]) - batch_y[i]
            error_sq = error ** 2
            new_loss_tab.append(error_sq)
            acc_tab.append(1)
        new_loss = sum(new_loss_tab) / n
        acc = float(sum(acc_tab) / n)

        # adjust LR
        if len(self.loss) > 0:
            if new_loss >= self.loss[-1]:
                self.model["theta_0"] += self.learning_rate * tmp_theta_0 / n
                self.model["theta_1"] += self.learning_rate * tmp_theta_1 / n
                self.learning_rate *=  0.5
            else:
                self.learning_rate *= 1.05
             
        return new_loss, acc
    
    def estimate(self, x):
        y = self.model["theta_0"] + self.model["theta_1"] * x
        return y

def meanfilt(tab, winsize):
    res = []
    for chunk in range(len(tab)//winsize):
        start = chunk * winsize
        end = chunk * winsize + winsize 
        mean = sum(tab[start:end]) / winsize
        res += [mean] * winsize
    return res


def ft_abs(number):
    if number > 0:
        return(number)
    else:
        return(-number)

@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("model_file", default="model.json")
@click.option("-sep", "sep", default=",", help="csv separator")
@click.option("-p", "plot", is_flag=True, help="plot data")
@click.option("-e", "epochs", default=1, help="epochs to train")
@click.option("-b", "batch_size", default=0, help="batch size")
@click.option("-l", "learning_rate", default=0.1, help="learning rate")
def main(data_file, sep, plot, model_file, epochs, batch_size, learning_rate):
    trainer = Trainer(data_file, sep, plot, model_file, epochs, batch_size, learning_rate)
    if trainer.plot:
        plt.ion()
    trainer.train()

if __name__ == "__main__":
    main()