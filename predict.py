import os
import re
import click
import json

class Predictor(object):

    def __init__(self, model_file, sep, x):
        self.model_file = model_file
        self.sep = sep
        self.model = [0, 0]
        # Read model
        if len(self.model_file):
            self.read_model()
        y = self.predict(int(x))
        print("Preditction is : %d" % y)

    
    def read_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, "r") as f:
                for line in f:
                    data = json.load(line)
                    self.model["theta_0"] = data["theta_0"]
                    self.model["theta_1"] = data["theta_1"]
                    self.model["x_min"] = data["x_min"]
                    self.model["x_max"] = data["x_max"]
                    self.model["y_min"] = data["y_min"]
                    self.model["y_max"] = data["y_max"]
    
    def predict(self, x):
        y = self.model[0] + self.model[1] * x
        return y

@click.command()
@click.argument("model_file", default="model.csv")
@click.option("-sep", "sep", default=",", help="csv separator")
def main(model_file, sep):
    x = click.prompt("Please type X value (int)", type=int)
    Predictor(model_file, sep, x)

if __name__ == "__main__":
    main()