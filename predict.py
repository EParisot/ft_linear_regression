import os
import re
import click
import json

class Predictor(object):

    def __init__(self, model_file, x):
        self.model_file = model_file
        self.model = {}
        # Read model
        if len(self.model_file):
            self.read_model()
        y = self.predict(int(x))
        print("Preditction is : %d" % y)
            
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
                else:
                    print("Empty model")
                    exit(0)
        else:
            print("No model file")
            exit(0)
    
    def predict(self, x):
        # normalise x
        x -= self.model["x_min"]
        x /= self.model["x_max"]
        y = self.model["theta_0"] + self.model["theta_1"] * x
        # "de-normalise" y
        y *= self.model["y_max"]
        y += self.model["y_min"]
        return y

@click.command()
@click.argument("model_file", default="model.json")
def main(model_file):
    x = click.prompt("Please type X value (int)", type=int)
    Predictor(model_file, x)

if __name__ == "__main__":
    main()