import os
import re
import click

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
                    line = line.replace('\n', '')
                    line_data = line.split(self.sep)
                    r = re.compile('-?\d+(\.\d+)?')
                    if len(line_data) == 2 and all([r.match(value) for value in line_data]):
                        self.model[0] = float(line_data[0])
                        self.model[1] = float(line_data[1])
    

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