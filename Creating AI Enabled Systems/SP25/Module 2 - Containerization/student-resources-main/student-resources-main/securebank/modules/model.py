
import random
import pandas as pd


class Model():
    """
    You will need to implement your own Model class.
    """

    def __init__(self, model_type="random"):
        self.model_type = model_type

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        return random.randint(0, 1)
    

if __name__ == "__main__":
    model = Model(model_type="random")
    print(model.predict())

