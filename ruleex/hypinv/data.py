from gtrain import Data
import numpy as np


class DataForBoundary(Data):
    """
    Data model that is applied in the gtrain training of the finding closest point on the boundary
    """
    def __init__(self):
        self.desired_output = np.float32([[0.5, 0.5]])

    def set_placeholders(self, pl_list):
        self.output = pl_list[0]

    def get_next_batch(self):
        return {self.output: self.desired_output}

    def accumulate_grad(self):
        return False

    def get_next_dev_batch(self):
        return {self.output: self.desired_output}

    def train_ended(self):
        pass
