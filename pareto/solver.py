"""
Computes weighted loss based on preference ray
"""
class LinearScalarizationSolver():

    def __init__(self):
        super().__init__()

    def get_weighted_loss(self, losses, ray):
        return (losses * ray).sum()

    def __call__(self, losses, ray):
        return self.get_weighted_loss(losses, ray)
