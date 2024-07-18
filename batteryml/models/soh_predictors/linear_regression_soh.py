from batteryml.builders import MODELS
from batteryml.models.sklearn_model import SklearnModel
from sklearn.linear_model import LinearRegression

@MODELS.register()
class LinearRegressionSOHPredictor(SklearnModel):
    def __init__(self, *args, workspace: str = None, **kwargs):
        super().__init__(workspace)
        self.model = LinearRegression(*args, **kwargs)
