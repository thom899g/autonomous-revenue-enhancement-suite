import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class RevenueOptimizer:
    """
    Implements machine learning models for revenue optimization.
    Handles edge cases such as missing data and categorical variables.
    """

    def __init__(self):
        self.models = {
            'price_prediction': None,
            'customer_segmentation': None
        }

    def build_pipeline(self, model_type: str) -> Pipeline:
        """
        Builds a machine learning pipeline based on the model type.
        
        Args:
            model_type: Type of model to build
            
        Returns:
            Machine learning pipeline
        """
        if model_type not in ['price_prediction', 'customer_segmentation']:
            raise ValueError("Invalid model type")

        if model_type == 'price_prediction':
            numeric_features = ['Cost', 'Quantity']
            categorical_features = ['Product_Category']

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(), categorical_features)
                ]
            )

            from sklearn.linear_model import LinearRegression
            model = LinearRegression()

        elif model_type == 'customer_segmentation':
            numeric_features = ['Revenue', 'Frequency']
            categorical_features = []

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features)
                ]
            )

            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=5)

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        return pipeline

    def train_model(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains a machine learning model.
        
        Args:
            model_type: Type of model to train
            X: Features DataFrame
            y: Target Series
            
        Returns:
            None
        """
        try:
            pipeline = self.build_pipeline(model_type)
            pipeline.fit(X, y)
            self.models[model_type] = pipeline
        except Exception as e:
            logging.error(f"Training failed for {model_type}: {str(e)}")

    def predict(self, model_type: str, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained model.
        
        Args:
            model_type: Type of model to use
            X: Features DataFrame
            
        Returns:
            Predictions as a numpy array
        """
        try:
            if self.models[model_type] is None:
                raise ValueError(f"Model {model_type} not trained")
            return self.models[model_type].predict(X)
        except Exception as e:
            logging.error(f"Inference failed for {model_type}: {str(e)}")