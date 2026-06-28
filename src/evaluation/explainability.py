import torch
import numpy as np

class FeatureExplainer:
    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns
        
    def get_feature_importance(self, x_tensor):
        """
        Calculates feature importance using input gradients (Saliency).
        x_tensor: shape [1, sequence_length, num_features]
        """
        # Ensure x_tensor requires gradient
        x_tensor = x_tensor.clone().detach().requires_grad_(True)
        self.model.eval()
        
        # Forward pass
        preds = self.model(x_tensor) # [1, horizons]
        
        # We care about what is driving the 30-min horizon (horizon 0)
        # index: [batch=0, horizon=0]
        target_pred = preds[0, 0]
        
        self.model.zero_grad()
        target_pred.backward()
        
        # The gradient of the input tensor gives the sensitivity of the prediction to each input feature
        saliency = x_tensor.grad.abs().squeeze(0).cpu().numpy() # [sequence_length, num_features]
        
        # Aggregate importance over the time sequence (sum of absolute gradients across time)
        feature_importance = saliency.sum(axis=0)
        
        # Normalize to percentages
        if feature_importance.sum() > 0:
            feature_importance = feature_importance / feature_importance.sum() * 100
            
        # Map to feature names and sort
        importance_dict = {
            self.feature_columns[i]: float(feature_importance[i])
            for i in range(len(self.feature_columns))
        }
        
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        
        # Return top 3 drivers
        top_drivers = {k: round(v, 2) for k, v in list(sorted_importance.items())[:3]}
        return {
            "top_drivers": top_drivers,
            "method": "Input Gradients (Saliency) on 95th Percentile 30-min Forecast"
        }
