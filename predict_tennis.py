import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
import numpy as np

class TennisPredictor:
    """
    A class to predict whether to play tennis based on weather conditions.
    Uses a pre-trained decision tree model or creates a new one if needed.
    """
    
    def __init__(self):
        """Initialize the predictor with model and encoders."""
        self.model = None
        self.feature_encoders = {}
        self.target_encoder = LabelEncoder()
        
        # Try to load a pre-trained model, or train a new one if not found
        if os.path.exists('tennis_model.joblib') and os.path.exists('encoders.joblib'):
            self.model = joblib.load('tennis_model.joblib')
            self.feature_encoders, self.target_encoder = joblib.load('encoders.joblib')
            print("Loaded pre-trained model and encoders.")
        else:
            print("No pre-trained model found. Training a new model...")
            self._train_model()
    
    def _train_model(self):
        """Train a new model using the play_tennis.csv dataset."""
        try:
            # Load the data
            data = pd.read_csv('play_tennis.csv')
            
            # Display dataset statistics
            print("\nDataset Information:")
            print(f"Total examples: {len(data)}")
            print(f"Play tennis (Yes): {len(data[data['play'] == 'Yes'])}")
            print(f"Play tennis (No): {len(data[data['play'] == 'No'])}")
            
            # Prepare features (X) and target (y)
            X = data.drop(['play', 'day'], axis=1)
            y = data['play']
            
            # Encode categorical features
            X_encoded = X.copy()
            for column in X.columns:
                encoder = LabelEncoder()
                X_encoded[column] = encoder.fit_transform(X[column])
                self.feature_encoders[column] = encoder
            
            # Encode target variable
            y_encoded = self.target_encoder.fit_transform(y)
            
            # Display encoded values for reference
            print("\nEncoding Reference:")
            for column in X.columns:
                encoder = self.feature_encoders[column]
                print(f"{column}: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
            print(f"Target (play): {dict(zip(self.target_encoder.classes_, self.target_encoder.transform(self.target_encoder.classes_)))}")
            
            # Train the model
            self.model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
            self.model.fit(X_encoded, y_encoded)
            
            # Display feature importance
            print("\nFeature Importance:")
            for feature, importance in zip(X.columns, self.model.feature_importances_):
                print(f"{feature}: {importance:.4f}")
            
            # Save the model for future use
            joblib.dump(self.model, 'tennis_model.joblib')
            joblib.dump((self.feature_encoders, self.target_encoder), 'encoders.joblib')
            
            print("\nModel trained successfully!")
            
        except Exception as e:
            print(f"Error training model: {e}")
    
    def predict(self, outlook, temp, humidity, wind):
        """
        Predict whether to play tennis based on weather conditions.
        
        Parameters:
        outlook (str): 'Sunny', 'Overcast', or 'Rain'
        temp (str): 'Hot', 'Mild', or 'Cool'
        humidity (str): 'High' or 'Normal'
        wind (str): 'Weak' or 'Strong'
        
        Returns:
        str: 'Yes' or 'No'
        """
        if self.model is None:
            print("Error: No model available for prediction.")
            return None
        
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'outlook': [outlook],
            'temp': [temp],
            'humidity': [humidity],
            'wind': [wind]
        })
        
        try:
            # Encode the input data using the SAME encoders from training
            input_encoded = pd.DataFrame()
            for column in input_data.columns:
                if column in self.feature_encoders:
                    encoder = self.feature_encoders[column]
                    if input_data[column][0] in encoder.classes_:
                        input_encoded[column] = encoder.transform([input_data[column][0]])
                    else:
                        print(f"Warning: Unknown value '{input_data[column][0]}' for feature '{column}'")
                        # Handle unknown value by using the most common value
                        input_encoded[column] = [0]
            
            # Make prediction
            prediction = self.model.predict(input_encoded)
            
            # Print prediction probabilities to see the model's confidence
            proba = self.model.predict_proba(input_encoded)[0]
            classes = self.target_encoder.classes_
            print("\nPrediction probabilities:")
            for cls, prob in zip(classes, proba):
                print(f"  {cls}: {prob:.4f} ({prob*100:.1f}%)")
            
            # Convert numeric prediction back to original label
            result = self.target_encoder.inverse_transform([prediction[0]])[0]
            return result
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None


def explain_decision_path(model, input_encoded, feature_names, target_encoder):
    """Explain the decision path the model took to reach its prediction."""
    # Get the decision path
    node_indicator = model.decision_path(input_encoded)
    leaf_id = model.apply(input_encoded)
    
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    
    # Traverse the decision path for the first sample
    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]
    
    print("\nDecision path:")
    for node_id in node_index:
        # If not a leaf node
        if (leaf_id[sample_id] != node_id):
            # Get the feature used for the split
            feature_id = feature[node_id]
            if feature_id != -2:  # -2 indicates a non-split node
                feature_name = feature_names[feature_id]
                # Get the threshold value
                threshold_value = threshold[node_id]
                # Get the actual value for this feature
                feature_value = input_encoded.iloc[sample_id, feature_id]
                
                # Format the comparison
                if feature_value <= threshold_value:
                    comparison = "≤"
                else:
                    comparison = ">"
                
                print(f"  Decision: Is {feature_name} {comparison} {threshold_value}? Answer: Yes")
        else:
            # This is a leaf node - get the prediction
            value = model.tree_.value[node_id][0]
            class_prediction = np.argmax(value)
            predicted_class = target_encoder.inverse_transform([class_prediction])[0]
            confidence = (value[class_prediction] / np.sum(value)) * 100
            print(f"  Final prediction: {predicted_class} (Confidence: {confidence[0]:.1f}%)")


def main():
    """Main function to run the command-line interface."""
    print("\n===== Tennis Decision Maker =====\n")
    
    predictor = TennisPredictor()
    
    while True:
        print("\nEnter weather conditions to get a prediction:")
        
        # Get user input
        outlook = input("What's the outlook? (Sunny/Overcast/Rain): ").capitalize()
        if outlook not in ['Sunny', 'Overcast', 'Rain']:
            print("Invalid input. Please enter Sunny, Overcast, or Rain.")
            continue
            
        temp = input("What's the temperature? (Hot/Mild/Cool): ").capitalize()
        if temp not in ['Hot', 'Mild', 'Cool']:
            print("Invalid input. Please enter Hot, Mild, or Cool.")
            continue
            
        humidity = input("How's the humidity? (High/Normal): ").capitalize()
        if humidity not in ['High', 'Normal']:
            print("Invalid input. Please enter High or Normal.")
            continue
            
        wind = input("How's the wind? (Weak/Strong): ").capitalize()
        if wind not in ['Weak', 'Strong']:
            print("Invalid input. Please enter Weak or Strong.")
            continue
        
        # Display the input
        print(f"\nInput conditions: Outlook={outlook}, Temperature={temp}, Humidity={humidity}, Wind={wind}")
        
        # Get prediction
        result = predictor.predict(outlook, temp, humidity, wind)
        
        # Display result
        if result:
            print(f"\nPrediction: Should you play tennis? {result}")
            
            # Create encoded input for explanation
            input_data = pd.DataFrame({
                'outlook': [outlook],
                'temp': [temp],
                'humidity': [humidity],
                'wind': [wind]
            })
            
            input_encoded = pd.DataFrame()
            for column in input_data.columns:
                if column in predictor.feature_encoders:
                    encoder = predictor.feature_encoders[column]
                    input_encoded[column] = encoder.transform([input_data[column][0]])
            
            # Explain the decision path
            #try:
                #explain_decision_path(predictor.model, input_encoded, list(input_encoded.columns), predictor.target_encoder)
            #except Exception as e:
                #print(f"Could not explain decision path: {e}")
        
        # Ask if user wants another prediction
        again = input("\nWould you like to make another prediction? (y/n): ").lower()
        if again != 'y':
            break
    
    print("\nThank you for using the Tennis Decision Maker!")


if __name__ == "__main__":
    main()