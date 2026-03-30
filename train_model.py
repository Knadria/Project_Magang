import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    print("Loading dataset...")
    df = pd.read_csv('dataset_untuk_training.csv')
    
    # Feature Selection
    # Drop columns that are IDs, Names, or targets/leakage (e.g., ratings collected after study)
    target = 'efficiency_score'
    drop_columns = [
        'student_id', 'student_name', 'overall_satisfaction',
        'campus_facilities_rating', 'teaching_quality_rating',
        target
    ]
    
    # Optional: Fill missing values for numericals, though CatBoost can handle them in some ways.
    # We will let CatBoost handle NaNs in categorical features, but let's do a basic fill for safety.
    df = df.fillna(0)
    
    X = df.drop(columns=drop_columns)
    y = df[target]
    
    # Identify Categorical Features
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print("Categorical features detected:", cat_features)
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training CatBoost Classifier...")
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
        verbose=50,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    
    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Saving the model...")
    joblib.dump(model, 'catboost_model.pkl')
    print("Model saved to 'catboost_model.pkl'.")

if __name__ == '__main__':
    main()
