import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append('..')
from src.detection.nlp_detector import TrainableToxicityDetector
import mlflow

mlflow.set_experiment("nlp-toxicity-detection")

detector = TrainableToxicityDetector()
auc = detector.train("C:\\Users\\sande\\OneDrive\\Documents\\content-moderation-ai\\data\\raw\\train.csv")
detector.save("C:\\Users\\sande\\OneDrive\\Documents\\content-moderation-ai\\models\\")

print(f"\nTraining complete. AUC: {auc:.4f}")
print("Model saved to models/")