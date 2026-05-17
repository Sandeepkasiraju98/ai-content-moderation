import sys
sys.path.append('..')
import mlflow
from src.detection.vision_detector import ResNetImageDetector

mlflow.set_experiment("vision-nsfw-detection")

detector = ResNetImageDetector()
auc = detector.train(
    dataset_path="data/raw/images/nsfw_dataset",
    epochs=5
)
detector.save("models/")

print(f"\nTraining complete. Final AUC: {auc:.4f}")