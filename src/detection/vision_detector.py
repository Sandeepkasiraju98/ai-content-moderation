import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import open_clip
import numpy as np
import mlflow
import mlflow.pytorch
import pickle
import os
import io
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from sklearn.metrics import classification_report, roc_auc_score


# ── 1. CLIP Zero-Shot Detector (no training needed, works immediately) ──

class CLIPImageDetector:
    """
    Uses CLIP to classify images as safe or unsafe
    via natural language prompts. Zero training needed.
    """
    def __init__(self):
        print("Loading CLIP model...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai'
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model.eval()

        # FIX 1 — Added chart/document prompts to balance against 5 harmful prompts
        self.harmful_prompts = [
            "explicit sexual content",
            "nudity",
            "graphic violence",
            "offensive imagery",
            "hate symbols"
        ]
        self.safe_prompts = [
            "safe and appropriate image",
            "normal everyday photo",
            "family friendly content",
            "data visualization or chart",      # ← NEW: handles pairplots, graphs
            "scientific diagram or graph",      # ← NEW: handles research images
            "document or screenshot"            # ← NEW: handles UI/text screenshots
        ]
        print("CLIP model loaded.")

    def predict(self, image_input) -> dict:
        """
        image_input: file path (str), PIL Image, or bytes
        """
        # Load image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input type")

        # Preprocess
        image_tensor = self.preprocess(image).unsqueeze(0)

        all_prompts = self.harmful_prompts + self.safe_prompts
        text_tokens = self.tokenizer(all_prompts)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).squeeze(0)
            probs = similarities.softmax(dim=-1).numpy()

        # Harmful score = sum of harmful prompt probabilities
        n_harmful = len(self.harmful_prompts)
        harmful_score = float(probs[:n_harmful].sum())
        safe_score = float(probs[n_harmful:].sum())

        # Map prompts to their scores
        prompt_scores = {
            prompt: round(float(prob), 4)
            for prompt, prob in zip(all_prompts, probs)
        }

        # FIX 2 — Low confidence guard: if scores are nearly uniform,
        # the model has no idea what the image is → default to safe
        score_std = float(np.std(probs))
        if score_std < 0.015:
            return {
                "harmful_score": 0.0,
                "safe_score": 1.0,
                "risk_score": 0.0,
                "prompt_scores": prompt_scores,
                "label": "safe",
                "confidence": "low — defaulted to safe",
                "note": "Model was uncertain. Likely a chart, diagram, or uncommon image type."
            }

        return {
            "harmful_score": round(harmful_score, 4),
            "safe_score": round(safe_score, 4),
            "risk_score": round(harmful_score, 4),
            "prompt_scores": prompt_scores,
            "label": "harmful" if harmful_score > 0.4 else "safe",
            "confidence": "high"
        }

    def get_risk_score(self, image_input) -> float:
        return self.predict(image_input)["risk_score"]


# ── 2. Fine-tunable ResNet Classifier ──

class NSFWDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image'].convert("RGB")
        label = 1 if item['label'] == 'nsfw' else 0

        if self.transform:
            image = self.transform(image)

        return image, label


class ResNetImageDetector:
    """
    Fine-tuned ResNet18 for binary safe/unsafe classification.
    Trainable on your own labeled image data.
    """
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        self.transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.transform_eval = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.model = None
        self.is_trained = False

    def build_model(self):
        # Load pretrained ResNet18, replace final layer
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 2)
        )
        return model.to(self.device)

    def train(self, dataset_path: str, epochs: int = 5):
        print("Loading dataset...")
        dataset = load_from_disk(dataset_path)

        # Split 80/20
        split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = NSFWDataset(split['train'], self.transform_train)
        test_dataset = NSFWDataset(split['test'], self.transform_eval)

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=2
        )

        self.model = self.build_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.5
        )

        with mlflow.start_run(run_name="resnet18_nsfw_classifier"):
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", 32)
            mlflow.log_param("learning_rate", 1e-4)
            mlflow.log_param("model", "resnet18")

            for epoch in range(epochs):
                # Training
                self.model.train()
                total_loss = 0
                for images, labels in train_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                scheduler.step()

                # Validation
                self.model.eval()
                all_preds, all_probs, all_labels = [], [], []
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(self.device)
                        outputs = self.model(images)
                        probs = torch.softmax(outputs, dim=1)[:, 1]
                        preds = outputs.argmax(dim=1)

                        all_preds.extend(preds.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
                        all_labels.extend(labels.numpy())

                auc = roc_auc_score(all_labels, all_probs)
                report = classification_report(
                    all_labels, all_preds,
                    target_names=['Safe', 'NSFW'],
                    output_dict=True
                )

                mlflow.log_metric("loss", avg_loss, step=epoch)
                mlflow.log_metric("roc_auc", auc, step=epoch)
                mlflow.log_metric(
                    "f1_nsfw", report['NSFW']['f1-score'], step=epoch
                )

                print(f"Epoch {epoch+1}/{epochs} "
                      f"| Loss: {avg_loss:.4f} "
                      f"| AUC: {auc:.4f}")

            print("\n" + classification_report(
                all_labels, all_preds, target_names=['Safe', 'NSFW']
            ))

        self.is_trained = True
        return auc

    def save(self, path: str = "models/"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}resnet_nsfw.pth")
        print(f"Vision model saved to {path}")

    def load(self, path: str = "models/"):
        self.model = self.build_model()
        self.model.load_state_dict(
            torch.load(f"{path}resnet_nsfw.pth",
                      map_location=self.device)
        )
        self.model.eval()
        self.is_trained = True
        print("Vision model loaded.")

    def predict(self, image_input) -> dict:
        if not self.is_trained:
            raise Exception("Model not trained. Call train() or load() first.")

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input type")

        tensor = self.transform_eval(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.softmax(output, dim=1)[0][1].item()

        return {
            "risk_score": round(prob, 4),
            "label": "nsfw" if prob > 0.5 else "safe",
            "confidence": round(prob if prob > 0.5 else 1 - prob, 4)
        }