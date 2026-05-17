from transformers import pipeline
from PIL import Image
import requests
import io

print("Loading Falconsai NSFW model — downloading once, cached forever...")

classifier = pipeline(
    "image-classification",
    model="Falconsai/nsfw_image_detection"
)

print("Model loaded successfully!")

# Test with a sample image
test_url = "https://fastly.picsum.photos/id/582/400/300.jpg?hmac=oXNs7zGVc4HdLj2zG5ZPssX0iCSdNfXxypkS0qdzax0"

response = requests.get(
    test_url,
    headers={"User-Agent": "Mozilla/5.0"},  # Some servers block requests without a UA
    timeout=10
)

# Check the response before trying to open it
print(f"HTTP status: {response.status_code}")
print(f"Content-Type: {response.headers.get('Content-Type', 'unknown')}")

if response.status_code != 200:
    raise Exception(f"Download failed with status {response.status_code}")

if "image" not in response.headers.get("Content-Type", ""):
    raise Exception(f"Response is not an image. Got: {response.headers.get('Content-Type')}")

img = Image.open(io.BytesIO(response.content)).convert("RGB")
result = classifier(img)
print(f"\nTest result: {result}")
print("\nFalconsai model ready to use.")