import requests

BASE = "http://127.0.0.1:8000"

# ── Text tests ──
def test_text(text, user_id, expected_action=None):
    response = requests.post(f"{BASE}/moderate/text", json={
        "text": text,
        "user_id": user_id
    })
    result = response.json()
    status = "✅" if (expected_action is None or result["action"] == expected_action) else "❌"
    print(f"{status} [{result['action']}] score={result['risk_score']} | {text[:60]}")
    return result

# ── Image tests ──
def test_image(image_path, user_id, expected_action=None):
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{BASE}/moderate/image",
            params={"user_id": user_id},
            files={"file": f}
        )
    result = response.json()
    status = "✅" if (expected_action is None or result["action"] == expected_action) else "❌"
    print(f"{status} [{result['action']}] score={result['risk_score']} | {image_path}")
    return result

if __name__ == "__main__":
    print("\n── TEXT TESTS ──")
    test_text("Have a wonderful day!", "u001", "approved")
    test_text("I will kill you", "u002", "flag_for_review")
    test_text("You are so stupid and ugly", "u003", "flag_for_review")
    test_text("The weather is nice today", "u004", "approved")
    test_text("I hate all people from that country", "u005", "warn_user")

    print("\n── IMAGE TESTS ──")
    # Put a few test images in tests/images/ folder
    # test_image("tests/images/safe_photo.jpg", "u006", "approved")
    # test_image("tests/images/pairplot.png", "u007", "approved")

    print("\nDone.")