"""
Generates a synthetic customer-support-ticket dataset for the text
classification demo.

Why synthetic data? It keeps the repo self-contained (no external
downloads / API keys needed to run the pipeline) while still producing
realistic, lexically varied text. Swap `load_raw_data()` in src/data.py
for a real source (a DB query, S3 file, Kaggle dataset, etc.) when you
move this to a real product.

Run:
    python src/generate_data.py
Produces:
    data/tickets_train.csv
    data/tickets_reference.csv   (baseline distribution — used by Evidently)
    data/tickets_current.csv     (drifted distribution — simulates 30 days later)
"""
import csv
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

LABELS = ["billing", "technical", "account", "general"]

TEMPLATES = {
    "billing": [
        "I was charged twice for my {plan} subscription this month.",
        "Can you explain the extra ${amount} fee on my latest invoice?",
        "My payment failed but the app still shows I owe ${amount}.",
        "I need a refund for the {plan} plan I cancelled last week.",
        "Why did my subscription price change from ${amount} to ${amount2}?",
        "The invoice for {month} doesn't match what I was quoted.",
        "I want to downgrade from {plan} to save money.",
        "Please update my card on file, the current one expired.",
        "I was billed after cancelling my {plan} subscription.",
        "Is there a discount available for annual {plan} billing?",
    ],
    "technical": [
        "The app crashes every time I try to open the {feature} screen.",
        "I'm getting a {error} error when I try to log in.",
        "Uploads to {feature} keep failing at around 80 percent.",
        "The {feature} page has been loading forever, it never finishes.",
        "Push notifications for {feature} stopped working after the update.",
        "I can't sync my data between devices, {feature} shows old info.",
        "The API returns a {error} when I call the {feature} endpoint.",
        "Dark mode breaks the layout on the {feature} dashboard.",
        "The mobile app freezes on the {feature} tab on Android.",
        "Search in {feature} returns no results even for exact matches.",
    ],
    "account": [
        "I forgot my password and the reset email never arrives.",
        "Can you merge my two accounts under one email address?",
        "I need to change the email linked to my account.",
        "My account got locked after too many login attempts.",
        "How do I enable two-factor authentication on my account?",
        "I want to delete my account and all associated data.",
        "Someone else may have accessed my account without permission.",
        "I can't update my profile picture, it keeps failing to upload.",
        "Please transfer ownership of my workspace to a teammate.",
        "My username shows up wrong on the {feature} page.",
    ],
    "general": [
        "Do you have a roadmap for upcoming {feature} improvements?",
        "Just wanted to say the new {feature} update looks great!",
        "What are your support hours for the {plan} plan?",
        "Is there a community forum where I can ask other users questions?",
        "Can you point me to documentation for the {feature} API?",
        "I'd like to suggest an improvement to the {feature} workflow.",
        "How do I provide feedback about the {plan} pricing tiers?",
        "Are there any webinars coming up about {feature}?",
        "What's the difference between the {plan} and enterprise plans?",
        "Thanks for the quick help yesterday, closing this ticket.",
    ],
}

FILL = {
    "plan": ["Pro", "Starter", "Team", "Enterprise", "Basic"],
    "amount": ["9.99", "19.99", "29.99", "49.00", "12.50"],
    "amount2": ["14.99", "24.99", "39.99", "59.00", "17.50"],
    "month": ["March", "April", "May", "June", "last month"],
    "feature": ["billing", "dashboard", "reports", "integrations", "settings",
                "notifications", "analytics", "export", "calendar", "messaging"],
    "error": ["500 Internal Server", "403 Forbidden", "timeout", "null pointer", "404 Not Found"],
}


def _fill(template: str) -> str:
    out = template
    for key, choices in FILL.items():
        if "{" + key + "}" in out:
            out = out.replace("{" + key + "}", random.choice(choices))
    return out


def _make_rows(n_per_class: int, label_weights=None):
    rows = []
    weights = label_weights or {label: 1.0 for label in LABELS}
    for label in LABELS:
        count = int(n_per_class * weights[label])
        templates = TEMPLATES[label]
        for _ in range(count):
            text = _fill(random.choice(templates))
            rows.append({"text": text, "label": label})
    random.shuffle(rows)
    return rows


def main():
    # --- Training set: balanced across classes ---
    train_rows = _make_rows(n_per_class=150)
    with open(DATA_DIR / "tickets_train.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        w.writerows(train_rows)

    # --- Reference set for drift monitoring: same distribution as training ---
    reference_rows = _make_rows(n_per_class=80)
    with open(DATA_DIR / "tickets_reference.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        w.writerows(reference_rows)

    # --- "Current" set: simulates 30 days later with shifted class balance
    #     (more technical tickets after a bad release) + new vocabulary,
    #     which is exactly what the drift monitor should flag. ---
    drifted_weights = {"billing": 0.6, "technical": 2.2, "account": 0.9, "general": 0.5}
    current_rows = _make_rows(n_per_class=80, label_weights=drifted_weights)
    # inject some out-of-vocabulary phrasing to simulate a real product change
    extra_phrases = [
        "The new checkout flow keeps throwing a payment gateway timeout error.",
        "Since the redesign, the {feature} tab is completely unresponsive on iOS 18.",
        "Your new AI assistant feature gave me an incorrect answer twice today.",
    ]
    for phrase in extra_phrases * 15:
        current_rows.append({"text": _fill(phrase), "label": "technical"})
    random.shuffle(current_rows)
    with open(DATA_DIR / "tickets_current.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        w.writerows(current_rows)

    print(f"Wrote {len(train_rows)} training rows -> data/tickets_train.csv")
    print(f"Wrote {len(reference_rows)} reference rows -> data/tickets_reference.csv")
    print(f"Wrote {len(current_rows)} current rows -> data/tickets_current.csv")


if __name__ == "__main__":
    main()
