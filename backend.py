import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -----------------------------
# Paths and global resources
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_REPO = "weihan4068/project_5240_ticket_classifier"
LABEL2ID_PATH = BASE_DIR / "label2id.json"
ID2LABEL_PATH = BASE_DIR / "id2label.json"

# -----------------------------
# Load label mappings
# -----------------------------
with open(LABEL2ID_PATH, "r", encoding="utf-8") as f:
    label2id = json.load(f)

with open(ID2LABEL_PATH, "r", encoding="utf-8") as f:
    id2label = json.load(f)

id2label = {int(k): v for k, v in id2label.items()}

# -----------------------------
# Load model and tokenizer
# -----------------------------
clf_tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
clf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
clf_model.eval()

# -----------------------------
# Classification functions
# -----------------------------
def predict_category_with_scores(text: str, max_length: int = 128, top_k: int = 5):
    inputs = clf_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    with torch.no_grad():
        outputs = clf_model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)[0]
    top_probs, top_indices = torch.topk(probs, k=top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append(
            {
                "label_id": idx.item(),
                "category": id2label[idx.item()],
                "probability": prob.item(),
            }
        )

    return results


def predict_category(text: str, max_length: int = 128) -> str:
    results = predict_category_with_scores(text, max_length=max_length, top_k=1)
    return results[0]["category"]


def predict_top_categories(text: str, max_length: int = 128, top_k: int = 3):
    return predict_category_with_scores(text, max_length=max_length, top_k=top_k)

# -----------------------------
# Lightweight business rule adjustment
# -----------------------------
def adjust_category_by_keywords(text: str, top_categories):
    """
    Light post-processing step to reduce obvious misclassification on more natural expressions.
    Keeps the original top-1 unless there is a strong keyword signal and a relevant class
    already appears in top-k candidates.
    """
    text_lower = text.lower()
    top_category = top_categories[0]["category"]

    shipping_keywords = [
        "package", "parcel", "tracking", "shipment", "shipping",
        "delivery", "arrived", "deliver", "shipped", "address"
    ]
    refund_keywords = [
        "refund", "reimbursement", "reimburse", "money back", "return"
    ]
    payment_keywords = [
        "payment", "charged", "charge", "billing", "paid", "card"
    ]
    account_keywords = [
        "account", "login", "log in", "password", "user", "profile"
    ]

    def find_candidate(preferred_categories):
        for item in top_categories:
            if item["category"] in preferred_categories:
                return item["category"]
        return None

    # Strong keyword-based correction for common business intents
    if any(k in text_lower for k in shipping_keywords):
        candidate = find_candidate(["SHIPPING", "DELIVERY"])
        if candidate:
            return candidate

    if any(k in text_lower for k in refund_keywords):
        candidate = find_candidate(["REFUND"])
        if candidate:
            return candidate

    if any(k in text_lower for k in payment_keywords):
        candidate = find_candidate(["PAYMENT"])
        if candidate:
            return candidate

    # If top-1 is SHIPPING/DELIVERY but text is strongly account related,
    # allow correction back to ACCOUNT if ACCOUNT is already in top-k.
    if top_category in ["SHIPPING", "DELIVERY"]:
        if any(k in text_lower for k in account_keywords):
            candidate = find_candidate(["ACCOUNT"])
            if candidate:
                return candidate

    return top_category

# -----------------------------
# Support note generation
# -----------------------------
def generate_support_note(text: str, category: str) -> str:
    text_lower = text.lower()

    if category in ["DELIVERY", "SHIPPING"]:
        if "tracking" in text_lower and ("not updated" in text_lower or "hasn't updated" in text_lower):
            return "Customer reports shipment delay and outdated tracking information."
        elif "not arrived" in text_lower or "hasn't arrived" in text_lower or "delayed" in text_lower:
            return "Customer reports delayed delivery and requests shipment status update."
        elif "address" in text_lower:
            return "Customer requests help with shipping or delivery address information."
        else:
            return "Customer has a shipping or delivery-related request."

    elif category == "REFUND":
        if "status" in text_lower or "reimbursement" in text_lower:
            return "Customer asks about refund eligibility or reimbursement status."
        else:
            return "Customer requests refund-related assistance."

    elif category == "PAYMENT":
        if "charged" in text_lower or "payment" in text_lower or "billing" in text_lower:
            return "Customer reports a payment-related issue and requests assistance."
        else:
            return "Customer has a payment-related request."

    elif category == "ACCOUNT":
        if "login" in text_lower or "password" in text_lower:
            return "Customer reports an account access issue."
        elif "account" in text_lower or "user" in text_lower or "profile" in text_lower:
            return "Customer requests help with account information or settings."
        else:
            return "Customer has an account-related request."

    elif category == "ORDER":
        if "cancel" in text_lower:
            return "Customer requests order cancellation support."
        elif "order" in text_lower:
            return "Customer asks for help with order details or order status."
        else:
            return "Customer has an order-related request."

    elif category == "INVOICE":
        return "Customer requests invoice-related support."

    elif category == "CONTACT":
        return "Customer asks how to contact customer support."

    elif category == "CANCEL":
        return "Customer wants to cancel an order or subscription."

    elif category == "FEEDBACK":
        return "Customer provides feedback or comments about the service."

    elif category == "SUBSCRIPTION":
        return "Customer requests support regarding subscription settings or plans."

    else:
        return "Customer submitted a support request that requires further review."

# -----------------------------
# Reply template generation
# -----------------------------
def get_reply_template(category: str) -> str:
    templates = {
        "DELIVERY": "Thank you for contacting our support team. We are checking your delivery status and will update you as soon as possible.",
        "SHIPPING": "Thank you for contacting our support team. We will help you with your shipping-related request shortly.",
        "REFUND": "Thank you for contacting our support team. We are reviewing your refund-related request and will get back to you soon.",
        "PAYMENT": "Thank you for contacting our support team. We are checking your payment issue and will assist you as soon as possible.",
        "ACCOUNT": "Thank you for contacting our support team. We are reviewing your account-related issue and will assist you shortly.",
        "ORDER": "Thank you for contacting our support team. We are checking your order-related request and will respond shortly.",
        "INVOICE": "Thank you for contacting our support team. We will help you with your invoice-related request.",
        "CONTACT": "Thank you for contacting our support team. Our support team will assist you with your contact request.",
        "CANCEL": "Thank you for contacting our support team. We are reviewing your cancellation request.",
        "FEEDBACK": "Thank you for your feedback. We appreciate your comments and will review them carefully.",
        "SUBSCRIPTION": "Thank you for contacting our support team. We are reviewing your subscription-related request.",
    }

    return templates.get(
        category,
        "Thank you for contacting our support team. We will review your request and get back to you soon.",
    )

# -----------------------------
# End-to-end pipeline
# -----------------------------
def run_pipeline(text: str):
    top_categories = predict_top_categories(text, top_k=3)
    final_category = adjust_category_by_keywords(text, top_categories)

    support_note = generate_support_note(text, final_category)
    reply_template = get_reply_template(final_category)

    return {
        "top_category": final_category,
        "top_categories": top_categories,
        "support_note": support_note,
        "reply_template": reply_template,
    }
