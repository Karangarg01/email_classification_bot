import os
import pandas as pd

from classifier import (
    load_email_dataset,
    train_and_save_model,
    predict_category,
)
from utils import get_logger, validate_prediction

PROCESSED_DIR = "processed"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "classified_emails.xlsx")


def main():
    logger = get_logger()

    # Load dataset
    df = load_email_dataset()
    print("🔹 Dataset preview:")
    print(df.head())
    print("\n🔹 Label distribution:")
    print(df["label"].value_counts())

    # Train model and save it (you can skip this once model is trained)
    train_and_save_model()

    # Classify each email using the saved model
    records = []
    for _, row in df.iterrows():
        filename = row["filename"]
        text = row["text"]
        true_label = row["label"]

        predicted_label = predict_category(text)
        status, comment = validate_prediction(true_label, predicted_label)

        logger.info(f"{filename} - {status} - {comment}")

        records.append({
            "Filename": filename,
            "True Label": true_label,
            "Predicted Label": predicted_label,
            "Status": status,
            "Comments": comment,
            "Text": text,
        })

    # Save results to Excel
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    result_df = pd.DataFrame(records)
    result_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Classified email report saved to: {OUTPUT_FILE}")
    print("Email classification pipeline finished!")


if __name__ == "__main__":
    main()
