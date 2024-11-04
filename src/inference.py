import joblib
import pandas as pd

# Load the model from the saved pickle file
model = joblib.load(r"notebooks\mlruns\322751337751950671\d2fb042a3fb641878f9d62a86a2e326c\artifacts\model\model.pkl")

def inference_batch(data):
    """
    Perform batch inference using the LightGBM model and return the DataFrame with added columns.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the required columns.

    Returns:
    - pd.DataFrame: Original DataFrame with added "fraud_score" and "Risk Level" columns.
    """
    
    # Columns expected by the model
    model_features = [
        "step", "amount", "balanceChangeOrig", "balanceChangeDest",
        "amountToOldBalanceOrigRatio", "amountToOldBalanceDestRatio",
        "transactionCountOrig", "transactionCountDest", "type_CASH_OUT",
        "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
    ]
    
    # Ensure only the model's expected columns are present
    data = data[model_features]
    
    # Perform prediction (fraud score)
    data["fraud_score"] = model.predict_proba(data)[:, 1]  # Assuming column 1 is the probability of fraud

    # Determine risk level based on fraud score
    data["Risk Level"] = data["fraud_score"].apply(lambda x: "High" if x > 0.7 else "Medium" if x > 0.4 else "Low")

    return data.drop(columns=["step"], axis=1)


# if __name__ == "__main__":
#     # Load the data from the CSV file in chunks
#     chunks = pd.read_csv(r"C:\Users\hapijul_h\spaces\triglens\data\raw\PS_log.csv", chunksize=10000)

#     # Process each chunk and concatenate results
#     processed_chunks = []
#     for chunk in chunks:
#         processed_data = feature_engineering(chunk)
#         processed_chunk = inference_batch(processed_data)
#         processed_chunks.append(processed_chunk)

#     # Concatenate all processed chunks into a single DataFrame
#     data_with_predictions = pd.concat(processed_chunks, ignore_index=True)

#     # Display the DataFrame with added columns
#     print(data_with_predictions)
#     print(data_with_predictions.columns)
