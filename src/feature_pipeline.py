import pandas as pd

def feature_engineering(data):
    """
    Perform feature engineering for fraud detection on a transaction dataset.
    
    Parameters:
    - data (pd.DataFrame): Input data with transaction details.
    
    Returns:
    - pd.DataFrame: Transformed data with engineered features.
    """
    # Create new features to capture balance changes
    data['balanceChangeOrig'] = data['newbalanceOrig'] - data['oldbalanceOrg']
    data['balanceChangeDest'] = data['newbalanceDest'] - data['oldbalanceDest']

    # Calculating the amount-to-balance ratio for origin and destination accounts
    epsilon = 1e-9
    data['amountToOldBalanceOrigRatio'] = data['amount'] / (data['oldbalanceOrg'] + epsilon)
    data['amountToOldBalanceDestRatio'] = data['amount'] / (data['oldbalanceDest'] + epsilon)

    # Drop redundant balance columns
    data = data.drop(['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], axis=1)

    # Transaction frequency: Count transactions per customer
    data['transactionCountOrig'] = data.groupby('nameOrig')['nameOrig'].transform('count')
    data['transactionCountDest'] = data.groupby('nameDest')['nameDest'].transform('count')

    # Drop unnecessary columns and get dummy variables for transaction type
    data = data.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    data = pd.get_dummies(data, columns=['type'], drop_first=True)
    
    return data

# Example usage:
# processed_data = feature_engineering(data)
