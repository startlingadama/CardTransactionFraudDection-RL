# Credit Card Fraud Detection with Reinforcement Learning

## Description

This project uses Reinforcement Learning (RL) to detect fraudulent credit card transactions. The goal is to train an RL agent that can classify transactions as fraudulent or non-fraudulent based on a dataset of credit card transactions.

## Dataset

The dataset contains credit card transactions from European cardholders. It includes:
- 284,807 transactions,
- 492 frauds (0.172% of total transactions),
- Features V1 to V28 are PCA-transformed,
- `Time` and `Amount` are raw features.

The dataset is highly imbalanced, with very few fraudulent transactions.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/startlingadama/CardTransactionFraudDetection-RL.git
cd CardTransactionFraudDetection-RL
python3 -m venv venv
./venv/Scripts/activate # Windows

pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook
```


## License

This project is licensed under the MIT License.

## Acknowledgments

* Credit card fraud detection dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Developer:
 - **Adama Coulibaly**: AI/ML Engineer

