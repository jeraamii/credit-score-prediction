"""
src/pipeline/aws_pipeline.py
Cloud-based ML Pipeline — AWS SageMaker + S3

Prerequisites:
    pip install boto3 sagemaker awscli
    aws configure   (masukkan Access Key, Secret Key, Region)

Langkah:
    1. Upload data C.csv ke S3
    2. Training via SageMaker SKLearn Estimator
    3. Deploy endpoint
    4. Test inferencing via endpoint
"""

import os
import json
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# CONFIG
BUCKET_NAME   = "credit-score-2802504876"    
PREFIX        = "credit-score"                 
REGION        = "us-east-1"                   
ROLE_ARN      = "arn:aws:iam::334251649576:role/LabRole"  
DATA_PATH     = "../../data/C.csv"                
ENDPOINT_NAME = "credit-score-endpoint"           
INSTANCE_TRAIN  = "ml.t3.medium"
INSTANCE_DEPLOY = "ml.t3.medium"            

session    = boto3.Session(region_name=REGION)
sm_session = sagemaker.Session(boto_session=session)


# STEP 1: Upload data ke S3
def upload_data() -> str:
    s3  = session.client("s3")
    key = f"{PREFIX}/data/C.csv"
    print(f"[1/4] Uploading data → s3://{BUCKET_NAME}/{key}")
    s3.upload_file(DATA_PATH, BUCKET_NAME, key)
    uri = f"s3://{BUCKET_NAME}/{key}"
    print(f"  Done: {uri}")
    return uri


# STEP 2: Training via SageMaker
def train(s3_data_uri: str):
    """
    SageMaker akan menjalankan pipeline.py di container SKLearn.
    Pastikan pipeline.py menerima argumen --data_path dan --output_dir.
    """
    print(f"\n[2/4] Starting SageMaker training job ...")
    estimator = SKLearn(
        entry_point="pipeline.py",
        source_dir=os.path.dirname(__file__),  
        role=ROLE_ARN,
        instance_type=INSTANCE_TRAIN,
        framework_version="1.2-1",
        py_version="py3",
        hyperparameters={
            "data_path":  "/opt/ml/input/data/training/C.csv",
            "output_dir": "/opt/ml/model",
        },
        sagemaker_session=sm_session,
    )
    estimator.fit({"training": s3_data_uri})
    print(f"  Training complete. Artifacts: {estimator.model_data}")
    return estimator


# STEP 3: Deploy endpoint
def deploy(estimator):
    print(f"\n[3/4] Deploying endpoint '{ENDPOINT_NAME}' ...")
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_DEPLOY,
        endpoint_name=ENDPOINT_NAME,
    )
    print(f"  Endpoint ready: {ENDPOINT_NAME}")
    return predictor


# STEP 4: Test Cases
TEST_CASES = [
    # Good credit
    {
        "Age": "28", "Occupation": "Engineer", "Annual_Income": "90000.0",
        "Monthly_Inhand_Salary": 7000.0, "Num_Bank_Accounts": 2, "Num_Credit_Card": 3,
        "Interest_Rate": 7, "Num_of_Loan": "1", "Type_of_Loan": "Auto Loan",
        "Delay_from_due_date": 0, "Num_of_Delayed_Payment": "0",
        "Changed_Credit_Limit": "3.0", "Num_Credit_Inquiries": 1.0,
        "Credit_Mix": "Good", "Outstanding_Debt": "200.0",
        "Credit_Utilization_Ratio": 15.0, "Credit_History_Age": "6 Years and 0 Months",
        "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 120.0,
        "Amount_invested_monthly": "500.0",
        "Payment_Behaviour": "Low_spent_Large_value_payments", "Monthly_Balance": 1200.0,
    },
    # Standard credit
    {
        "Age": "35", "Occupation": "Teacher", "Annual_Income": "45000.0",
        "Monthly_Inhand_Salary": 3200.0, "Num_Bank_Accounts": 4, "Num_Credit_Card": 5,
        "Interest_Rate": 14, "Num_of_Loan": "3",
        "Type_of_Loan": "Personal Loan, and Student Loan",
        "Delay_from_due_date": 10, "Num_of_Delayed_Payment": "5",
        "Changed_Credit_Limit": "6.0", "Num_Credit_Inquiries": 4.0,
        "Credit_Mix": "Standard", "Outstanding_Debt": "1500.0",
        "Credit_Utilization_Ratio": 35.0, "Credit_History_Age": "4 Years and 6 Months",
        "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 250.0,
        "Amount_invested_monthly": "150.0",
        "Payment_Behaviour": "High_spent_Small_value_payments", "Monthly_Balance": 300.0,
    },
    # Poor credit
    {
        "Age": "46", "Occupation": "Teacher", "Annual_Income": "9786.86",
        "Monthly_Inhand_Salary": 951.57, "Num_Bank_Accounts": 9, "Num_Credit_Card": 10,
        "Interest_Rate": 34, "Num_of_Loan": "2",
        "Type_of_Loan": "Personal Loan, and Student Loan",
        "Delay_from_due_date": 58, "Num_of_Delayed_Payment": "14",
        "Changed_Credit_Limit": "15.53", "Num_Credit_Inquiries": 6.0,
        "Credit_Mix": "Standard", "Outstanding_Debt": "2149.9",
        "Credit_Utilization_Ratio": 23.45, "Credit_History_Age": "3 Years and 2 Months",
        "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 11.56,
        "Amount_invested_monthly": "54.46",
        "Payment_Behaviour": "Low_spent_Medium_value_payments", "Monthly_Balance": 309.13,
    },
]


def test_endpoint(predictor):
    from sagemaker.serializers import JSONSerializer
    from sagemaker.deserializers import JSONDeserializer
    predictor.serializer   = JSONSerializer()
    predictor.deserializer = JSONDeserializer()

    print("\n[4/4] Running test cases ...")
    for i, case in enumerate(TEST_CASES, 1):
        result = predictor.predict(case)
        print(f"  Test case {i}: {result}")


# MAIN
if __name__ == "__main__":
    print("=" * 60)
    print("  AWS SageMaker Pipeline — Credit Score Classification")
    print("=" * 60)

    s3_uri    = upload_data()
    estimator = train(s3_uri)
    predictor = deploy(estimator)
    test_endpoint(predictor)

    print(f"\n✅ Pipeline selesai. Endpoint: {ENDPOINT_NAME}")
    print("⚠️  Jangan lupa hapus endpoint setelah selesai agar tidak dikenakan biaya:")
    print(f"    predictor.delete_endpoint()")
