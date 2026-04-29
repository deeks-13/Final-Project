import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline as SkPipeline
import shap
from joblib import load

warnings.simplefilter("ignore")

# ── Path Configuration ────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Make sure current_dir is in path so Custom_Classes can be found
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ── Custom Classes (must import after sys.path is set) ────────────────────────
from Custom_Classes import (
    RecodeCategoricals,
    ParseCreditDate,
    EncodeGrade,
    OneHotEncodeCats,
    LoanFeatureEngineer,
    AutoPowerTransformer,
    FeatureSelector
)

# ── Load Dataset ──────────────────────────────────────────────────────────────
file_path = os.path.join(current_dir, 'accepted_2007_to_2018Q4.csv')
dataset = pd.read_csv(file_path, index_col=0, low_memory=False)
keep_statuses = ['Fully Paid', 'Charged Off']
dataset = dataset[dataset['loan_status'].isin(keep_statuses)].copy()

# ── Secrets ───────────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── AWS Session ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Model Configuration ───────────────────────────────────────────────────────
MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "explainer_sentiment.shap",
    "pipeline"  : "finalized_loan_model.tar.gz",
    "inputs"    : [
        {"name": "loan_amnt",  "min": 0.0, "max": 40000.0,  "default": 10000.0, "step": 100.0},
        {"name": "int_rate",   "min": 0.0, "max": 35.0,     "default": 12.0,    "step": 0.1},
        {"name": "dti",        "min": 0.0, "max": 50.0,     "default": 15.0,    "step": 0.1},
        {"name": "annual_inc", "min": 0.0, "max": 500000.0, "default": 60000.0, "step": 1000.0},
    ]
}

FEATURE_COLS = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade',
    'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
    'purpose', 'dti', 'delinq_2yrs', 'earliest_cr_line',
    'fico_range_low', 'fico_range_high', 'inq_last_6mths',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc'
]

# ── Load Pipeline from S3 ─────────────────────────────────────────────────────
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]
    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(joblib_file)

# ── Load SHAP Explainer from S3 ───────────────────────────────────────────────
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return load(f)

# ── Prediction ────────────────────────────────────────────────────────────────
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        mapping  = {0: "Fully Paid ✅", 1: "Charged Off ⚠️"}
        return mapping.get(pred_val), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# ── SHAP Explanation ──────────────────────────────────────────────────────────
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    # Rebuild preprocessing pipeline — remove SMOTE and classifier
    pre_steps = [(n, s) for n, s in best_pipeline.steps if n not in ('clf', 'smote')]
    pre_pipe  = SkPipeline(steps=pre_steps)

    input_df             = pd.DataFrame(input_df)
    input_df_transformed = pre_pipe.transform(input_df)

    # Generic feature names (pipeline output is numpy so names are positional)
    feature_names        = [f'feature_{i}' for i in range(input_df_transformed.shape[1])]
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)

    shap_values = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 1])  # class 1 = Charged Off / Default
    st.pyplot(fig)

    top_feature = pd.Series(
        shap_values[0, :, 1].values,
        index=shap_values[0, :, 1].feature_names
    ).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("🏦 Loan Default Prediction")

with st.form("pred_form"):
    st.subheader("Loan Application Inputs")
    cols = st.columns(2)
    user_inputs = {}
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'],
                max_value=inp['max'],
                value=inp['default'],
                step=inp['step']
            )
    submitted = st.form_submit_button("Run Prediction")

# Build input row from dataset defaults + user overrides
original = dataset[FEATURE_COLS].iloc[0:1].to_dict()
original.update(user_inputs)

if submitted:
    res, status = call_model_api(original)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(original, session, aws_bucket)
    else:
        st.error(res)
