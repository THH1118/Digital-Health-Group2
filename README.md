# Predicting Parathyroidectomy Success Rate and Postoperative Hungry Bone Syndrome Using Machine Learning

## Overview

This project aims to apply machine learning (ML) techniques to predict:

1. The success of **parathyroidectomy (PTX)** surgery.
2. The risk of developing **Hungry Bone Syndrome (HBS)** postoperatively.
3. The postoperative **serum calcium concentration** levels.

These predictions are based on preoperative clinical and biochemical data of patients with secondary hyperparathyroidism (SHPT), particularly those undergoing long-term dialysis.

## Motivation

Secondary hyperparathyroidism is a serious complication in chronic kidney disease (CKD) patients, often requiring surgical intervention. However, outcomes such as HBS can vary and lead to serious complications. Accurate risk stratification and outcome prediction are crucial for improving patient care.

## Data

- 1,524 patients from Kaohsiung Chang Gung Memorial Hospital
- Data includes:
  - Demographics
  - Dialysis history
  - Biochemical markers: iPTH, ALP, calcium, ferritin, etc.

## Approach

- Data preprocessing and imputation of missing values
- Feature selection and engineering
- Model training using multiple ML algorithms (e.g., XGBoost, Random Forest, CatBoost)
- Evaluation via cross-validation and metrics such as AUC, MAE

## Goals

- Assist clinicians in **preoperative decision-making**
- Reduce the incidence of **postoperative complications**
- Enable **personalized treatment strategies** for SHPT patients

