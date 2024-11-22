# Jay_Khodiyar
## India AI CyberGuard

## Project Title
**Crime Report Classification Using NLP**

## Institute Name
Monark University

## Team Members
- **Amit Gurjar** - Team Leader | Expertise: AI/ML, Computer Vision | [Email](mailto:amitgurjar8155@gmail.com)
- **Ronak Gohil** - Team Member | Expertise: AI/ML, Python | [Email](mailto:ronakgohil240@gmail.com)
- **Harshil Sangani** - Team Member | Expertise: AI/ML, Data Visualization | [Email](mailto:harshilsangani07@gmail.com)
- **Yash Sapra** - Team Member | Expertise: AI/ML, Python | [Email](mailto:yprajapati276@gmail.com)

## Project Overview
Our project tackles subcategory prediction in crime-related datasets, created for the **India AI Cyberguard Hackathon**. The dataset contains raw, unstructured text about crime reports classified into categories, subcategories, and crime descriptions. We aim to develop a robust NLP model to classify these text entries accurately.

## Objective
Enable accurate classification of crime-related data into subcategories and category to streamline and enhance data insights and regulatory compliance.

## Project Structure
```plaintext
├── Datasets
│ ├── Draft+IndiaAI+CyberGuard+AI+Hackathon_+Dataset+Usage+Guidelines.pdf
│ ├── test.csv
│ └── train.csv
├── Images
│ ├── category_distribution.png
│ ├── Screenshot 2024-11-05 222313.png
│ ├── subcategory_distribution.png
│ ├── WhatsApp Image 2024-11-07 at 15.17.03_9da14cad.jpg
│ ├── WhatsApp Image 2024-11-07 at 15.19.20_02f834d6.jpg
│ └── wordcloud.png
├── eda
│ ├── eda_final - train.ipynb
│ ├── eda_final-test.ipynb
│ └── eda_imbalance_data.ipynb
├── models
│ ├── main model
│ │ ├── gru_attention_model (1).h5
│ │ ├── label_encoder.pkl
│ │ └── tokenizer.pkl
│ └── other models
│   ├── le.joblib
│   ├── le_cat.joblib
│   ├── log_reg_pipeline_model.joblib
│   ├── vectorizer.joblib
│   └── xgb_model.joblib
├── prediction
│ └── predicted_output1.csv
├── processed_datasets
│ ├── Balanced1.0.csv
│ └── test_preproccessed.csv
├── trial_model_training
│ ├── cat_pred.ipynb
│ ├── prediction.ipynb
│ ├── preprocess_+_Xgboost.ipynb
│ ├── random_f.ipynb
│ └── sub_xgb.ipynb
├── .gitattributes
├── LICENSE
├── README.md
├── final_report.pdf
└── full-pipeline-code.ipynb
```
## Methodology
1. **Our Data Structure**
   ## Categories and Subcategories
| Category                   | Subcategory                                                                                               |
|----------------------------|----------------------------------------------------------------------------------------------------------|
| **Other Cyber Crime**      | - Any Other Cyber Crime <br> - Cheating by Impersonation <br> - Cyber Bullying/Stalking/Sexting <br> - Cyber Terrorism <br> - Damage to Computer Systems <br> - Data Breaches <br> - Defacement of Websites or Unauthorized Changes <br> - Defacement/Hacking <br> - Denial of Service (DoS) and Distributed Denial of Service (DDoS) attacks <br> - Email Hacking <br> - Email Phishing <br> - Fake/Impersonating Profile <br> - Impersonating Email <br> - Intimidating Email <br> - Online Matrimonial Fraud <br> - Malicious code attacks (e.g., virus, worm, Trojan, Bots, Spyware, Ransomware, Crypto miners) <br> - Online Cyber Trafficking <br> - Online Job Fraud <br> - Profile Hacking/Identity Theft <br> - Provocative Speech of Unlawful Acts <br> - Ransomware <br> - SQL Injection <br> - Tampering with Computer Source Documents <br> - Unauthorized Access/Data Breach |
| **Financial Fraud Crimes** | - UPI-Related Frauds <br> - Internet Banking-Related Fraud <br> - E-Wallet Related Frauds <br> - Debit/Credit Card Fraud or SIM Swap Fraud <br> - Fraud Call/Vishing <br> - Cryptocurrency Crime <br> - Demat/Depository Fraud <br> - Online Gambling/Betting Fraud <br> - Business Email Compromise/Email Takeover |
| **Women/Child Related Crime** | - Rape/Gang Rape-Sexually Abusive Content <br> - Sale, Publishing and Transmitting Obscene Material/Sexually Explicit Material <br> - Child Pornography/Child Sexual Abuse Material (CSAM) |



3. **Data Preprocessing**:
- NLTK: Used the Natural Language Toolkit (NLTK) for text processing, including tokenization, stopword removal, and lemmatization.
- Normalized, Tokenized, and Cleaned Text Data: Performed text normalization, tokenization, and cleaning to ensure that the data was suitable for modeling.
- Regular Expressions (RegEx): Applied RegEx to remove noise, such as special characters, URLs, and unnecessary punctuation.
- Exploratory Data Augmentation (EDA): Implemented data augmentation techniques to balance the dataset and address any class imbalance.

4. **Data Balancing**:
- Initial Distribution: Checked initial distribution of categories and subcategories.
- Main Categories Balancing Oversampled categories to match the maximum count.
- Subcategories Balancing Balanced subcategories within each main category using resampling or SMOTE.
- Final Dataset Verified balanced distribution of both categories and subcategories.


5. **Feature Engineering**:
- Word Embeddings (300-Dimensional): Used pre-trained word embeddings (e.g., GloVe or Word2Vec) to convert text into dense 300-dimensional vectors, capturing 
  rich contextual information.
- TF-IDF: Converted text to vectors using TF-IDF with unigram and bigram tokens, capturing important terms and their relevance in the dataset.

    
6. **Model Selection**:
   - Choose **GRU** (Gated Recurrent Unit) for primary classification.
   - Fine-tuned parameters to optimize precision, recall, and F1-score.

---

## Results
The final model (GRU) achieved

 **Classification Report Precision and Recall for Each Subcategory**

| **SUBCATEGORY** | **PRECISION** | **RECALL** | **F1-SCORE** | **SUPPORT** |
|-----------------|---------------|------------|--------------|-------------|
| Any Other Cyber Crime | 0.9099 | 0.8296 | 0.8679 | 4103 |
| Business Email Compromise/Email Takeover | 0.9987 | 1.0000 | 0.9993 | 4589 |
| Cheating by Impersonation | 0.9759 | 0.9910 | 0.9834 | 4006 |
| Child Pornography/Child Sexual Abuse Material (CSAM) | 0.9971 | 0.9947 | 0.9959 | 8897 |
| Cryptocurrency Crime | 0.9970 | 1.0000 | 0.9985 | 4597 |
| Cyber Bullying/Stalking/Sexting | 0.9787 | 0.9764 | 0.9775 | 3946 |
| Cyber Terrorism | 0.9951 | 1.0000 | 0.9975 | 4032 |
| Damage to Computer Systems | 1.0000 | 1.0000 | 1.0000 | 3920 |
| Data Breaches | 1.0000 | 1.0000 | 1.0000 | 4062 |
| Debit/Credit Card Fraud or SIM Swap Fraud | 0.9400 | 0.9453 | 0.9426 | 4659 |
| Defacement of Websites or Unauthorized Changes | 1.0000 | 1.0000 | 1.0000 | 3866 |
| Defacement/Hacking | 1.0000 | 1.0000 | 1.0000 | 4108 |
| Demat/Depository Fraud | 0.9793 | 1.0000 | 0.9896 | 4504 |
| Denial of Service (DoS) and Distributed Denial of Service (DDoS) attacks | 1.0000 | 1.0000 | 1.0000 | 4072 |
| E-Wallet Related Frauds | 0.9704 | 0.9799 | 0.9751 | 4717 |
| Email Hacking | 0.9432 | 1.0000 | 0.9707 | 4048 |
| Email Phishing | 0.9990 | 0.9977 | 0.9983 | 3861 |
| Fake/Impersonating Profile | 0.9785 | 0.9924 | 0.9854 | 3938 |
| Fraud Call/Vishing | 0.9520 | 0.9701 | 0.9610 | 4683 |
| Impersonating Email | 1.0000 | 1.0000 | 1.0000 | 4141 |
| Internet Banking-Related Fraud | 0.9354 | 0.9505 | 0.9429 | 4648 |
| Intimidating Email | 1.0000 | 1.0000 | 1.0000 | 3968 |
| Malicious code attacks (e.g., virus, worm, Trojan, Bots, Spyware, Ransomware, Crypto miners) | 1.0000 | 1.0000 | 1.0000 | 4039 |
| Online Cyber Trafficking | 0.9985 | 0.9980 | 0.9983 | 4028 |
| Online Gambling/Betting Fraud | 0.9968 | 0.9917 | 0.9943 | 4716 |
| Online Job Fraud | 0.9869 | 0.9995 | 0.9931 | 3980 |
| Online Matrimonial Fraud | 0.9969 | 1.0000 | 0.9985 | 3870 |
| Profile Hacking/Identity Theft | 0.9886 | 0.9881 | 0.9884 | 3953 |
| Provocative Speech of Unlawful Acts | 0.9966 | 0.9964 | 0.9965 | 3846 |
| Ransomware | 0.9998 | 1.0000 | 0.9999 | 4147 |
| Rape/Gang Rape-Sexually Abusive Content | 0.9999 | 1.0000 | 0.9999 | 8772 |
| SQL Injection | 1.0000 | 1.0000 | 1.0000 | 3948 |
| Sale, Publishing and Transmitting Obscene Material/Sexually Explicit Material | 0.9870 | 0.9951 | 0.9910 | 8773 |
| Tampering with Computer Source Documents | 1.0000 | 1.0000 | 1.0000 | 4067 |
| UPI-Related Frauds | 0.9179 | 0.8289 | 0.8711 | 4762 |
| Unauthorized Access/Data Breach | 0.9886 | 0.9931 | 0.9908 | 3921 |

### Accuracy
- **Accuracy**: 0.9840

### Macro Average
- **Macro avg**: Precision: 0.9835, Recall: 0.9838, F1-Score: 0.9835

### Weighted Average
- **Weighted avg**: Precision: 0.9838, Recall: 0.9840


---

## Challenges Faced
1. **Messy Text**: Managed inconsistencies, typos, and informal language.
2. **Data Imbalance**: Used sampling techniques to improve model fairness across classes.
3. **Model Overfitting**: Addressed through regularization, cross-validation, and batch training.

## Conclusion
The model has the potential to enhance the National Cyber Crime Reporting Portal by guiding users to categorize their reports accurately in real-time. It shows promise in improving the efficiency of report handling in cybercrime cases.

## Future Work
1. **Advanced NLP Models**: Integrating models like BERT for better context understanding.
2. **Real-Time Deployment**: Deploying as a classification service to process new cases.

## References
- Data Processing: NumPy, pandas, Scikit-learn, NLTK, Imblearn
- Model Training: **GRU (Gated Recurrent Unit)**, XGBoost, LogisticRegression ,Random Forest
- Data Balancing: Scikit-learn Resampling
- EDA: Matplotlib, Seaborn
