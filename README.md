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
│   ├── train.csv
│   ├── test.csv
│   └── Draft+IndiaAI+CyberGuard+AI+Hackathon_+Dataset+Usage+Guidelines.pdf
├── data_preprossesing&eda
│   ├── data cleaning.ipynb
│   ├── eda_final - train.ipynb
│   ├── eda_final - test.ipynb
│   ├── train_preprocessing.ipynb
│   └── test_preprocessing.ipynb
├── Images
│   ├── image1.png
│   ├── image2.png
│   ├── image3.png
│   ├── image4.png
│   ├── image5.png
│   └── image6.png
├── main_model_train
│   ├── cat_pred.ipynb
│   └── sub_xbg.ipynb
├── models
│   ├── le.joblib
│   ├── li_cat.joblib
│   ├── log_reg_pipeline_model.joblib
│   ├── vectorizer.joblib
│   └── xgb_model.joblib
├── prediction
│   └──prediction.ipynb
├── proccessed_datasets
│   ├── final_test_2.csv
│   └── abbreviated_file.csv
├── trial_model_training
│   ├── preprocess_+_Xgboost.ipynb
│   └── random_F.ipynb
├── India_AI_Cyberguard_Report.pdf
├── LICENCE
└── README.md
```
## Methodology
1. **Our Data Structure**
   ## Categories and Subcategories

| Category                   | Subcategory                                                                                               |
|----------------------------|----------------------------------------------------------------------------------------------------------|
| **Other Cyber Crime**      | - Fake/Impersonating Profile <br> - Any Other Cyber Crime <br> - Cyber Bullying/Stalking/Sexting <br> - Cheating by Impersonation <br> - Unauthorized Access/Data Breach <br> - Online Job Fraud <br> - Ransomware <br> - Malware Attacks <br> - Profile Hacking/Identity Theft <br> - Provocative Speech of Unlawful Acts <br> - Attacks on applications (e.g., E-Governance, E-Commerce) <br> - Impersonating Email <br> - Data Breaches <br> - Tampering with Computer Source Documents <br> - Defacement/Hacking <br> - Online Matrimonial Fraud <br> - Email Hacking <br> - Intimidating Email <br> - Email Phishing <br> - Online Cyber Trafficking <br> - Denial of Service (DoS) and Distributed Denial of Service (DDoS) attacks <br> - Damage to Computer Systems <br> - Cyber Terrorism |
| **Financial Fraud Crimes** | - UPI-Related Frauds <br> - Internet Banking-Related Fraud <br> - E-Wallet Related Frauds <br> - Debit/Credit Card Fraud <br> - Fraud Call/Vishing <br> - Cryptocurrency Crime <br> - Demat/Depository Fraud <br> - Online Gambling/Betting Fraud <br> - Business Email Compromise/Email Takeover |
| **Women/Child Related Crime** | - Rape/Gang Rape-Sexually Abusive Content <br> - Sale, Publishing and Transmitting Obscene Material/Sexually Explicit Material <br> - Child Pornography/Child Sexual Abuse Material (CSAM) |


3. **Data Preprocessing**:
   - Normalized, tokenized, and cleaned the text data.
   - Used Regular Expressions (RegEx) for noise removal.
   - Applied Exploratory Data Augmentation (EDA) for balancing.

4. **Feature Engineering**:
   - Converted text to vectors using **TF-IDF** with unigram and bigram tokens.

5. **Model Selection**:
   - Chose **XGBoost** for primary classification.
   - Fine-tuned parameters to optimize precision, recall, and F1-score.

## Results
The final model (XGBoost, Logistic regression) achieved:
- **Category Accuracy**: 0.98
- **Category F1-score**: 0.98
- **Subcategory Accuracy**: 0.93
- **Macro Precision**: 0.93
- **Macro Recall**: 0.93
- **Macro F1-score**: 0.93
## Precision and Recall for Each Subcategory

| SUBCATEGORY | PRECISION | RECALL | F1-SCORE | SUPPORT |
|-------------|-----------|--------|----------|---------|
| ANY OTHER CYBER CRIME | 0.9195 | 0.7408 | 0.8205 | 43771 |
| ATTACKS ON APPLICATIONS (E.G., E-GOVERNANCE, E-COMMERCE) | 0.9730 | 0.9774 | 0.9752 | 22055 |
| BUSINESS EMAIL COMPROMISE/EMAIL TAKEOVER | 0.9900 | 1.0 | 0.9950 | 26514 |
| CHEATING BY IMPERSONATION | 0.9076 | 0.7951 | 0.8476 | 21668 |
| CHILD PORNOGRAPHY/CHILD SEXUAL ABUSE MATERIAL (CSAM) | 0.9872 | 1.0 | 0.9935 | 24471 |
| CRYPTOCURRENCY CRIME | 0.9896 | 1.0 | 0.9947 | 26615 |
| CYBER BULLYING/STALKING/SEXTING | 0.8959 | 0.7533 | 0.8184 | 21596 |
| CYBER TERRORISM | 0.9980 | 1.0 | 0.9990 | 21682 |
| DAMAGE TO COMPUTER SYSTEMS | 0.9994 | 1.0 | 0.9997 | 21633 |
| DATA BREACHES | 0.9613 | 0.9808 | 0.9710 | 22055 |
| DEBIT/CREDIT CARD FRAUD | 0.8248 | 0.8457 | 0.8351 | 26446 |
| DEFACEMENT/HACKING | 0.9835 | 0.9725 | 0.9779 | 43756 |
| DEMAT/DEPOSITORY FRAUD | 0.9513 | 0.9789 | 0.9649 | 26342 |
| DENIAL OF SERVICE (DOS) AND DISTRIBUTED DENIAL OF SERVICE (DDOS) ATTACKS | 0.9749 | 0.9913 | 0.9831 | 22055 |
| E-WALLET RELATED FRAUDS | 0.9085 | 0.8903 | 0.8993 | 26410 |
| EMAIL HACKING | 0.9836 | 0.9971 | 0.9903 | 21832 |
| EMAIL PHISHING | 0.9963 | 1.0 | 0.9981 | 21757 |
| FAKE/IMPERSONATING PROFILE | 0.8990 | 0.8920 | 0.8955 | 21578 |
| FRAUD CALL/VISHING | 0.7601 | 0.8004 | 0.7797 | 26353 |
| IMPERSONATING EMAIL | 0.9999 | 1.0 | 0.9999 | 22055 |
| INTERNET BANKING-RELATED FRAUD | 0.8484 | 0.8083 | 0.8279 | 26657 |
| INTIMIDATING EMAIL | 1.0 | 1.0 | 1.0 | 22055 |
| MALWARE ATTACKS | 0.9802 | 0.9902 | 0.9852 | 22055 |
| ONLINE CYBER TRAFFICKING | 0.9933 | 1.0 | 0.9966 | 21795 |
| ONLINE GAMBLING/BETTING FRAUD | 0.9865 | 1.0 | 0.9932 | 26441 |
| ONLINE JOB FRAUD | 0.9111 | 0.9936 | 0.9505 | 21754 |
| ONLINE MATRIMONIAL FRAUD | 0.9976 | 1.0 | 0.9988 | 21084 |
| PROFILE HACKING/IDENTITY THEFT | 0.9244 | 0.9202 | 0.9223 | 21630 |
| PROVOCATIVE SPEECH OF UNLAWFUL ACTS | 0.9825 | 0.9977 | 0.9900 | 20944 |
| RANSOMWARE | 0.9836 | 0.9782 | 0.9809 | 44110 |
| RAPE/GANG RAPE-SEXUALLY ABUSIVE CONTENT | 0.9995 | 0.9976 | 0.9985 | 24791 |
| SALE, PUBLISHING AND TRANSMITTING OBSCENE MATERIAL/SEXUALLY EXPLICIT MATERIAL | 0.7954 | 0.9357 | 0.8599 | 49017 |
| TAMPERING WITH COMPUTER SOURCE DOCUMENTS | 0.9857 | 0.9716 | 0.9786 | 22055 |
| UPI-RELATED FRAUDS | 0.7242 | 0.7130 | 0.7186 | 26370 |
| UNAUTHORIZED ACCESS/DATA BREACH | 0.9214 | 0.9813 | 0.9504 | 21614 |

---

## Category Results

| CATEGORY                    | PRECISION | RECALL | F1-SCORE | SUPPORT |
|-----------------------------|-----------|--------|----------|---------|
| FINANCIAL FRAUD CRIMES      | 0.98      | 0.98   | 0.98     | 47,530  |
| OTHER CYBER CRIME           | 0.99      | 0.99   | 0.99     | 113,173 |
| WOMEN/CHILD RELATED CRIME   | 0.98      | 0.97   | 0.98     | 19,901  |
| **ACCURACY**                |           |        | **0.98** | 180,604 |
| **MACRO AVG**               | 0.98      | 0.98   | 0.98     | 180,604 |
| **WEIGHTED AVG**            | 0.98      | 0.98   | 0.98     | 180,604 |

 
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
- Model Training: XGBoost, LogisticRegression ,Random Forest
- Data Balancing: Scikit-learn Resampling
- EDA: Matplotlib, Seaborn
