
# coding: utf-8

from re import X
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

df_1=pd.read_csv("Upsample_Churn.csv")

q = ""

@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    
    '''
    AGE 
    CUS_Month_Income 
    CUS_Gender
    CUS_Marital_Status
    YEARS_WITH_US
    S1_total_debit_transactions
    S2_total_debit_transactions
    S3_total_debit_transactions
    S1_total_debit
    S2_total_debit
    S3_total_debit
    S1_total_credit_transactions
    S2_total_credit_transactions
    S3_total_credit_transactions
    S1_total_credit
    S2_total_credit
    S3_total_credit
    total debit amount
    total debit transactions
    total credit amount
    total credit transactions
    total transactions
    CUS_Target
    TAR_Desc
    '''
    

    
    inputQuery1 = int(request.form['query1'])
    inputQuery2 = float(request.form['query2'])
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = int(request.form['query5'])
    inputQuery6 = int(request.form['query6'])
    inputQuery7 = int(request.form['query7'])
    inputQuery8 = int(request.form['query8'])
    inputQuery9 = float(request.form['query9'])
    inputQuery10 = float(request.form['query10'])
    inputQuery11 = float(request.form['query11'])
    inputQuery12 = int(request.form['query12'])
    inputQuery13 = int(request.form['query13'])
    inputQuery14 = int(request.form['query14'])
    inputQuery15 = float(request.form['query15'])
    inputQuery16 = float(request.form['query16'])
    inputQuery17 = float(request.form['query17'])
    inputQuery18 = float(request.form['query18'])
    inputQuery19 = int(request.form['query19'])
    inputQuery20 = float(request.form['query20'])
    inputQuery21 = int(request.form['query21'])
    inputQuery22 = int(request.form['query22'])
    inputQuery23 = int(request.form['query23'])
    inputQuery24 = request.form['query24']

    model = pickle.load(open("model.sav", "rb"))
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19,inputQuery20,inputQuery21,
             inputQuery22,inputQuery23,inputQuery24]]
    
    new_df = pd.DataFrame(data, columns = ['AGE', 'CUS_Month_Income', 'CUS_Gender', 'CUS_Marital_Status',
       'YEARS_WITH_US', 'S1_total_debit_transactions',
       'S2_total_debit_transactions', 'S3_total_debit_transactions',
       'S1_total_debit', 'S2_total_debit', 'S3_total_debit',
       'S1_total_credit_transactions', 'S2_total_credit_transactions',
       'S3_total_credit_transactions', 'S1_total_credit', 'S2_total_credit',
       'S3_total_credit', 'total debit amount', 'total debit transactions',
       'total credit amount', 'total credit transactions',
       'total transactions', 'CUS_Target', 'TAR_Desc'])
    
    df_2 = pd.concat([df_1, new_df], ignore_index = True) 

    df_new = df_2.copy()
    
    #drop column customerID and tenure
    df_2.drop(['Unnamed: 0', 'Status','CUS_Customer_Since', 'category','Suggestions'], axis=1, inplace= True)  
    
    df_new.drop(['Unnamed: 0', 'Status','CUS_Customer_Since','Suggestions'], axis=1, inplace= True)
    
    df_2['CUS_Gender'] = le.fit_transform(df_2['CUS_Gender'])
    df_2['CUS_Marital_Status'] = le.fit_transform(df_2['CUS_Marital_Status'])
    
    dict_new = {'PLATINUM':0, 'EXECUTIVE':1, 'MIDLE': 2, 'LOW': 3}
    df_2.replace({'TAR_Desc': dict_new}, inplace=True)
    

    #df_new["category"] = np.where(df_new["total debit transactions"]>df_new["total credit transactions"], 'debit', 'credit')
    #Recommendation Engine  Code
    if int(df_new.tail(1)["total debit transactions"])>int(df_new.tail(1)["total credit transactions"]):
        x_o3 = 'debit'
    else:
        x_o3 = 'credit'   

    if x_o3 == 'debit':
        if max(float(df_new.tail(1)['S1_total_debit']), float(df_new.tail(1)['S2_total_debit']), float(df_new.tail(1)['S3_total_debit'])) == float(df_new.tail(1)['S1_total_debit']):
            recommend = "Debit Transactions for S1 service"
        elif max(float(df_new.tail(1)['S1_total_debit']), float(df_new.tail(1)['S2_total_debit']), float(df_new.tail(1)['S3_total_debit'])) == float(df_new.tail(1)['S2_total_debit']):
            recommend = "Debit Transactions for S2 service"
        elif max(float(df_new.tail(1)['S1_total_debit']), float(df_new.tail(1)['S2_total_debit']), float(df_new.tail(1)['S3_total_debit'])) == float(df_new.tail(1)['S3_total_debit']):
            recommend = "Debit Transactions for S3 service"
        else:
            recommend = "NI_debit"
    elif x_o3 == 'credit':
        if max(float(df_new.tail(1)['Credit Transactions for S1 service']), float(df_new.tail(1)['S2_total_credit']), float(df_new.tail(1)['S3_total_credit'])) == float(df_new.tail(1)['S1_total_credit']):
            recommend = "S1_credit"
        elif max(float(df_new.tail(1)['Credit Transactions for S2 service']), float(df_new.tail(1)['S2_total_credit']), float(df_new.tail(1)['S3_total_credit'])) == float(df_new.tail(1)['S2_total_credit']):
            recommend = "S2_credit"
        elif max(float(df_new.tail(1)['Credit Transactions for S3 service']), float(df_new.tail(1)['S2_total_credit']), float(df_new.tail(1)['S3_total_credit'])) == float(df_new.tail(1)['S3_total_credit']):
            recommend = "S3_credit"
        else:
            recommend = "NI_credit"

    
    #Conditions for output
    single = model.predict(df_2.tail(1))
    probablity = model.predict_proba(df_2.tail(1))[:,1]
    
    if single==1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {}".format(probablity*100)
        o3 = f"Customer has majorly used: {x_o3} Card, we can provide offers on {recommend}"
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probablity*100)
        if (probablity*100) < 75:
            o3 = f"Customer has majorly used: {x_o3} Card, we can provide offers on {recommend}"
            
        else:
            o3 = "Customer is active with us and we can provide more offers for loyalty and we can provide offers on {recommend}"
        
    return render_template('home.html', output1=o1, output2=o2, output3= o3,
                           query1 = request.form['query1'], 
                           query2 = request.form['query2'],
                           query3 = request.form['query3'],
                           query4 = request.form['query4'],
                           query5 = request.form['query5'], 
                           query6 = request.form['query6'], 
                           query7 = request.form['query7'], 
                           query8 = request.form['query8'], 
                           query9 = request.form['query9'], 
                           query10 = request.form['query10'], 
                           query11 = request.form['query11'], 
                           query12 = request.form['query12'], 
                           query13 = request.form['query13'], 
                           query14 = request.form['query14'], 
                           query15 = request.form['query15'], 
                           query16 = request.form['query16'], 
                           query17 = request.form['query17'],
                           query18 = request.form['query18'], 
                           query19 = request.form['query19'],
                           query20 = request.form['query20'],
                           query21 = request.form['query21'],
                           query22 = request.form['query22'],
                           query23 = request.form['query23'],
                           query24 = request.form['query24'])
    
app.run()

