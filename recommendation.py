from pymongo import MongoClient
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import flask
from flask import request, jsonify

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from bson.objectid import ObjectId

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/associativeRules', methods=['GET'])
def api_all():
    request_data = request.get_json()	
    client = MongoClient('mongodb://Haris:5337618@cluster0-shard-00-00.uqzho.mongodb.net:27017,cluster0-shard-00-01.uqzho.mongodb.net:27017,cluster0-shard-00-02.uqzho.mongodb.net:27017/myFirstDatabase?ssl=true&replicaSet=atlas-734d59-shard-0&authSource=admin&retryWrites=true&w=majority')
    #client = MongoClient('mongodb+srv://Haris:5337618@cluster0.uqzho.mongodb.net/Ar-Salon?retryWrites=true&    w=majority')
    mydb = client["myFirstDatabase"]
    coll = mydb["appointments"]
    dataset = []
    
    objInstance = ObjectId(request_data["salon_id"])
    #print(request_data["salon_id"])
    #print(coll.find()[0]['_id'])
    #print(coll.find()[0]['_id'] == request_data["id"])
    
    for x in coll.find({'salon_id': objInstance}):
        smp = []
        #if (x['_id'] == request_data["id"]):
        for i in x['services']:
            smp.append(i)
        dataset.append(smp)
    if dataset == []:
        return jsonify({"status": "No data found"})

    for i in range(0,len(dataset)):
        for j in range(0,len(dataset[i])):
            dataset[i][j] = dataset[i][j]['title']


    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)

    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)



    res = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.79)
    res1 = res[['antecedents', 'consequents', 'confidence']]

    output = {}
    outputArr = []
    for i in range(0,len(res1)):
        output["antecedents"]= list(res1.iloc[i]['antecedents'])
        output["consequents"]= list(res1.iloc[i]['consequents'])
        outputArr.append(output)
    	
    #print(outputArr)
    return jsonify(outputArr)


@app.route('/SalonReccomender', methods=['GET'])
def api_salon():
    request_data = request.get_json()
    data = pd.read_csv('SalonsDataset.csv')
    data.head()
    X= data[['Service_1', 'Service_2', 'Service_3', 'Service_4', 'Service_5']]
    y = data['cat_primary']
    cat_cols = ['Service_1', 'Service_2', 'Service_3', 'Service_4', 'Service_5']
    enc = preprocessing.LabelEncoder()
    for col in cat_cols:
        X[col] = X[col].astype('str')
        X[col] = enc.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

    #Create a Gaussian Classifier
    gnb = GaussianNB()

    #Train the model using the training sets
    gnb.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = gnb.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    #print("Accuracy:",(metrics.accuracy_score(y_test, y_pred))*100 ,"%")

    test_arr = request_data['services']
    test_arr = enc.fit_transform(test_arr)
    result= gnb.predict([test_arr])
    result_json = {"category":result[0]}
    return jsonify(result_json)

app.run(port=1200)
    

#res1 = res1['antecedents'].apply(lambda x: ', '.join(list(x))).astype("unicode")
#print(type(res1))
#print (res1[(res1['confidence'] >= 0.7)].apply(lambda x: ', '.join(list(x))).astype("unicode"))