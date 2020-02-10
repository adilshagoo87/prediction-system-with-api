from flask import Flask,request,jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np

app=Flask(__name__)

@app.route('/predict',methods=["POST"])
def predict():
	if lr:
		try:
			json_=request.get_json()
			print(json_)
			query=pd.get_dummies(pd.DataFrame(json_))
			print(query.head())
			query=query.reindex(columns=model_columns,fill_value=0)
			prediction=list(lr.predict(query))
			return jsonify({"prediction":str(prediction)})
		except:
			return jsonify({"trace":traceback.format_exc()})
	else:
		print("train the model first")
		print("no model here to use")
if __name__ == "__main__":
	try:
		port=int(sys.argv[1])
	except:
		port = 12345
	lr=joblib.load("model.pkl")
	model_columns=joblib.load("model_columns.pkl")
	print("model columns loaded")
	
	app.run(port=port,debug=True)
