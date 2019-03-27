from django.shortcuts import render
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import pickle
from django.http import HttpResponse
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Create your views here.

def index(request):
	return render(request, 'index.html')

def predict(request):
	if request.method == 'POST':
		if request.POST['head_size']:
			head_size = int(request.POST['head_size'])
			head_size_input = np.array([head_size]).reshape(-1,1)
			filename = 'LR_model.pkl'
			#file = open(os.path.join(BASE_DIR, 'LR_models', filename),'rb')
			#print('working')
			#file.close()
			LR_model = pickle.load(open(os.path.join(BASE_DIR, 'LR_models', filename),'rb'))
			output = {
		               "prediction_of_brain_weight_in_grams" : '-1'
		             }
			output['prediction_of_brain_weight_in_grams'] = LR_model.predict(head_size_input)[0]
			output['prediction_of_brain_weight_in_grams'] = int(output['prediction_of_brain_weight_in_grams'])
			return render(request,'result.html',{'output_predicted':output['prediction_of_brain_weight_in_grams']})

	return render(request,'predict.html')

	
		



	






    



