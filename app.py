import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from flask import Flask, render_template, request, jsonify, session
import os


app = Flask(__name__)
# Secret key for session management
app.secret_key = os.getenv('app_key')
print(app.secret_key)

df = pd.read_csv('new_df.csv')
# Loading the vectorizer 
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

# Loading the similarity matrix
similarity = pickle.load(open('model/cos_sim.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    render_template('index.html'), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    product=None
    if request.headers['Content-Type'] == 'application/json': 
        product = request.get_json()['product']
    else:
        product = request.form.get('product', '')
    names = df['name']
    product_index = df[df['name']==product].index[0]
    matrix = similarity[product_index]    
    product_list = sorted(list(enumerate(matrix)), reverse=True, key=lambda x:x[1])[1:10]
    product_indices = [i[0] for i in product_list]
    # Convert the indexes back into titles 
    recommendation = names.iloc[product_indices]
    return jsonify({'response':recommendation})




    

