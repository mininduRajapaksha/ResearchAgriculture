import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from model_loader import load_model_and_scaler, predict_price
from flask import current_app
import requests
from flask import jsonify
from flask_cors import CORS
from flask_caching import Cache


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
CORS(app)  # Add this after app = Flask(__name__)

cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

# MongoDB configuration
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)


@app.route('/')
def index():
    fruits = list(mongo.db.fruits.find({}))
    vegetables = list(mongo.db.vegetables.find({}))  # Add this line
    return render_template('index.html', fruits=fruits, vegetables=vegetables)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)