from flask import Flask, request, render_template, redirect, session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import pickle
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.metrics import Precision, Recall
from keras.callbacks import ModelCheckpoint
import os
from flask import send_from_directory
from sqlalchemy import func
import sqlite3  
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'
model = pickle.load(open("RFmodel.pkl", "rb"))
soil_model = load_model('best_model.hdf5')




class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()


class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    crop_name = db.Column(db.String(100), nullable=False)
    predicted_price = db.Column(db.Float, nullable=False)

    def __init__(self, user_id, crop_name, predicted_price):
        self.user_id = user_id
        self.crop_name = crop_name
        self.predicted_price = predicted_price

# Create the tables
with app.app_context():
    db.create_all()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid user')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html', user=user)
    return redirect('/login')

@app.route('/croppred')
def croppred():
    return render_template('croppred.html')

@app.route('/crop-predict', methods=["GET", "POST"])
def crop_prediction():
    if request.method == "POST":
        # Nitrogen
        nitrogen = float(request.form["nitrogen"])
        # Phosphorus
        phosphorus = float(request.form["phosphorus"])
        # Potassium
        potassium = float(request.form["potassium"])
        # Temperature
        temperature = float(request.form["temperature"])
        # Humidity Level
        humidity = float(request.form["humidity"])
        # PH level
        phLevel = float(request.form["ph"])
        # Rainfall
        rainfall = float(request.form["rainfall"])

        # Making predictions from the values:

        # predictions = model.predict([[nitrogen, phosphorus, potassium, temperature, humidity, phLevel, rainfall]])
        # output = predictions[0:11]
        # finalOutput = output.capitalize()
        # cropStatement= finalOutput+" should be grown"
        predictions_proba = model.predict_proba([[nitrogen, phosphorus, potassium, temperature, humidity, phLevel, rainfall]])
        classes = model.classes_
        probabilities = predictions_proba[0]
        # Combine the classes and probabilities into a list of tuples
        results = list(zip(classes, probabilities))
        # Sort the results by the probability in descending order
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        # Get the top 10 results
        top_results = sorted_results[:5]
        session['top_results'] = top_results
        # Create a statement for the top 10 crops
        top_crop_statement = "The top 5 predicted crops are: " + ", ".join([crop.capitalize() for crop, _ in top_results])

        
        crop_images = {
            "rice": "../static/rice.jpg",
            "mango": "../static/mango.jpg",
            "chickpea": "../static/chickpea.jpg",
            "orange": "../static/orange.jpg",
            "pomegranate": "../static/pomegranate.jpg",
            "apple": "../static/apple.jpg",
            "banana": "../static/banana.jpg",
            "coconut": "../static/coconut.jpg",
            "coffee": "../static/coffee.jpg",
            "cotton": "../static/cotton.jpg",
            "grapes": "../static/grapes.jpg",
            "jute": "../static/jute.jpg",
            "kidneybeans": "../static/kidneybeans",
            "lentil": "../static/lentil.jpg",
            "maize": "../static/maize.jpg",
            "mothbeans": "../static/mothbeans.jpg",
            "mungbeans": "../static/mungbeans.jpg",
            "muskmelon":"../static/muskmelon.jpg",
            "papaya": "../static/papaya.jpg",
            "pigeonpeas": "../static/pigeonpeas.jpg",
            "watermelon": "../static/watermelon",
            "blackgram": "../static/blackgram.jpg",

            
            # Add more crop names and image paths as needed
        }
        
        predicted_crops = [crop_images.get(crop.lower(), "../static/default.jpg") for crop, _ in top_results]

        imagefile = request.files['imagefile']
        image_path = "./static/images/" + imagefile.filename
        imagefile.save(image_path)
        img = image.load_img(image_path)
        img = img.resize((224, 224))
        # Convert the image to a numpy array and normalize the pixel values
        img_array = np.array(img) / 255.0
        # Add a batch dimension to the array
        img_array = np.expand_dims(img_array, axis=0)
        # Make the prediction
        pred = soil_model.predict(img_array)
        # Get the true label
        true_label = np.argmax(pred)


        # Save the prediction results to the database
        user = User.query.filter_by(email=session['email']).first()
        for crop, _ in top_results:
            new_prediction = PredictionResult(user_id=user.id, crop_name=crop, predicted_price=_)  # Replace '_' with the actual predicted price
            db.session.add(new_prediction)
        db.session.commit()

        return render_template('crop_result.html', top_results=top_results, prediction_text=top_crop_statement, crop_images=predicted_crops, prediction=true_label, image_path=image_path)

    # Return a default response if the request method is not "POST"
    return "Invalid request"



@app.route('/align-to-market', methods=['POST'])
def align_to_market():
    # # Get the selected district from the form
    # district = request.form['district']
    # top_results = session.get('top_results', None)
    # crop_names = [crop for crop, _ in top_results]

    crop_images = {
        "rice": "../static/rice.jpg",
        "mango": "../static/mango.jpg",
        "chickpea": "../static/chickpea.jpg",
        "orange": "../static/orange.jpg",
        "pomegranate": "../static/pomegranate.jpg",
        "apple": "../static/apple.jpg",
        "banana": "../static/banana.jpg",
        "coconut": "../static/coconut.jpg",
        "coffee": "../static/coffee.jpg",
        "cotton": "../static/cotton.jpg",
        "grapes": "../static/grapes.jpg",
        "jute": "../static/jute.jpg",
        "kidneybeans": "../static/kidneybeans",
        "lentil": "../static/lentil.jpg",
        "maize": "../static/maize.jpg",
        "mothbeans": "../static/mothbeans.jpg",
        "mungbeans": "../static/mungbeans.jpg",
        "muskmelon": "../static/muskmelon.jpg",
        "papaya": "../static/papaya.jpg",
        "pigeonpeas": "../static/pigeonpeas.jpg",
        "watermelon": "../static/watermelon",
        "blackgram": "../static/blackgram.jpg",

    }
    # # Connect to the SQLite database
    # conn = sqlite3.connect('your_database.db')
    # cursor = conn.cursor()
    #
    # try:
    #     # Execute a query to select commodity and modal price for the given district and crops
    #     query = f"SELECT Commodity, Modal_x0020_Price FROM your_table_name WHERE District = ? AND Commodity IN ({','.join(['?' for _ in crop_names])}) ORDER BY Modal_x0020_Price DESC"
    #     cursor.execute(query, (district, *crop_names))
    #
    #     # Fetch all rows from the result
    #     results = cursor.fetchall()
    #
    #     # Create a dictionary to map crop names to their prices
    #     crop_prices = {crop.lower(): price for crop, price in results}
    #
    #     # Display the sorted results
    #     sorted_crops = []
    #     for crop, _ in top_results:
    #         sorted_crops.append({"commodity": crop, "modal_price": crop_prices.get(crop.lower(), 'Not Available')})
    #
    #     # Fetch the user's prediction results from the database
    #     user = User.query.filter_by(email=session['email']).first()
    #     user_predictions = PredictionResult.query.filter_by(user_id=user.id).all()
    #
    #     return render_template('market.html', district=district, crop1=crop_names, crops=sorted_crops, crop_images=crop_images, user_predictions=user_predictions)
    #
    # except sqlite3.Error as e:
    #     print("Error executing query:", e)
    #     return "Error executing query"
    #
    # finally:
    #     # Close the connection
    #     conn.close()
    #
    # # Return a default response if the request method is not "POST"
    # return "Invalid request"
    # Get the selected district from the form
    district = request.form['district']
    top_results = session.get('top_results', None)
    crop_names = [crop for crop, _ in top_results]

    # Connect to the SQLite database
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()

    try:
        # Execute a query to select commodity and modal price for the given district and crops
        query = f"SELECT Commodity, Modal_x0020_Price FROM your_table_name WHERE District = ? AND Commodity IN ({','.join(['?' for _ in crop_names])}) ORDER BY Modal_x0020_Price DESC"
        cursor.execute(query, (district, *crop_names))

        # Fetch all rows from the result
        results = cursor.fetchall()

        # Create a dictionary to map crop names to their prices
        crop_prices = {crop.lower(): price for crop, price in results}

        # Display the sorted results
        sorted_crops = []
        for crop, _ in top_results:
            sorted_crops.append({"commodity": crop, "modal_price": crop_prices.get(crop.lower(), 'Not Available')})

        # Prepare data for the graph
        crop_labels = [crop['commodity'] for crop in sorted_crops]
        crop_prices = [crop['modal_price'] for crop in sorted_crops]

        # Plot the graph
        plt.figure(figsize=(10, 6))
        plt.bar(crop_labels, crop_prices, color='skyblue')
        plt.xlabel('Crop')
        plt.ylabel('Price (Rs)')
        plt.title('Crop Prices')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/crop_prices.png')  # Save the plot as an image
        plt.close()

        # Fetch the user's prediction results from the database
        user = User.query.filter_by(email=session['email']).first()
        user_predictions = PredictionResult.query.filter_by(user_id=user.id).all()

        return render_template('market.html', district=district, crop1=crop_names, crops=sorted_crops,
                               crop_images=crop_images, user_predictions=user_predictions)

    except sqlite3.Error as e:
        print("Error executing query:", e)
        return "Error executing query"

    finally:
        # Close the connection
        conn.close()


@app.route('/explore')
def explore():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        user_predictions = PredictionResult.query.filter_by(user_id=user.id).all()
        return render_template('explore.html', user_predictions=user_predictions)
    return redirect('/login')




@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)
