from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Weekday', 'Month', 'Year']

        input_data = [data[feature] for feature in features]

        x = np.array([input_data])

        prediction = model.predict(x)

        prediction = prediction.tolist()

        return jsonify({'prediction': prediction[0]})

    except KeyError as e:
        return jsonify({'error': f'Missing key in input data: {str(e)}'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

#from flask import Flask, request, jsonify
#import pickle
#import numpy as np

#app = Flask(__name__)

# Load the pre-trained model
#with open(r'C:\Users\clt\Downloads\Walmart Prediction\model.pkl', 'rb') as file:
#    model = pickle.load(file)

#@app.route('/predict', methods=['POST'])
#def predict():
#    try:
#        # Get the JSON data from the request
#        data = request.get_json()

        # Convert JSON data to a NumPy array
#        x = np.array(data)

        # Predict
#        prediction = model.predict(x)

        # Convert prediction to a serializable format
#        prediction = prediction.tolist()

#        return jsonify({'prediction': prediction[0]})

#    except Exception as e:
#        return jsonify({'error': str(e)})

#if __name__ == '__main__':
#    app.run(debug=True)

#@app.route('/',methods=['GET','POST'])
#def home():
#    if request.method =='POST':
#        st=request.form['Store']
#        hf = request.form['Holiday_Flag']
#        tmp = request.form['Temperature']
#        fp = request.form['Fuel_Price']
#        cpi = request.form['CPI']
#        unem = request.form['Unemployment']
#        wd = request.form['Weekday']
#        m = request.form['Month']
#        y = request.form['Year']
#        x = np.array([[st, hf, tmp, fp,cpi,unem,wd,m,y]])
#    return model.predict(x)


#if __name__ == '__main__':
#    app.run(debug=True)