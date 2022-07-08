from flask import Flask, render_template, request
import pickle
import pandas

# Use pickle to load in the pre-trained model.
with open(f'models/SVM.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template("index.html")
    if request.method == 'POST':
        Nitrogen = request.form['Nitrogen']
        P = request.form['P']
        K = request.form['K']
        Temperature = request.form['Temperature']
        Humidity = request.form['Humidity']
        PH = request.form['PH']
        Rainfall = request.form['Rainfall']
        input = pandas.DataFrame(
            [[Nitrogen, P, K, Temperature, Humidity, PH, Rainfall]])
        prediction = model.predict(input)[0]
        print(prediction)
        return render_template("index.html", result=prediction)


if __name__ == '__main__':
    app.run(debug=True)
