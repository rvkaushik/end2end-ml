from flask import Flask, request, app, jsonify, url_for, render_template

from infer import predict

app = Flask(__name__)
model_path = "saved_model"

@app.route("/")
def home():
   return render_template('home.html')

@app.route('/predict_api', methods=["POST"])
def predict_api():
   data = request.json['data']
   image_path = data["image_path"]
   print(image_path)
   try:
      result = predict(model_path, image_path)
   except Exception as e:
      print(e)
      result="None"
   return str(result)

if __name__ == "__main__":
   app.run(debug=True)
