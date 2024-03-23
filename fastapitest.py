from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
# Define a request body model
class Sentence(BaseModel):
    text: str
# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
app = FastAPI()
@app.post("/predict/")
async def predict(sentence: Sentence):
    try:
        # Transform the input sentence using the loaded vectorizer
        transformed_sentence = vectorizer.transform([sentence.text])
        # Predict using the loaded model
        prediction = model.predict(transformed_sentence)
        # Return the prediction
        return {"prediction": prediction[0]}
    except Exception as e:
        # If an error occurs, return an HTTPException
        raise HTTPException(status_code=500, detail=str(e))