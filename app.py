import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
#from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(ngram_range=(1,3), 
#                              min_df=0.001, 
#                              max_df=0.7, 
#                              analyzer='word')
lr = joblib.load('model.pkl')
vr = joblib.load('vectorizer.pkl')
print(lr.predict(vr.transform(['do you know about weather?'])))
question_types = ["whQuestion","ynQuestion"]
def is_ques_using_nltk(ques):
    # question_type = classifier.classify(dialogue_act_features(ques))
    question_type = lr.predict(vr.transform([ques]))
    return question_type in question_types


question_pattern = ["do i", "do you", "what", "who", "is it", "why","would you", "how","is there","could you",
                    "are there", "is it so", "is this true" ,"to know", "is that true", "are we", "am i", 
                   "question is", "tell me more", "can i", "can we", "tell me", "can you explain",
                   "question","answer", "questions", "answers", "ask"]

helping_verbs = ["is","am","can", "are", "do", "does"]

def is_question(question):
    question = question.lower().strip()
    if not is_ques_using_nltk(question):
        is_ques = False
        # check if any of pattern exist in sentence
        for pattern in question_pattern:
            is_ques  = pattern in question
            if is_ques:
                break

        # there could be multiple sentences so divide the sentence
        sentence_arr = question.split(".")
        for sentence in sentence_arr:
            if len(sentence.strip()):
                # if question ends with ? or start with any helping verb
                # word_tokenize will strip by default
                first_word = nltk.word_tokenize(sentence)[0]
                if sentence.endswith("?") or first_word in helping_verbs:
                    is_ques = True
                    break
        return is_ques    
    else:
        return True

# q1='what is about weather?'
# print(is_question(q1))


class Sentence(BaseModel):
    text: str
# Load the model and vectorizer
# model = joblib.load('model.pkl')
# vectorizer = joblib.load('vectorizer.pkl')
app = FastAPI()
@app.post("/predict/")
async def predict(sentence: Sentence):
    try:
        # Transform the input sentence using the loaded vectorizer
        # transformed_sentence = vectorizer.transform([sentence.text])
        # # Predict using the loaded model
        # prediction = model.predict(transformed_sentence)
        # Return the prediction
        prediction=is_question(sentence.text)
        return {"prediction": prediction}
    except Exception as e:
        # If an error occurs, return an HTTPException
        raise HTTPException(status_code=500, detail=str(e))