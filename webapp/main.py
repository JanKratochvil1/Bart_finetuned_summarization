from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI, Response
from pydantic import BaseModel

# summary_min_length = 0
# summary_max_length = 150

# generator = pipeline('text-generation', model='gpt2')

tokenizer = AutoTokenizer.from_pretrained("knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM")
model = AutoModelForSeq2SeqLM.from_pretrained("knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM")

app = FastAPI()


class Body(BaseModel):
    text: str
    summary_min_length: int
    summary_max_length: int

@app.get('/')
def root():
    return Response("<h1>An API to interact with finetuned Bart model</h1>")


@app.post('/generate')
def predict(body: Body):
    tokens = tokenizer.encode(body.text, return_tensors='pt', truncation=True)
    ids = model.generate(tokens, min_length=body.summary_min_length, max_length=body.summary_max_length)
    summary = tokenizer.decode(ids[0], skip_special_tokens=True)
    return summary
