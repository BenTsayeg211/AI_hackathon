from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
classifier = pipeline("ner", model=model, tokenizer=tokenizer)
classifier("Alya told Jasmine that Andrew could pay with cash..")

# MODEL = "facial_expression_classifier"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# m = AutoModelForTokenClassification.from_pretrained(MODEL, use_auth_token=True)
