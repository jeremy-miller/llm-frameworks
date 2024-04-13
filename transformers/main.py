from datasets import Audio, load_dataset
from torch import nn

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

classifier = pipeline("sentiment-analysis")
results = classifier(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."]
)
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column(
    "audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate)
)
result = speech_recognizer(dataset[:4]["audio"])
print([d["text"] for d in result])


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print(classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers."))


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
pt_outputs = pt_model(**pt_batch)
pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)
