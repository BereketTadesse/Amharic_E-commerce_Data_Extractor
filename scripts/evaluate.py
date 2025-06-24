from transformers import pipeline

ner_pipeline = pipeline("ner", model="models/checkpoints/", aggregation_strategy="simple")

text = "አዲስ ጫማ በ1500ብር በአዲስ አበባ ላይ"
results = ner_pipeline(text)

for entity in results:
    print(f"{entity['word']} - {entity['entity_group']} ({entity['score']:.2f})")
