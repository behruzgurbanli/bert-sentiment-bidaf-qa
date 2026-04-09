# Project 4 Task 1 Summary

- Dataset: IMDb Large Movie Review Dataset
- Model: distilbert-base-uncased-finetuned-sst-2-english
- Train reviews: 25000
- Test reviews: 25000
- Classes: 2
- Max input tokens: 512
- Case sensitive: False
- Runtime model loaded: False

## Key Points
- Input: Single review text string
- Output: Predicted sentiment label
- IMDb labels: neg, pos
- IMDb is a binary sentiment dataset, so a binary sequence-classification BERT model is the cleanest fit for demos, evaluation, and report discussion.
- The English IMDb setup is not sufficient evidence for Azerbaijani performance. Azerbaijani is agglutinative, so subword fragmentation and domain mismatch can hurt results unless the model is multilingual and additionally evaluated or fine-tuned on relevant data.
