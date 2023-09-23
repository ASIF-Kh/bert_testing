

from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Input prompt
# prompt = "What are the traits of a good text corpus or wordlist?"
prompt = "What is the primary use of a plain text corpus in machine learning?"

# Tokenize and encode the prompt
input_ids = tokenizer.encode(prompt, add_special_tokens=True)
input_tensor = torch.tensor([input_ids])

# Generate embeddings for the input prompt
with torch.no_grad():
    outputs = model(input_tensor)
    prompt_embedding = outputs.last_hidden_state.mean(dim=1)  # Average pooling over tokens

# Load your large text corpus and preprocess it
with open("refined_text_corpus.txt", 'r') as f:
    single_text_corpus = f.read()

text_corpus = [single_text_corpus[first:first+500] for first in range(0, len(single_text_corpus), 500)]

# Calculate similarity scores between the prompt and each sentence/paragraph
similarity_scores = []
for text_block in text_corpus:
    # Tokenize and encode the text block
    input_ids = tokenizer.encode(text_block, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])

    # Generate embeddings for the text block
    with torch.no_grad():
        outputs = model(input_tensor)
        text_block_embedding = outputs.last_hidden_state.mean(dim=1)

    # Calculate cosine similarity between prompt and text block embeddings
    similarity = cosine_similarity(prompt_embedding, text_block_embedding)
    similarity_scores.append(similarity[0][0])

# Rank text blocks by similarity scores
sorted_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)

# Select the top N relevant text blocks
top_n_blocks = [text_corpus[i] for i in sorted_indices[:5]]

# Post-processing and presenting the output
print(top_n_blocks)

# [
# 'luded in the corpus genre unless corpus has been collected for specific tasks it should include different genres such as newspapers magazines blogs academic journals etc size a corpus of half a million words or more ensures that low frequency words are also adequately represented clean a wordlist giving word forms of the same word can be messy to process a better corpus would include only the lemma and part of speech what are the different types of text corpora for nlp a plain text corpus is sui', 
# 'orpus or wordlist its said that a prototypical corpus must be machinereadable in unicode it must be a representative sample of the language in current use balanced and collected in natural settings a good corpus or wordlist must have the following traits depth a wordlist for instance should include the top k words and not just the top k words recent corpus based on outdated texts is not going to suit todays tasks metadata metadata should indicate the sources assumptions limitations and whats inc', 
# 'arking models typically each text corpus is a collection of text sources there are dozens of such corpora for a variety of nlp tasks this article ignores speech corpora and considers only those in text form while english has many corpora other natural languages too have their own corpora though not as extensive as those for english using modern techniques its possible to apply nlp on lowresource languages that is languages with limited text corpora discussion what are the traits of a good text c', 
# 'devopedia for developers by developers text corpus for nlp summary discussion what are the traits of a good text corpus or wordlist what are the different types of text corpora for nlp what are the types of annotations that we can have on a text corpus what are some nlp taskspecific training corpora could you list some nlp text corpora by genre what are some generic training corpora for nlp derived from text corpus which datasets are useful for nlp tasks which are some corpora for nonenglish lan', 
# 'ken language is often different from written language hub english is a dataset thats a transcription of telephone conversations signed language can also be annotated and transcribed to create a corpus since languages evolve when analyzing old text our models need to be trained likewise examples include doe corpus ss and coha ss another special case is of learners who are likely to express ideas differently the open cambridge learner corpus contains k student responses of million words its also c'
# ]