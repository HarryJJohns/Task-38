import spacy

# ---- Simpler language model ----#
#nlp = spacy.load('en_core_web_sm')

# No word vectors - warning about similarity method being based on taggers, parsers and NER. 'may not give useful
# similarity judgements' The sm stands for small. It is possible to add your own word vectors but the other
# models include vectors. The similarities kind of work but more abstract associations don't seem to register as well.


nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1, word2)
print(word1.similarity(word2))
print(word3, word2)
print(word3.similarity(word2))
print(word3, word1)
print(word3.similarity(word1))

# Interesting that Banana and monkey have a higher similarity than banana and cat.
# Shows knowledge that monkeys eat bananas and it's not just things that are catahgorised the same that
# have recognisable similarity.

# my eg:
w1 = nlp("Orange")
w2 = nlp("Lime")
w3 = nlp ("Soda")

print(w1, w2)
print(w1.similarity(w2))
print(w3, w2)
print(w3.similarity(w2))
print(w3, w1)
print(w3.similarity(w1))
# Orange and lime have the highest similarity as they were each fruit. Wanted to see if there was
# an association with lime and soda - it returns nearly double soda orange so it looks like there is

# Just curious about orange yellow to see how similarity compares to orange lime
w4 = nlp("Yellow")
print(w1, w4)
print(w1.similarity(w4))
# Associated more than Orange lime but less than Lime soda.

tokens = nlp("cat apple monkey banana")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text,token1.similarity(token2))


sentence_to_compare = "Why is my cat in the car"

sentences = ["Where did my dog go",
"Hello, there is my car",
"I've lost my car",
"I'd like my boat back",
"I will name my dog Hugo"]

model_sentence = nlp(sentence_to_compare)

print("\n----------")
print(f"Comparison sentence:\n{sentence_to_compare}\n----------")
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)