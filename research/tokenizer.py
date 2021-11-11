
import nltk
nltk.download('punkt')
data = "This is a first line. Now, this is the second line. Finaly, this is the last line."

nltk_sentence = nltk.sent_tokenize(data)
print(nltk_sentence)

nltk_word = nltk.word_tokenize(data)
print(nltk_word)

