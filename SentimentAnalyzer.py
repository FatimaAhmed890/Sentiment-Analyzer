"""
Task 1: Terminal
Task 2: Interpreter
Task 3: Variables
Task 4: Text Editor
Task 5: Functions
Task 6: Lists and Tuples
Task 7: Conditionals
Task 8: For loop
Task 9: User input & While loop
"""

# TASK 1, 2
import sys
print(sys.version)

import nltk  # python library -> Natural Language Toolkit
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from termcolor import colored
import matplotlib.pyplot as plt

print(colored("\n***Sentiment Analyzer using NLP***\n", 'red'))

nltk.download("vader_lexicon")  # pre-trained sentimental analysis model

# TASK 3, 6

texts = [
    "I am a positive piece of text.",
    "I am very very negative text.",
    "I am a neutral sentence.", 
    "Is fire on fire?",
    "The most introspective of hearts tends to be the most sentimental. We cling to the smallest moments from our past because we fear that emotion will never come our way again.",
    "Oz thinks I'm beautiful,she whispered to the stars.",
    "When the house is built I'll make a fiesta. I'll remember you, then"
]

# TASK 5, 7

def sentiment_analysis(text):
    sentiment = SentimentIntensityAnalyzer().polarity_scores(text)
    if sentiment['compound'] > 0.05:
        print(colored(f"The sentiment of '{text}' is {sentiment} with a score of {sentiment['compound']}\n", 'green'))
        return colored("Sentiment: Positive",'red', 'on_yellow', ['bold', 'underline'])
    elif sentiment['compound'] < 0.05:
        print(colored(f"The sentiment of '{text}' is {sentiment} with a score of {sentiment['compound']}\n", 'green'))
        return colored("Sentiment: Negative", 'red', 'on_yellow', ['bold', 'underline'])
    else:
        return colored("Sentiment: Neutral",'red', 'on_yellow', ['bold', 'underline'])

def plot(text):
    sentiment = SentimentIntensityAnalyzer().polarity_scores(text)
    return sentiment

sentiments = [plot(text) for text in texts]

positive = [sentiment for sentiment in sentiments if sentiment['compound'] >= 0.05]
negative = [sentiment for sentiment in sentiments if sentiment['compound'] <= 0.05]
neutral = [sentiment for sentiment in sentiments if sentiment['compound'] > -0.05 and sentiment['compound'] < 0.05]

# TASK 8, 9

while True:
    text = input("If you want to quit type 'q'.\nIf you want to see sentiment analysis on pre-stored text type 'p'.\nIf you want to give some text type 't'\n")
    if text == 'q':
        break
    elif text == 'p':
        for text in texts:
            print(sentiment_analysis(text), "\n")
    elif text == 't':
        text = input("Give some text for sentiment analysis: ")
        print(sentiment_analysis(text))

# Pie charts
chart_labels = ['Positive', 'Negative', 'Neutral']
sizes = [len(positive), len(negative), len(neutral)]

fig_1, axis_1 = plt.subplots()
axis_1.pie(sizes, labels=chart_labels, autopct='%1.1f%%', startangle=90)
axis_1.axis('equal')
plt.show()