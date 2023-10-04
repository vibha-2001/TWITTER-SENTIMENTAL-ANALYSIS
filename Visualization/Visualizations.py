import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load your dataset
dataset = pd.read_csv('../Dataset/Tweets.csv')

# Sentiment Distribution (Bar Chart)
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=dataset)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Text Length Distribution (Histogram)
plt.figure(figsize=(8, 6))
dataset['text_length'] = dataset['text'].apply(lambda x: len(str(x)))  # Calculate text length
sns.histplot(data=dataset, x='text_length', bins=30)
plt.title('Text Length Distribution')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Word Cloud for each sentiment class

sentiments = dataset['sentiment'].unique()
for sentiment in sentiments:
    plt.figure(figsize=(10, 8))
    text_for_wordcloud = ' '.join(dataset['text'].astype(str))  # Convert 'text' column to strings
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_for_wordcloud)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud of Text')
    plt.axis('off')
    plt.show()

# Sentiment vs. Text Length (Box Plot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='sentiment', y='text_length', data=dataset)
plt.title('Sentiment vs. Text Length')
plt.xlabel('Sentiment')
plt.ylabel('Text Length')
plt.xticks(rotation=45)
plt.show()


# Correlation Heatmap
correlation_matrix = dataset.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Selected Text vs. Text
# Create relevant plots comparing 'selected_text' and 'text' for different sentiments.

# Word Frequency (Bar Chart)
from collections import Counter


dataset['text'] = dataset['text'].astype(str)
words = ' '.join(dataset['text']).split()
word_counts = Counter(words)
common_words = word_counts.most_common(20)  # Get the 20 most common words
common_words_df = pd.DataFrame(common_words, columns=['Word', 'Count'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Word', data=common_words_df)
plt.title('Top 20 Most Common Words')
plt.xlabel('Count')
plt.ylabel('Word')
plt.show()

# Box Plots for Text Length (Grouped by Sentiment)
plt.figure(figsize=(10, 6))
sns.boxplot(x='sentiment', y='text_length', data=dataset)
plt.title('Text Length Distribution by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Text Length')
plt.xticks(rotation=45)
plt.show()

# Pair Plots (for numerical features)
# sns.pairplot(dataset[['text_length', 'word_count', 'sentiment_score']])
# plt.show()
