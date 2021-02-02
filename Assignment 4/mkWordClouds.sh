#!/usr/bin/env python
"""
 Derived from the minimal word cloud plotting example from 
 the word cloud GIT repo:

 https://github.com/amueller/word_cloud

 WordCloud by Andreas Mueller 
"""

from os import path
from wordcloud import WordCloud

d = path.dirname(__file__)

# Read the whole text.
for cc in range(5):
  name="centers_"+str(cc)+".txt"
  text = open(path.join(d, name)).read()

  # Generate a word cloud image
  wordcloud = WordCloud(collocations=False).generate(text)

  # Display the generated image:
  # the matplotlib way:
  import matplotlib.pyplot as plt

  # Display with unbound font size
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")

  # lower max_font_size
#  wordcloud = WordCloud(max_font_size=40).generate(text)
#  plt.figure()
#  plt.imshow(wordcloud, interpolation="bilinear")
#  plt.axis("off")

  plt.show()
