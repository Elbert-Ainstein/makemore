# Makemore

As you know, we are going to make the Makemore model. Have fun reading and playing around!
This is lecture #1 of the 5-video series.

Credits: Andrej Karpathy

## About Makemore

- A character-level language model
- treats every line as an example, and for each example it's treated as a sequence of characters
- Example: molly => ['m', 'o', 'l', 'l', 'y']

## We are going to implement lots of language model neural nets. They include:

- Bigram
- Bag of Words
- Multi-Layer Perceptron (MLP)
- Recurrent Neural Networks (RNN)
- Gated Recurral Units (GRU)
- Transformer

Note: The Transformer that we are building will be almost identical to GPT-2. Yeah, it is very modern haha. 

- Jupyter Notebook provided.
- Andrej's Notebook is [here](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/makemore/makemore_part1_bigrams.ipynb)
  

## Let's get rollin' with some Code and more notes

### Reading, Exploring, and Playing around with the dataset

First thing to do, is to obviously load the dataset in. We want to load in the entire set in as a massive string, and then split them up into lists of individual words.

```python
# List of Strings
words = open('names.txt', 'r').read().splitlines()
```

So if we run the following,

```python
print(words[:10])
```

This is the result that we are going to get.

```python
['emma',
 'olivia',
 'ava',
 'isabella',
 'sophia',
 'charlotte',
 'mia',
 'amelia',
 'harper',
 'evelyn']
```

Just to check that the amount of words is correct, if we run

```python
print(len(word))
```

The result should be **32033**.

What if we want to get the shortest word? Well, we can run this:

```python
print(min(len(w) for w in words))
```

To get the shortest name in the list, which is 2. In the mean time, the longest name in the list

```python
print(max(len(w) for w in words))
```

should be 15.

**Now let's do a bit of thinking.**

We know that we are building a *character level language model*, which is that we take a bunch of characters and we predict the next character that comes in a sequence, given some sequence of characters before it. In a single word, for example using the word "isabella", we find out that it is actually some examples packed in a word. So, this word tells us that, 

1. the letter "i" is likely to be the first character of a name
2. the letter "s" is likely to be the after "i".
3. the letter "a" is likely to be come after "is".
4. the letter "s" is likely to be come after "isa".
5. ...
6. after the word "isabella" the word is likely to end.

In a word there is information provided into the statistical structure that determines which characters are at what position, and when the word ends. We have 32 thousand of these words. Loads of structures to model here :D

The first model that we would like to make is the Bigram Language Model. In the model, we are always just working with two characters at a time. We are given one character, and we are predicting the next character in the sequence following it. 

### Exploring Bigrams in a dataset


