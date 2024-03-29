<style>
.blue {
    color: blue;
    font-weight: 400;
}
</style>
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

Let's begin looking at the bigrams in the dataset. Bigrams are just two characters in a row.

```python
# Sliding thru the words 
for w in words:
    for ch1, ch2 in zip(w, w[1:]):
        print(ch1, ch2)
```

the output should be something like this:

```sh
e m
m m
m a
o l
l i
i v
v i
i a
a v
...
```

Let me explain how this works. Take the word emma a look. It reads

```python
e m
m m
m a
```

The reason for this to work is that *w* = 'emma', and *w[1:]* = 'mma'. The zip function essentially takes two iterators and pairs them up, and iterates the tuples. If one list is shorter than the other, it would just be deleted. The tuples looks something like this:

```python
 w|w[1:]
(e, m)
(m, m)
(m, a)
# the 'a' in the w list gets removed
```

Just keep in mind that there isn't just **these** consecutive characters in our dataset; There are a lot more. However, just based on this first set of consecutive letters, we can tell that the letter *e* is likely to be the beginning of a name, and the letter *a* is likely to be the end of a name.

What we are going to do right now is to make a special array with special start and end tokens to indicate the beginning and end of a name. We are also going to wrap the <span class="blue">list</span>(w)

```python
for w in words:

    characters = ['<S>'] + list(w) + ['<E>']

    for ch1, ch2 in zip(w, w[1:]):
        print(ch1, ch2)
```
