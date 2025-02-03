---
layout: post
title:  "LLMs Need to Learn To Think"
date:   2024-12-04 12:00:00 +0800
last_modified_at: 2024-12-04 12:00:00 +0800
categories: [Statistical Modeling]
---
<br>

LLM performance has seemed to plateau as the hoovering of the internet nears completion. The prevailing line is that this is a sign that LLMs have advanced as far as they can advance. However, this might not be the case. Rather it is possible that LLMs know more than we think they do, but they haven't learned how to think before they speak. 

As far as one believes that humans convey their reasoning and knowledge through language, and as far as one believes that human language can be approximated by an arbitrarily complex statistical model, then we can say that there is a probability distribution over all possible sequences of words that humans are likely to generate.

$$p(x_0,...,x_i,...,x_n)$$

The weights inside ChatGPT encode correlations between words, sentences, and paragraphs. However, the billions of weights inside an LLM represent only a static set of variables that are not capable of generating a sequence on their own. Rather, a method is required to "decode" the weights inside the model into a generated sequence:

$$(x_0,...,x_i,...,x_n)$$

This is a tremendously hard problem. Consider a DNA strand. If we wanted to write down all possible ways to write a 100 base pair long DNA molecule, then at each position we have to include all 4 possible of ATCG. There are $4^100$ or $$1 \times 10^{60}$$ possible strings of A,T,C,G that are length 100. This is impossible even when there are only 4 choices at each position, in human language the vocabulary is immense.

Therefore, we need an approximation or estimation of the distribution $\hat{p}$ that minimizes the difference between natural language and our ability to approximate it

$$\min |p(x) - \hat{p}(x)|$$

Of course, in the era of ChatGPT and Claude, our approximation of language is given by the LLM that assigns a probability to each possible sequence.

$$\hat{p}(x_1,...,x_n) = f_{LLM}(x_1,...,x_n)$$


The problem encountered by modern LLMs is that succeeding in building very good likelihood estimators $f_{LLM}$, where $f$ represents an arbitrary language model, was only half the problem. The other half, as with DNA, is to guarantee the best response given input to the model. However, this requires checking all $M^n$ possible sequences, where $M$ is the number of tokens the model chooses at any point. 

The set of sequences is therefore distributed as defined by the LLM. 

$$x ~ f_{LLM}(x_0,...x_i,...,x_n)$$

To avoid having to sample more sequences than there are atoms in the universe, a method is needed to quickly sample a sequence that is likely to be near the maximum of $f_{LLM}$. To accomplish this, heuristic driven algorithms are necessary. The simplest and fastest approach is take the most likely token at each position independent of the rest, however this method is very susceptible to ending up in local minima, possibly very far from the true best scoring sequence. More complicated algorithms exist, such as beam search, however the number of samples taken is still miniscule and greedy heuristics are still used.

Users of ChatGPT, Claude, and other LLMs have noticed that the models now "think" before replying. It is unlikely that much or any formal logic has been encoded into the model. Rather, the companies are devising more elaborate sampling stragies for their LLMs. The startling implication is that these LLMs may actually know our language  better than we currently think, and that LLM reasoning may appear simply as a consequence of more efficiently obtaining better generated responses from the model. 

Given these limitations, current LLM performance may be no where near its true level of knowledge or approximation of human language. It is possible that improvements to LLMs will come from a team of mathematicians and computer scientists in a few lines of math, rather than relying on data. 

Our brains effortlessly sample language with essentially no error, especially on common knowledge subjects. The frequent and nonsensical "hallucinations" of an LLM are essentially a poor estimate for $\max{f_{LLM}}$ that a human brain would never make. 

As sampling algorithms for generative AI improves, we will get a better picture of the true level of knowledge store in modern LLMs. I strongly suspect that better exploration of the sequence space via improved markov-chain monte carlo methods or similar will result in improved AI performance despite the data ceiling.


