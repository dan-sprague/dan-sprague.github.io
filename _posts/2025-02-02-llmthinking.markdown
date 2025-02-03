---
layout: post
title:  "LLMs Need To Think More"
date:   2025-02-02 12:00:00 +0800
last_modified_at: 2025-02-02 12:00:00 +0800
categories: [Statistical Modeling]
---
<br>

LLM performance has seemed to plateau as the hoovering of the internet nears completion. The prevailing line is that this is a sign that LLMs have advanced as far as they can advance. However, this might not be the case. Rather it is possible that LLMs know more than we think they do, but they haven't learned how to think before they speak. <br/>

As far as one believes that humans convey their reasoning and knowledge through language, and as far as one believes that human language can be approximated by an arbitrarily complex statistical model, then we can say that there is a probability distribution over all possible sequences of words that humans are likely to generate.

$$p(x_0,...,x_i,...,x_n)$$

The weights inside ChatGPT encode correlations between words, sentences, and paragraphs. However, the billions of weights inside an LLM represent only a static set of variables that are not capable of generating a sequence on their own. Rather, a method is required to "decode" the weights inside the model into a generated sequence given contextualizing information $c$. In other words, a method is required to roll the metaphorical dice to generate a sequence $x$ with probability $p$. 

$$x = (x_0,...,x_i,...,x_n)$$

$$ x \mid c \sim p(x) $$

This is a tremendously hard problem. Consider a DNA strand. To construct all possible 100 base pair long DNA molecules, then at each position there must be a subset of sequences that contain all possibilities of A,T,C, or G. For this reason, there are $4^{100}$ or $$1 \times 10^{60}$$ possible DNA strands that are length 100. Evaluating all $4^100$ molecules is impossible even when there are only 4 choices at each position.

Therefore, two things are necessary to generate biologically meaningful sequences from the soup of stochasicity. The first is a scoring function that attributes each sequence with a score, where the score encodes some notion of "better". In biology, the score is fitness -- a small number of possible DNA molecules improve an organisms chance of survival but the vast majority don't. The second is a method that can quickly find higher scoring, or "better", sequences from the soup of incredibly unlikely ones.

In context of human language, the score of a sequence of words $x$ is given by an LLM, such as ChatGPT or Claude. The LLM scoring function $f_\theta$ assigns a probability to each possible sequence out of the universe of all possible sequences $x \in X$. Therefore, an LLM is an approximating density for the set of sequences $x \in X$, conditional on the model parameters $\theta$ and input context $c$, $\hat{p}(x \mid \theta,c)$, that minimizes the difference between natural language and our ability to approximate it

$$p(x | c) \approx \hat{p}(x \mid \theta,c)$$

$$\hat{p}(x \mid \theta,c) = f_\theta(x)$$

Since $f_\theta$ calculates a probability for any sequence $x$, then the best response $x_{\texttt{best}}$ for a given context $c$ to the LLM and the model parameters $\theta$ is the the point of highest conditional density $p(x \mid c,\theta)$. This requires us to find the sequence $x_{\texttt{best}}$ that maximizes $f_\theta$.

$$(x_0,...,x_n)_{\texttt{best}} = \max f_\theta$$

The problem encountered by modern LLMs is that building a very good likelihood estimator $f_{LLM}$, where $f$ represents an arbitrary language model, was only half the problem. The other half, as with DNA, is to generate the best response given input to the model. This is equivalent to finding the maximum of the likelihood function $f_{LLM}$. However, this requires checking all $M^n$ possible sequences, where $M$ is the number of tokens the model chooses at any point. For human language, $M > 10^4$<br>

To avoid having to sample more sequences than there are atoms in the universe, a method is needed to quickly sample a sequence that is likely to be near the maximum of $f_{LLM}$. The simplest and fastest approach is take the most likely token at each position independent of the rest, however this method is very susceptible to ending up in local minima, possibly very far from the true best scoring sequence. More complicated algorithms exist, however the number of samples taken is still miniscule and greedy heuristics are still used.<br>

Users of ChatGPT, Claude, and other LLMs have noticed that the models now "think" before replying. It is unlikely that much or any formal logic has been encoded into the model. Rather, the companies are devising more elaborate sampling stragies for their LLMs. The startling implication is that these LLMs may actually know our language  better than we currently think they do, and that improved LLMs will emerge simply as a consequence of more efficiently obtaining better generated responses from the model.<br>

Given these limitations, current LLM performance may be no where near its true level of knowledge or approximation of human language. It is possible that improvements to LLMs will come from a team of mathematicians and computer scientists in a few lines of math, rather than relying on exponentially more data. <br>

Our brains effortlessly sample language with essentially no error, especially on common knowledge subjects. The frequent and nonsensical "hallucinations" of an LLM are essentially a poor estimate for $\max{f_{LLM}}$ that a human brain would never make. <br>

As sampling algorithms for generative AI improves, we will get a better picture of the true level of knowledge store in modern LLMs. I strongly suspect that better exploration of the sequence space via improved markov-chain monte carlo methods or similar will result in improved AI performance despite the data ceiling.


