---
layout: post
title:  "How LLMs Think"
date:   2025-02-02 12:00:00 +0800
last_modified_at: 2025-02-02 12:00:00 +0800
categories: [Statistical Modeling]
---
# Table of Contents
- [Language Models](#language-models)
- [LLMs Are Approximating Densities of Natural Language](#llms-are-approximating-densities-of-natural-language)
<hr>
<br><br>

LLM performance has seemed to plateau as the hoovering of the internet nears completion. The prevailing line is that this is a sign that LLMs have advanced as far as they can advance. However, this might not be the case. Rather it is possible that LLMs know more than we think they do, but they haven't learned how to think before they speak. <br/>

# Language Models

As far as one believes that humans convey their reasoning and knowledge through language, and as far as one believes that human language can be approximated by an arbitrarily complex statistical model, then we can say that there is a probability distribution over all possible sequences of words that humans are likely to generate.

$$p(x)$$

where $x$ is an arbitrary sequence of words $x = (x_1,...,x_n)$.

Of course, the "true" nature of $p(x)$ is unknown and therefore requires mathematical approximation. Enter LLMs. The billions of weights inside an LLM encode correlations between words, sentences, and paragraphs. However, these weights represent only a static set of variables that are not capable of generating a sequence on their own. A method is required to "decode" the weights inside the model into a generated sequence given contextualizing information $c$, such as a prompt provided to the LLM from a human. In other words, a method is required to roll the metaphorical dice to generate a sequence $x$ with probability $p$, given context $c$. 

$$ x \mid c \sim p(x) $$

## Combinatorial Explosion

Unlike rolling a die, sampling the set of all possible sequences is a tremendous problem. Consider a DNA strand. To construct all possible 100 base pair long DNA molecules, then at each position there must be a subset of sequences that contain all possibilities of A,T,C, or G. For this reason, there are $4^{100}$ or $$1 \times 10^{60}$$ possible DNA strands that are length 100. Evaluating all $4^{100}$ molecules is impossible.

To work around this problem, two things are necessary to generate/sample/identify biologically meaningful sequences. The first is a language model that attributes each sequence with a score, where the score encodes some notion of "better". In biology, the score is fitness -- a small number of possible DNA molecules improve an organisms chance of survival but the vast majority don't. The second is a method that can quickly find higher scoring, or "better", sequences from the soup of meaningless ones.

# LLMs Are Approximating Densities of Natural Language

LLMs represent a transformational improvement in our ability to score generated human language. An LLM implemented as a neural network $f_\theta$ assigns a probability to each possible sequence of words out of the universe of all possible sequences $x \in X$. Therefore, an LLM is an approximating density for $p(x)$. To score possible generated sequences conditional on the model parameters $\theta$ and input context $c$, LLMs evaluate $\hat{p}(x \mid \theta,c)$, and attempt to sample a response that has the highest possible score. The best response $x_{\texttt{best}}$ is then the point of highest conditional density $\hat{p}(x \mid c,\theta)$. To guarantee that the best possible response from the model, $x_{\texttt{best}}$ must maximize $f_\theta$.

$$p(x | c) \approx \hat{p}(x \mid \theta,c)$$

$$\hat{p}(x \mid \theta,c) = f_\theta(x)$$

$$x_{\texttt{best}} = \max_{x} f_\theta(x)$$

The problem encountered by modern LLMs is that building a very good likelihood estimator $f_{LLM}$, where $f$ represents an arbitrary language model, was only half the problem. The other half, as with DNA, is to generate the best response given input to the model. This is equivalent to finding the maximum of the likelihood function $f_{LLM}$. However, this requires checking all $M^n$ possible sequences, where $M$ is the number of tokens the model chooses at any point. For human language, $M > 10^4$<br>

To avoid having to sample more sequences than there are atoms in the universe, a method is needed to quickly sample a sequence that is likely to be near the maximum of $f_{LLM}$. The simplest and fastest approach is take the most likely token at each position independent of the rest, however this method is very susceptible to ending up in local minima, possibly very far from the true best scoring sequence. More complicated algorithms exist, however the number of samples taken is still miniscule and greedy heuristics are still used.<br>

Users of ChatGPT, Claude, and other LLMs have noticed that the models now "think" before replying. It is unlikely that much or any formal logic has been encoded into the model. Rather, the companies are devising more elaborate sampling stragies for their LLMs. The startling implication is that these LLMs may actually know our language  better than we currently think they do, and that improved LLMs will emerge simply as a consequence of more efficiently obtaining better generated responses from the model.<br>

Given these limitations, current LLM performance may be no where near its true level of knowledge or approximation of human language. It is possible that improvements to LLMs will come from a team of mathematicians and computer scientists in a few lines of math, rather than relying on exponentially more data. <br>

Our brains effortlessly sample language with essentially no error, especially on common knowledge subjects. The frequent and nonsensical "hallucinations" of an LLM are essentially a poor estimate for $\max{f_{LLM}}$ that a human brain would never make. <br>

As sampling algorithms for generative AI improves, we will get a better picture of the true level of knowledge store in modern LLMs. I strongly suspect that better exploration of the sequence space via improved markov-chain monte carlo methods or similar will result in improved AI performance despite the data ceiling.


