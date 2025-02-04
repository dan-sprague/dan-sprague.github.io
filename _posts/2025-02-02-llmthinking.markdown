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
<br>

LLM performance has seemed to plateau as the hoovering of the internet nears completion. The prevailing line is that this is a sign that LLMs have advanced as far as they can advance. However, this might not be the case. Rather it is possible that LLMs know more than we think they do, but they haven't learned how to think before they speak. <br/>

# Language Models

As far as one believes that humans convey their reasoning and knowledge through language, and as far as one believes that human language can be approximated by an arbitrarily complex statistical model, then we can say that there is a probability distribution over all possible sequences of words $x = (x_1,\dots,x_n)$ that humans are likely to generate.

$$p(x)$$

Of course, the "true" nature of human language $p(x)$ is unknown and therefore requires mathematical approximation. Enter LLMs. The billions of weights inside an LLM encode correlations between words, sentences, and paragraphs. However, these weights represent only a static set of variables that are not capable of generating a sequence on their own. A method is required to "decode" the weights inside the model into a generated sequence given contextualizing information $c$, such as a prompt provided to the LLM from a human. In other words, a method is required to roll metaphorical dice to generate a sequence $x$ with probability $p$, given context $c$. 

$$ x \mid c \sim p(x) $$

# Estimating p(x) with an LLM

LLMs represent a transformational improvement in our ability to estimate the likelihood of human language. An LLM is a function $f$ parameterised by $\theta$, $f_\theta$, that calculates the likelihood^*^ of any sequence $x$. Therefore, an LLM is an approximating density for $p(x)$. To score possible generated sequences conditional on the model parameters $\theta$ and input context $c$, LLMs evaluate $\hat{p}(x \mid \theta,c)$, and attempt to sample a response that has the highest possible score. The best response $x_{\texttt{best}}$ is then the point of highest conditional density $\hat{p}(x \mid c,\theta)$. To guarantee that the best possible response from the model, $x_{\texttt{best}}$ must maximize $f_\theta$.

$$p(x | c) \approx \hat{p}(x \mid \theta,c)$$

$$\hat{p}(x \mid \theta,c) = f_\theta(x)$$

$$x_{\texttt{best}} = \max_{x} f_\theta(x)$$

The problem encountered by modern LLMs is that building a very good likelihood estimator was only half the problem. The other half, as with DNA, is to generate the best response given input to the model. This is equivalent to finding the maximum of the likelihood function $f_{LLM}$. However, this requires checking all $M^n$ possible sequences, where $M$ is the number of tokens the model chooses at any point. For human language, $M > 10^4$<br>

# Sequence generation

The number of possible ways to compose human language into text is functionally infinite, however even when the maximum length is restricted the problem is intractable. DNA sequences are a helpful analog here as the choices available at each position is only A,T,C, or G. Consider just 100 basepairs of DNA. To construct all possible 100 base pair DNA molecules, then at each position from 1 to 100 there must be a subset of sequences that contain all possibilities of A,T,C, and G. For this reason, there are $4^{100}$ or $$1 \times 10^{60}$$ possible DNA strands that are length 100. Evaluating all $4^{100}$ molecules is impossible.

To avoid having to evaluate more sequences than there are atoms in the universe, a method is needed to quickly sample a DNA sequence or set of DNA sequences that are likely to be near the maximum of $\hat{p}(x \mid c)$. For DNA, this means to sample sequences that maximize the fitness function for a given phenotype. In the case of LLMs, this means to scan the set of possible responses the model might reply to you with, and identify the response that, in a way, simply makes the most sense given your prompt.

Users of ChatGPT, Claude, and other LLMs have noticed that the models now "think" before replying. It is unlikely that much or any formal logic has been encoded into the model. Rather, the companies are devising more elaborate sampling stragies for their LLMs. The startling implication is that these LLMs may actually know our language  better than we currently think they do, and that improved LLMs will emerge simply as a consequence of more efficiently obtaining better generated responses from the model.<br>

![Finding the best generated response](/assets/images/path_opt.png)
| Figure 1. Gradient based generated sequence optimization. Left: Given an initial prediction from the model, the gradient of the LLM $\nabla f_\theta$ points the next prediction in a direction that is guaranteed to give a higher likelihood response, however these methods get trapped in local minima. Right: Gradient-based monte carlo samplers such as HMC use the gradient of the LLM $\nabla f_\theta$ to draw samples from $f_\theta$ proportionally to how likely the samples are from the model. |

<br><br>
Given these limitations, current LLM performance may be no where near its true level of knowledge or approximation of human language. It is possible that improvements to LLMs will come from a team of mathematicians and computer scientists in a few lines of math, rather than relying on exponentially more data. <br>

Our brains effortlessly sample language with essentially no error, especially on common knowledge subjects. The frequent and nonsensical "hallucinations" of an LLM are essentially a poor estimate for $\max{f_{LLM}}$ that a human brain would never make. <br>

As sampling algorithms for generative AI improves, we will get a better picture of the true level of knowledge store in modern LLMs. I strongly suspect that better exploration of the sequence space via improved markov-chain monte carlo methods or similar will result in improved AI performance despite the data ceiling.


