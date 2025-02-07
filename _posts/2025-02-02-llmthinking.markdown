---
layout: post
title:  "Thinking Machines"
date:   2025-02-02 12:00:00 +0800
last_modified_at: 2025-02-02 12:00:00 +0800
categories: [Statistical Modeling]
---
## Table of Contents
- [Language Models](#language-models)
- [LLMs Are Approximating Densities of Natural Language](#llms-are-approximating-densities-of-natural-language)
<hr>
<br>

LLM performance has seemed to plateau as the hoovering of the internet nears completion. The prevailing belief is that this is because of fundamental limitations in the architecture of transformers and the availability of real human language for training data. It is likely if not probable that transformers represent a fundamental limitation to the performance of LLMs, yet that does not mean that there is not room for improvement in current text generation. Compared to the intricacy of the transformer models themselves, the methods used to sample responses to prompts are simple and heuristic driven. The reason for this is simple: each time a LLM chooses a new word in generated text, the model has to consider which of the $>10^4$ possible words is the most likely. 

A class of highly efficient algorithms have been developed for efficient sampling of complex functions: gradient based Markov Chain Monte Carlo (MCMC) methods. The challenges are numerous, as LLMs are high dimensional with a complex likelihood, and sampling sequences is a discrete and non-differentiable operation. The obstacles can perhaps be overcome with recently developed methodologies. Last, while gradient MCMC is extremely efficient at mapping functions, it is expensive and time consuming. While common use cases may not require such sampling, entities like businesses and governments may value obtaining the best response from an LLM so highly that they have no problem waiting hours for a response if such a sampling approach yields meaningful and actionable improvement.

## Language Models

As far as one believes that humans convey their reasoning and knowledge through language, and as far as one believes that human language can be approximated by an arbitrarily complex statistical model, then we can say that there is a probability distribution over all possible sequences of words $x = (x_1,\dots,x_n)$ that humans are likely to generate.

$$p(x)$$

Of course, the "true" nature of human language $p(x)$ is unknown and therefore requires mathematical approximation. Enter LLMs. The billions of weights inside an LLM encode correlations between words, sentences, and paragraphs. However, these weights represent only a static set of variables that are not capable of generating a sequence on their own. A method is required to "decode" the weights inside the model into a generated sequence given contextualizing information $c$, such as a prompt provided to the LLM from a human. In other words, a method is required to roll metaphorical dice to generate a sequence $x$ with probability $p$, given context $c$. 

$$ x \mid c \sim p(x) $$

The challenge LLMs tackle is twofold: estimating $p(x)$ and then sampling (generating) a reply to a prompt $c$ that maximizes $p(x \mid c)$. 

## Estimating p(x) with an LLM

LLMs represent a transformational improvement in our ability to estimate the likelihood of human language. An LLM is a function $f$ parameterised by $\theta$, $f_\theta$, that consumes context $c$ and estimates the probability distribution over the vocabulary position $i$ within the sequence. For this reason, an LLM can be viewed as an approximating density for $p(x \mid c) \approx \hat{p}(x \mid \theta,c)$. The best response $x_{\texttt{best}}$ from the LLM is then the point of highest conditional density $\hat{p}(x \mid c,\theta)$, which directly means that $x_{\texttt{best}}$ must maximize $f_\theta$.

$$p(x \mid c) \approx \hat{p}(x \mid \theta,c)$$

$$\hat{p}(x \mid \theta,c) = f_\theta(x)$$

$$x_{\text{best}} = \underset{x}{\operatorname{argmax}} f_\theta(x)$$

The problem encountered by modern LLMs is that building a very good likelihood estimator was only half the problem. The other half, as with DNA, is to generate the best response given input to the model. This is equivalent to finding the maximum of the likelihood function $f_{LLM}$. However, this requires checking all $M^n$ possible sequences, where $M$ is the number of tokens the model chooses at any point. For human language, $M > 10^4$<br>

## Peak Finding 

<img src="/assets/images/everest_srt_dem_lrg.jpg" height="200" alt="description">
| Figure 1. Zoomed in area around Mt. Everest in the Himalayas.|
<br><br>

Imagine you have decided to drop yourself at a random location in the middle of Himalayas. Your goal is to find the highest peak, Everest, by only walking upwards. Crucially, once any peak in the mountain range is reached, you can no longer climb upwards so the task must complete. There are 3,411 named peaks in the Himalayas, of which only 1 is Mt. Everest. Clearly it is quite improbable to find Mt. Everest using this algorithm. This algorithm is known as gradient ascent, and it is related to how LLMs are trained.

Now imagine that you once again find yourself in the middle of the Himalayas. Rather than only climbing until a peak is reached, you continuously move around in a way that respects these rules:

1. As you move uphill, you lose momentum and tend to make smaller moves
2. As you move downhill, you gain momentum and start making larger moves
3. In flat regions, your momentum doesn't change and you make large moves until you either start climbing again (1) or descending again (2)

If this is done with an element of randomness, it will be the case that you end up visiting peaks in the Himalayas proportionally to how high they are -- virtually ensuring that you will eventually find the peak of Mt. Everest. 

To avoid having to evaluate more sequences than there are atoms in the universe, a method is needed to quickly sample a DNA sequence or set of DNA sequences that are likely to be near the maximum of $\hat{p}(x \mid c)$. For DNA, this means to sample sequences that maximize the fitness function for a given phenotype. In the case of LLMs, this means to scan the set of possible responses the model might reply to you with, and identify the response that, in a way, simply makes the most sense given your prompt.

This is a very challenging task within the space of human language. To practically solve this, LLMs have employed heuristics. These heuristics translate into large, rapid, and aggresive moves in the function space $f_\theta$, leading the model to find a decent but suboptimal reply to a prompt. It is possible that some hallucations are simply the product of a poor model estimate of $\max_{x} f_\theta(x)$.

Crucially, these heuristics work pretty well because the underlying models are so large and have acquired so much data. However, as LLMs are asked increasingly nuanced questions and particularly questions that may require synthesis of unseen connections in the training data, methods that make more intelligent samples will generate better responses.


## Thinking like an LLM
While the generated response from heuristics might be suboptimal, the response is very probably now in the neighborhood of the best response. Fortunately, there is more information available that guarantees a better prediction on a second attempt. Because an LLM is a neural network that has been trained with gradient descent, it has a calculable gradient $\nabla f_{\theta}$. The gradient points in the direction of change in $f_\theta$, essentially indicating how to generate <br>

Gradient Ascent<br>

$x_{i+1} \leftarrow x_i + η ∇f_θ(x_i)$

Gradient MCMC Samplers (HMC)<br>

$p_{i+1/2} \leftarrow p_i + (η/2) ∇f_θ(x_i)$

$x_{i+1} \leftarrow x_i + η p_{i+1/2}$

$p_{i+1} \leftarrow p_{i+1/2} + (η/2) ∇f_θ(x_{i+1})$

![Finding the best generated response](/assets/images/path_opt.png)
| Figure 2. Gradient based generated sequence optimization. Left: Given an initial prediction from the model, the gradient of the LLM $\nabla f_\theta$ points the next prediction in a direction that is guaranteed to give a higher likelihood response, however these methods get trapped in local minima. Right: Gradient-based monte carlo samplers such as HMC use the gradient of the LLM $\nabla f_\theta$ to draw samples from $f_\theta$ proportionally to how likely the samples are from the model. |




<br><br>
Given these limitations, current LLM performance may be no where near its true level of knowledge or approximation of human language. It is possible that improvements to LLMs will come from a team of mathematicians and computer scientists in a few lines of math, rather than relying on exponentially more data. <br>

Our brains effortlessly sample language with essentially no error, especially on common knowledge subjects. The frequent and nonsensical "hallucinations" of an LLM are essentially a poor estimate for $\max{f_{LLM}}$ that a human brain would never make. <br>

As sampling algorithms for generative AI improves, we will get a better picture of the true level of knowledge store in modern LLMs. I strongly suspect that better exploration of the sequence space via improved markov-chain monte carlo methods or similar will result in improved AI performance despite the data ceiling.


## Notes

LLMs do not estimate the true likelihood of a sequence but rather the pseudolikelihood.

You generate an initial proposal response, but then reevaluate the response. You pass the initial response through the LLM which gives a probability/pseudolikelihood of the response. The calculation of that probability/pseudolikelihood is differentiable. 

However my sampling choices at each position in the sequence are discrete, meaning they cannot be connected to this gradient. This is resolvable, potentially, by using a continuous relaxation calculation of the samples.


Initial → Continuous Relaxation → Gradient Updates → Final Argmax
tokens    (Gumbel-Softmax)       (using f_θ)        (discrete tokens)

From yang song 


When sampling with Langevin dynamics, our initial sample is highly likely in low density regions when data reside in a high dimensional space. Therefore, having an inaccurate score-based model will derail Langevin dynamics from the very beginning of the procedure, preventing it from generating high quality samples that are representative of the data.

^ We solve this maybe by doing a greedy beam search then optimize