---
layout: post
title:  "The Case for Julia in Computational Biology"
date:   2024-12-04 12:00:00 +0800
last_modified_at: 2024-12-04 12:00:00 +0800
categories: [Julia]
---
<br>

## Table of Contents

- [Project Management and Reproducibility](#project-management-and-reproducibility)
- [Text Processing](#text-processing)
- [Fast and Easy Multithreading](#fast-and-easy-multithreading)
- [Functions Are Center Stage](#functions-are-center-stage)
<hr/>
<br>

Julia is a programming language that has a tremendous amount to offer scientific research and development. Python excels as a general purpose programming language and as a wrapper for neural network implementations, whereas R excels at data analysis, statistics, and its robust library of bioinformatics packages. What is missing from the computational biology toolkit is a language that is easy to write, read, while maximizing performance on numerical and sequence based data. This is the niche that Julia fills.<br><br>


## Native Project Management and Reproducibility
<br>
In the sciences, reproducibility and organization are perhaps the most important component of good research. This is generally true for most applications – which is why languages like Rust and Julia have robust environment management built directly into the language as a core feature.<br><br>

For a scientific paper or research project, the following organizational scheme is easy to obtain:<br>

- Project
    - Package # generalized code for entire project
        - Manifest.toml
        - Project.toml
        - src/
        - tests/
        - README.md 
        - ...
    - Figure 1
        - Manifest.toml
        - Project.toml
        - fig1.jl
        - data/
    - Figure 2
        - Manifest.toml
        - Project.toml
        - fig2.jl
        - data/
    - ... 

Perhaps most conveniently: no more remembering the name of all the environments you ever created and which project they map to! To truly compartmentalize one’s work in Python between projects, or even specific analyses within a project, would be practically difficult or impossible. In Julia: enter the project directory and launch Julia. The correct environment with all its dependencies will be loaded from there. The base Julia environment is kept clean.<br><br>


## Text Processing
<br>
Computational biologists and bioinformaticians often work with text-based data. Because Julia is JIT compiled and Chars are a first-class type, string data can be processed extremely quickly. BioSequences.jl has efficient minimal representations for biological characters that are intuitive and simple to operate on. For bioinformaticians, this has serious implications: rather than writing difficult to maintain code in C++ or Rust, it is possible to develop a short Julia program (with python-esque syntax) to analyze millions of biological sequences with speed that is comparable to C.<br><br>

### Example
<br>
Take a simple program to generate 100M short DNA sequences and check for palindromes. While DNA and RNA can and should be more efficiently represented (Julia has a package that implements efficient representations of DNA/RNA/Protein sequences), lets assume that we are simply optimizing for readability and implementation time.<br><br>

#### Python
```python

import random
import time


complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

def ceildiv(a, b):
    return -(a // -b)


def random_dna_sequence(n: int) -> str:
    letters = ["A", "T", "C", "G"]
    return ''.join(random.choice(letters) for _ in range(n))

def generate_dna_sequences(count: int) -> list:
    return [random_dna_sequence(random.randint(4, 10)) for _ in range(count)]

def is_palindrome(s: str) -> bool:
    if len(s) % 2 != 0:
        return False
    else:
        for i in range(ceildiv(len(s),2)):
            if s[i] != complement[s[len(s) - i - 1]]:
                return False
        return True 

# Generate 1 million random DNA strings
start = time.time()
dna_sequences = generate_dna_sequences(100_000_000)
end = time.time()
print("Sequence Generation Time: ",end - start)

start = time.time()
results = [is_palindrome(s) for s in dna_sequences]
end = time.time()
print("Palindrome check time: ",end - start)

```

Sequence generation took `246.97s` and palindrome checking took `12.87s`.<br>

#### Julia 

```julia

const complement = Dict('A'=>'T', 'T'=>'A', 'C'=>'G', 'G'=>'C');

function is_palindrome(s::Vector{Char})::Bool
    isodd(length(s)) && return false

    @inbounds for i in 1:cld(length(s),2)
        s[i] != complement[s[end-i+1]] && return false
    end

    true 
end



function random_dna_sequence(n::Int)::Vector{Char}
    rand(['A','T','C','G'],n)
end


@time nucs = map(random_dna_sequence, rand(4:10,Int(1e8)))

@time result = is_palindrome.(nucs)


```

Sequence generation took `21.52s` and palindrome checking took `1.82s`. There are several things to note about the implementations here. The first is that Julia required no imports. Second, Julia's `@time` macro saves a tremendous amount of repetitious code. Third, the `is_palindrome` function can be broadcasted over the `nucs` vector with the `.` syntax. This is despite being a handrolled function, something which is not really feasible in Python. Finally, these functions are so efficient that very little is gained very multithreading. Larger relative improvements in speed would be expected for more complex operations, such as sequence alignment and mapping.

## Fast and Easy Multithreading
<br>
Parallelism and broadcasting in Python are a major weakness of the language, and this is a problem because many if not most bioinformatics workflows are embarrassingly parallel. This is an area where Julia truly shines compared to Python, particularly for data-race free (embarrassingly parallel) situations. Combined with increased text-based processing speed and native numerical computation, this is when Julia really begins to shine. Last, Python’s global interpreter lock (GIL) prevents true multi-threading, putting the language at an immediate disadvantage.<br><br>

In Julia, threading over a loop is as simple as:<br><br>

```julia
using Base.Threads #brings Threads functions into the namespace, does not need to imported
  
n = 10_000_000
result = zeros(Float64, n)  
@threads for i in 1:n
  result[i] = sqrt(i)
end
```
<br><br>

## Elegant Machine Learning
<br>
Julia’s syntax, large scientific and numerical ecosystem, and native support results in elegant code for statistical modeling and machine learning that does not depend on DSLs. Below is a simple program that uses two of Julia’s most powerful packages to yield a MLE for a Gamma distributed sample in only a few lines of code. <br><br>


```julia
using Distributions
using Optim  

function f(α,β,x) # likelihood function
  -sum(logpdf(Gamma(exp(α),exp(β)), x))
end
 
x = rand(Gamma(2.0,1.5),500) # data  
θ_init = [1.0,1.0]
θ_mle= Optim.optimize(θ -> f(θ...,x), θ_init) # … is the “splat” operator in Julia 
```
<br><br>
Line 11 creates a closure such that the optimize routine only performs optimization on the estimated parameters captured in the theta vector. This is a common problem in likelihood functions, as both parameters and data are necessary arguments. Julia treats functions as first-class types, making anonymous functions, closures, and other functional programming paradigms a natural part of the language.<br><br>

In Julia, it is possible to design custom networks using native Julia code. In early project development, this often helps to quickly iterate on ideas. Significant time is saved from ensuring you understand the behavior of every function in Pytorch/Jax/Tensorflow, which requires substantial upfront time investment.<br><br>
