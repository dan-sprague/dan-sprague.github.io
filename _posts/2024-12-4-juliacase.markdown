---
layout: post
title:  "The Computational Biology Case for Julia"
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
Julia is a programming language that has a tremendous amount to offer scientific research and development. When people are introduced to Julia, the most common pitch is that it is fast. Indeed, it is fast. Julia is a just-in-time (JIT) compiled language, meaning that types can be defined or are inferred by the compiler and that the compiler can optimize code at run time. Julia is as fast as C while reading like Python. That being said – I think this undersells the language. <br><br>

In my experience, speed is often not a critical factor in scientific R&D programming development. For ML purposes, languages like Python have best in class libraries that get things done efficiently and correctly. This isn’t to say that there aren’t cases where speed really matters – I have developed Julia projects that reduced program run times from 30 minutes to less than a second and this was a substantial and meaningful result for the project. In most cases, speed is not a sufficient concern to warrant thinking about a new language. So then, why Julia?<br><br>


## Project Management and Reproducibility
<br>
In the sciences, reproducibility and organization are perhaps the most important component of good research. This is generally true for most applications – which is why languages like Rust and Julia have robust environment management built directly into the language as a core feature.<br><br>

Python has no native environment management and reproducibility functionality. To perform environment management in Python, one must download an environment manager like Anaconda. From here, it is still quite non-obvious to many new users such as graduate students how to correctly manage their research. When a new environment is created within Anaconda, it downloads a Python distribution each time and stores that information in an inscrutable location on the computer. For most, the workflow is to simply install packages as necessary into the base environment.<br><br>

Julia, in contrast, built package and environment management directly into the language. A new project in Julia has the following workflow: Create a project directory for any new project (Julia can do this for you, if desired), activate the project inside the Julia REPL (creating an environment), and add necessary packages to the project. This does a few things: first, a Project.toml and Manifest.toml file will be created for the project. These files manage dependencies automatically and in a way that is far more structured than a requirements.txt file, and second it will download all the dependencies to a common depot within the root Julia install.<br><br>

For a scientific paper or research project, the following approach organization is easy to obtain:<br>

- Project
    - Package # generalized code for entire project
        - Manifest.Toml
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
Much is made of Julia’s numerical chops (the language was built for it, after all), as well as Python’s world class numerical wrapper libraries (Pytorch, JAX, Tensorflow, numpy). However, computational biologists and bioinformaticians often work with text-based data. While speed may not often be limiting for many workflows, it does matter when processing HTS data. For Python users, it should go without saying that text-based computation is extremely slow. Often, this problem is so severe that multi-language pipelines become necessary. The underlying reason for this is that strings are generic objects in Python, and little to no optimization in performed at runtime.<br><br>

Because Julia is JIT compiled and has Chars as a first-class type, string data can be processed extremely quickly. For bioinformaticians, this has serious implications: rather than writing difficult to maintain code in C++ or Rust, it is possible to develop a short Julia program (with python-esque syntax) to analyze millions of biological sequences with speed that is comparable to C.<br><br>

Take a simple program to generate 100M short DNA sequences and check for palindromes. While DNA and RNA can and should be more efficiently represented (Julia has a package that implements efficient representations of DNA/RNA/Protein sequences), lets assume that we are simply optimizing for readability and implementation time.<br><br>

```python

import random
import time

def random_dna_sequence(n: int) -> str:
    letters = ["A", "T", "C", "G"]
    return ''.join(random.choice(letters) for _ in range(n))

def generate_dna_sequences(count: int) -> list:
    return [random_dna_sequence(random.randint(4, 10)) for _ in range(count)]
  
def is_palindrome(s: str) -> bool:
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

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

Sequence generation took `246.97s` and palindrome checking took `70.22s`.<br>
Here is a Julia implementation:<br><br>

```julia
function is_palindrome(s::String)::Bool
    cleaned = replace(lowercase(s), r"[^a-z]" => "")
    cleaned == reverse(cleaned)
end


function random_dna_sequence(n::Int)::String
    join(rand(['A','T','C','G'],n))
end


@time nucs = map(random_dna_sequence, rand(4:10,Int(1e8)))

@time result = is_palindrome.(nucs)
```

Sequence generation took `51.52s` and palindrome checking took `19.30s`. There are several things to note about the implementations here. The first is that Julia required no imports. Second, Julia's `@time` macro saves a tremendous amount of repetitious code. Third, the `is_palindrome` function can be broadcasted over the `nucs` vector with the `.` syntax. This is despite being a handrolled function, something which is not really feasible in Python. 

This calculation can be sped up even more without importing libraries:

```julia
@time result =  begin
    result = Vector{Bool}(undef, length(nucs))
    Threads.@threads for i in eachindex(nucs)
        result[i] = is_palindrome(nucs[i])
    end

    result
end
```

This expression (defined with the `begin ... end` syntax) evaluates in `6.44s`, resulting in an order of magnitude runtime decrease in overall evaluation. For most modern bioinformatics applications, the implications are substantial. C/C++/Java/Rust are not easy languages to develop in.

Finally, we can work with Julia's `Biosequences.jl` package. This implements 2-bit representations for DNA letters, and contains highly optimized routines for biosequences:

```julia
using BioSequences

@time seqs = [randseq(DNAAlphabet{2}(), rand(4:10)) for i in 1:1e8]
@time ispalindromic.(seqs) ## threaded loop is no faster due to overhead
```

These sequences are now generated in `14.15s` and palindrome checking is done in `2.35s`, in two lines! Python obviously has large repositories on its own, however even with exclusively native Julia code, we are able to get an order of magnitude decrease in run time.


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

## Built for Science
<br>
Writing mathematical and machine learning models is simple and intuitive in Julia. Broadcasting is built into the language and almost any expression on an iterable can be vectorized. Practically, this means that a tremendous amount of time is saved from having to incessantly call numpy/pytorch/jax functions on every single numerical operation that one wishes to perform.<br><br>

A more important point to note here is that in Python operations are limited to whatever numpy supports. Since Julia code is “Julia all the way down”, it is trivial to write new operations that will be performative.<br><br>



```julia
julia> x,y = [1,2,3],[4,5,6]
([1, 2, 3], [4, 5, 6])

julia> x .* y
3-element Vector{Int64}:
  4
 10
 18

julia> x' * y # apostrophe represents the transpose operation, making this an inner product calculation
32
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

In Julia, it is possible to design quite complex neural networks using native Julia code. In early project development, this often helps to quickly iterate on ideas.<br><br>

## Functions Are Center Stage
<br>
Most scientific research projects simply do not need the complexity and overhead introduced by OOP based programming. Julia is not an object-oriented language. Julia uses structs to create composite types and places an emphasis on functional programming design patterns (while not being a true functional language like Haskell). OOP is often unnecessary and even counterproductive to many scientific and machine learning applications.<br><br>



### Example: String Operations
<br>
A trivial example:<br><br>

```python

>>> x = "Hello world"
>>> x.upper()
'HELLO WORLD'
```

`upper()` is a method of the string class. For this reason, this function can only be applied to instances of `String`.<br><br>

Compared to Julia<br><br>

```julia
julia> x = "Hello world"
"Hello world"

julia> uppercase(x)
"HELLO WORLD"
```
<br><br>
In practice, this emphasis on functions and Julia's multiple dispatch system enable extremely portable code.<br><br>

```julia
abstract type Shape end

struct Circle <: Shape
    radius::Float64
end

struct Rectangle <: Shape
    width::Float64
    height::Float64
end

area(shape::Circle) = 3.14 * shape.radius^2
area(shape::Rectangle) = shape.width * shape.height
```
<br><br>
