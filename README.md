# Let's Model It Out

An implementation-driven blog about building mathematical and computational models from first principles.

The main series, **Bayes Through Physics**, develops probability as energy, Bayesian inference as physical dynamics, information geometry, least-action transport, online learning, and GFlowNets.

## Local development

Requirements:

- Julia 1.12 or compatible
- Quarto 1.7 or newer

Instantiate the Julia environment:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Preview the site:

```sh
./scripts/quarto preview
```

The wrapper uses a system Quarto installation when available and otherwise uses the portable copy in `.tools/quarto`.

The site uses Quarto's native Julia engine through `QuartoNotebookRunner.jl`; it does not require Jupyter or Python.

Planning documents live in [`planning/`](planning/).
