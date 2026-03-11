# Particularized Transformer

Augmenting the Transformer architecture with principles of lexicase selection in order to promote richer features and more precise long-range dependency handling.

## Motivation

What separates lexicase selection from standard average tournament selection in evolutionary algorithms is its tendency to select individuals that excel in one or two facets of overall performance rather than those who perform relatively well across all facets. It picks specialists rather than generalists. This preference of particularized over smooth allows the selection process to "pierce" the search space more effectively and find a solution more reliably and quickly.

The Transformer architecture benefits from the rich, dense feature representations that the self-attention mechanism promotes. However

- SEE IF IT'S MORE ADVERSARIALLY ROBUST
- TRY PIERCING WEIGHT ADJUSTMENT
   - Either make the gradients more sparse
   - Or consider small amounts of weights at a time
