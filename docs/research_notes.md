# Research Notes

## Why Voronoi Seeds for AGI?

This document captures the deeper research motivation behind the
`voronoi_intelligence` library.

### The Seed as a Primitive

Most work with Voronoi diagrams in ML treats them as a passive
visualisation or clustering tool. We take a different view:

> **The seed (generator point) is the primitive — the tessellation
> is a consequence.**

By making seeds *controllable*, *evolvable*, and *learnable*, we can
use the Voronoi structure to:

1. **Initialise populations** with guaranteed spacing (no accidental
   clustering)
2. **Maintain diversity** without explicit niche-counting or
   fitness sharing parameters
3. **Adapt exploration** — larger cells get larger mutation steps
4. **Assign territories** in multi-agent systems with provable
   coverage guarantees
5. **Define neighbourhoods** via cell adjacency rather than
   distance thresholds

### Connection to AGI

Several AGI-relevant properties emerge from seed-based thinking:

- **Sparse representations**: Voronoi cells define a sparse,
  adaptive partitioning of representation space. Each cell can
  carry its own "expert" or "policy."
  
- **Compositionality**: Seeds can be exchanged, recombined, and
  mutated independently, making them natural units for
  evolutionary composition.

- **Emergent specialisation**: As seeds evolve toward
  high-fitness regions, the tessellation automatically
  concentrates resolution where it matters.

- **Multi-scale reasoning**: Seeds can operate at different
  scales (LoD — Level of Detail via Voronoi), enabling both
  coarse and fine-grained reasoning in the same framework.

- **Attention territories**: In a transformer-like architecture,
  each token could own a Voronoi territory in key space, and
  attention becomes "which territory does this query fall into?"

### Open Research Questions

- Can we learn seed positions via gradient descent through a
  differentiable Voronoi layer?
- How do Voronoi-enhanced EAs scale to 100+ dimensions?
- Can centroidal Voronoi tessellation (Lloyd's algorithm)
  be used as a "diversity regulariser" in neural network training?
- What happens when agents can trade or communicate their seeds?
- Can Voronoi territories serve as a spatial attention mechanism
  in LLM architectures?

### References

1. Cortés, J., Martinez, S., Karatas, T., & Bullo, F. (2004).
   "Coverage control for mobile sensing networks."
   *IEEE Transactions on Automatic Control*.

2. Deb, K., & Goldberg, D. E. (1989).
   "An investigation of niche and species formation in genetic
   fitness sharing." *Proceedings of the 3rd ICGA*.

3. Lehman, J., & Stanley, K. O. (2011).
   "Abandoning objectives: Evolution through the search for
   novelty alone." *Evolutionary Computation*.

4. Lloyd, S. (1982). "Least squares quantization in PCM."
   *IEEE Transactions on Information Theory*.

5. Okabe, A., Boots, B., Sugihara, K., & Chiu, S. N. (2000).
   *Spatial Tessellations: Concepts and Applications of Voronoi
   Diagrams*. Wiley.

6. Vassilev, V., & Miller, J. F. (2000). "Embedding Voronoi
   partitioning in evolutionary algorithms for continuous
   optimisation." *Proceedings of CEC 2000*.
