# Bibliography: Foundations for AI and AIDEEN

This bibliography is organized with one purpose in mind:

- build a mathematically serious foundation,
- move from university-level mathematics into AI,
- and then use that foundation to understand AIDEEN.

This is not a list of random links.
It is a structured reading base.

## 1. Core Mathematical Foundations

### 1.1 Linear Algebra

1. **Gilbert Strang — MIT OCW Linear Algebra**
   - Link: [https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
   - Why it matters:
     this is one of the clearest full university courses for building real linear algebra intuition.
   - Use it for:
     vectors, matrices, systems of equations, subspaces, eigenvalues, orthogonality.

2. **Gilbert Strang — MIT OCW Scholar Linear Algebra**
   - Link: [https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/)
   - Why it matters:
     stronger self-study structure, problem-solving orientation.

3. **Mathematics for Machine Learning**
   - Link: [https://mml-book.github.io/](https://mml-book.github.io/)
   - Why it matters:
     connects linear algebra directly to machine learning applications.

### 1.2 Calculus

1. **MIT OCW Single Variable Calculus**
   - Link: [https://ocw.mit.edu/courses/mathematics/18-01sc-single-variable-calculus-fall-2010/](https://ocw.mit.edu/courses/mathematics/18-01sc-single-variable-calculus-fall-2010/)
   - Use it for:
     derivatives, slope, optimization intuition, chain rule.

2. **MIT OCW Calculus I**
   - Link: [https://ocw.mit.edu/courses/18-01-calculus-i-single-variable-calculus-fall-2020/](https://ocw.mit.edu/courses/18-01-calculus-i-single-variable-calculus-fall-2020/)

3. **Mathematics for Machine Learning**
   - Link: [https://mml-book.github.io/](https://mml-book.github.io/)
   - Use it for:
     multivariable derivatives and ML-motivated calculus.

### 1.3 Probability and Statistics

1. **Harvard Stat 110**
   - Link: [https://projects.iq.harvard.edu/stat110](https://projects.iq.harvard.edu/stat110)
   - Why it matters:
     a strong university-level probability course with excellent explanations.

2. **Deep Learning Book — Probability and Information Theory**
   - Link: [https://www.deeplearningbook.org/contents/prob.html](https://www.deeplearningbook.org/contents/prob.html)
   - Why it matters:
     helps bridge probability into cross-entropy, entropy, and modeling.

### 1.4 Optimization

1. **Convex Optimization — Boyd and Vandenberghe**
   - Link: [https://web.stanford.edu/~boyd/cvxbook/](https://web.stanford.edu/~boyd/cvxbook/)
   - Why it matters:
     the reference for optimization thinking.

2. **Dive into Deep Learning — Optimization**
   - Link: [https://d2l.ai/chapter_optimization/optimization-intro.html](https://d2l.ai/chapter_optimization/optimization-intro.html)
   - Why it matters:
     practical ML-oriented entry point.

## 2. Machine Learning Foundations

1. **Stanford CS229**
   - Link: [https://cs229.stanford.edu/materials.html](https://cs229.stanford.edu/materials.html)
   - Why it matters:
     one of the cleanest academic foundations for machine learning.

2. **Deep Learning — Goodfellow, Bengio, Courville**
   - Link: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
   - Why it matters:
     canonical deep learning reference.

3. **Dive into Deep Learning**
   - Link: [https://d2l.ai/](https://d2l.ai/)
   - Why it matters:
     practical complement to the more theoretical texts.

## 3. Neural Network Mechanics

1. **CS231n Notes**
   - Main site: [https://cs231n.github.io/](https://cs231n.github.io/)
   - Neural Nets Part 1: [https://cs231n.github.io/neural-networks-1/](https://cs231n.github.io/neural-networks-1/)
   - Backpropagation / Optimization: [https://cs231n.github.io/optimization-2/](https://cs231n.github.io/optimization-2/)
   - Why it matters:
     extremely clear explanations of backprop, gradients, and shape reasoning.

2. **Matrix Calculus for Deep Learning**
   - Link: [https://arxiv.org/abs/1802.01528](https://arxiv.org/abs/1802.01528)
   - Why it matters:
     direct bridge between derivatives and neural network notation.

3. **PyTorch Autograd Tutorial**
   - Link: [https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
   - Why it matters:
     practical grounding for how gradients are handled in actual code.

## 4. Sequence Models and Modern AI Architectures

1. **Attention Is All You Need**
   - Link: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
   - Why it matters:
     baseline transformer reference.

2. **The Annotated Transformer**
   - Link: [https://nlp.seas.harvard.edu/2018/04/03/attention.html](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
   - Why it matters:
     one of the best practical explanations of transformer internals.

3. **Stanford CS224N**
   - Link: [https://web.stanford.edu/class/cs224n/](https://web.stanford.edu/class/cs224n/)
   - Why it matters:
     natural language processing and language-modeling foundation.

4. **Mamba**
   - Link: [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)
   - Why it matters:
     important for understanding state-space alternatives and why AIDEEN uses Mamba-style memory ideas.

## 5. Deep Equilibrium Models and Implicit Systems

1. **Deep Equilibrium Models**
   - NeurIPS page: [https://papers.neurips.cc/paper/8358-deep-equilibrium-models](https://papers.neurips.cc/paper/8358-deep-equilibrium-models)
   - arXiv: [https://arxiv.org/abs/1909.01377](https://arxiv.org/abs/1909.01377)
   - Why it matters:
     foundational DEQ paper.

2. **Anderson Acceleration for Fixed-Point Iterations**
   - Link: [https://users.wpi.edu/~walker/Papers/Anderson_Accn_for_FP_Itns_Rep%2C2010.pdf](https://users.wpi.edu/~walker/Papers/Anderson_Accn_for_FP_Itns_Rep%2C2010.pdf)
   - Why it matters:
     fixed-point solver intuition.

3. **Multiscale DEQ**
   - Link: [https://arxiv.org/abs/2006.08656](https://arxiv.org/abs/2006.08656)
   - Why it matters:
     useful for deeper DEQ context once the basics are strong.

## 6. Normalization and Readout

1. **RMSNorm**
   - Link: [https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)
   - Why it matters:
     directly relevant because AIDEEN's exact LM path uses RMSNorm.

## 7. AIDEEN Internal Reading Order

Once the mathematical and AI base is strong enough, study AIDEEN's own docs in this order:

1. [README.md](/Users/sergiosolis/Programacion/AIDEEN/README.md)
2. [token_circuit_trace.txt](/Users/sergiosolis/Programacion/AIDEEN/docs/vision/token_circuit_trace.txt)
3. [memory_reference.md](/Users/sergiosolis/Programacion/AIDEEN/docs/vision/memory_reference.md)
4. [associative_memory_status.md](/Users/sergiosolis/Programacion/AIDEEN/docs/vision/associative_memory_status.md)
5. [distributed_inference.md](/Users/sergiosolis/Programacion/AIDEEN/docs/vision/distributed_inference.md)
6. [model_stabilization_master_plan.txt](/Users/sergiosolis/Programacion/AIDEEN/docs/vision/model_stabilization_master_plan.txt)

## 8. Recommended Order of Study

If you want the strongest progression, use this order:

1. Mathematics for Machine Learning
2. MIT Linear Algebra
3. MIT Calculus
4. Harvard Stat 110
5. Stanford CS229
6. CS231n Notes
7. Deep Learning book
8. Dive into Deep Learning
9. Transformer paper
10. Mamba paper
11. DEQ paper
12. AIDEEN internal documents

## 9. How to Use the Bibliography

Do not read everything passively.

For each major reference, write:

- what problem the text is trying to solve,
- the most important definitions,
- the most important equations,
- how the ideas appear in AIDEEN,
- what remains unclear.

The point is not to accumulate pages read.
The point is to build operational understanding.
