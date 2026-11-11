---
type: statement
kind: proposition
title: Freeing removes exactly what nothing needs any more
---

Let a retirement remove the nodes $R$ and give a model $M'$, let $K$ be the nodes $M'$ keeps, and
let $C$ be the retirement's candidates. Let $F$ be the largest set of nodes of $M'$, none of them a
quantity or in $K$, each an ancestor of a candidate or a candidate itself, and each with every
consumer in $F$. Then

1. no node outside $F$ is reached from a node of $F$, so removing $F$ changes no value that a node
   of $M' \setminus F$ computes, and so no term, binding, sampler, proposal or target that remains;
2. a node of $M' \setminus F$ with no consumer is a quantity, in $K$, or had no consumer before
   the retirement either;
3. $F$ is what reference counting computes: start from the candidates, and remove a node once it
   is neither a quantity nor in $K$ and has no consumer left, then visit its inputs.
