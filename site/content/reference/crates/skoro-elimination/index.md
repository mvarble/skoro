`skoro-elimination` is the note [elimination in square-root form](../../../math/square-root-elimination/)
as code, and nothing else: it depends on no other crate of skoro, and nothing in it knows where the
rows came from. A `Structure` says which blocks of variables each factor involves; a `Problem` holds
one set of numbers over it, each factor's rows and right-hand side, a linear term that is absent by
structure or present, and a constant; and a `Schedule`, computed from the structure alone, orders
the blocks by COLAMD on the block pattern and groups them into the cliques of a junction tree, to
eliminate every block, all but some kept ones, or only some from the factors joined to them.

`Problem::eliminate` reduces each clique's front by faer's unpivoted Householder QR, on the host in
double precision, into a conditional with a positive diagonal and a new factor on its separator,
pushing the linear terms by forward substitution, appending damping rows where they are asked for,
and refusing a block its rows do not determine at a threshold set by the precision the rows were
computed at. The `Elimination` answers the minimizer, the minimum, the integral, `R⁻¹ z` for draws
and the log-determinant, the `Marginal` on the kept blocks as a lower-triangular whitening, and the
`Reduced` factor a replacement leaves, with no decision of rank. No faer type is public: the crate
has a `Matrix` of its own.
