// The workspace's crates and their direct workspace dependencies, as reported by
// `cargo metadata --no-deps --format-version 1` (normal dependencies only). Regenerate the
// `deps` lists by hand when the workspace changes; the layers are derived from them.

export interface Crate {
    name: string;
    purpose: string;
    deps: string[];
    /** The bottom-up page that introduces it, if any. */
    page?: string;
}

export const CRATES: Crate[] = [
    {
        name: 'skoro-backend',
        purpose: 'The supported burn backends (flex, cubecl) and batched tensor shape aliases.',
        deps: [],
        page: 'concepts/backends',
    },
    {
        name: 'skoro-core',
        purpose: 'Ids, errors, roles, packing, samplers, and discrete kinds.',
        deps: ['skoro-backend'],
        page: 'concepts/core',
    },
    {
        name: 'skoro-operation',
        purpose: 'The differentiation engine: forward, VJP and HVP for each operation.',
        deps: ['skoro-backend', 'skoro-core'],
        page: 'concepts/operations',
    },
    {
        name: 'skoro-dense',
        purpose: 'Host-side f64 factorizations (Cholesky, triangular, QR least squares) via faer.',
        deps: ['skoro-backend', 'skoro-core'],
        page: 'concepts/host-linear-algebra',
    },
    {
        name: 'skoro-riemann',
        purpose: 'The Manifold traits, retractions, and the flat manifolds.',
        deps: ['skoro-backend', 'skoro-core', 'skoro-operation'],
        page: 'concepts/manifolds',
    },
    {
        name: 'skoro-graph',
        purpose: 'The probabilistic DAG, its bindings, and the compiler.',
        deps: ['skoro-backend', 'skoro-core', 'skoro-operation', 'skoro-riemann'],
        page: 'concepts/graph',
    },
    {
        name: 'skoro-model',
        purpose: 'ModelClient, typed Q handles, distributions and factors, and the readers.',
        deps: ['skoro-backend', 'skoro-core', 'skoro-graph', 'skoro-operation', 'skoro-riemann'],
        page: 'concepts/model',
    },
    {
        name: 'skoro-linalg',
        purpose: 'Differentiable matmul, transpose, norms, cross products, and Tril.',
        deps: ['skoro-model', 'skoro-graph', 'skoro-riemann'],
        page: 'concepts/quantity-operations',
    },
    {
        name: 'skoro-pointwise',
        purpose: 'Differentiable elementwise log and abs.',
        deps: ['skoro-model', 'skoro-graph', 'skoro-riemann'],
        page: 'concepts/quantity-operations',
    },
    {
        name: 'skoro-so3',
        purpose: 'The rotation group SO(3), with exp, log, compose, and retractions.',
        deps: ['skoro-linalg', 'skoro-model'],
        page: 'concepts/quantity-operations',
    },
    {
        name: 'skoro-normal',
        purpose: 'The manifold-aware Gaussian, its Fisher root, and GaussianBlock.',
        deps: ['skoro-linalg', 'skoro-pointwise', 'skoro-model'],
        page: 'concepts/distributions',
    },
    {
        name: 'skoro-se3',
        purpose: 'Rigid-body transforms SE(3), built on SO(3).',
        deps: ['skoro-so3', 'skoro-linalg'],
        page: 'concepts/quantity-operations',
    },
    {
        name: 'skoro',
        purpose: 'The umbrella crate: re-exports everything, plus a curated prelude.',
        deps: ['skoro-dense', 'skoro-normal', 'skoro-se3', 'skoro-pointwise'],
    },
];

// The `deps` above are trimmed to the edges that set a crate's layer (a dependency implied by
// another is left out), which is all the diagram needs. A crate's layer is one more than the
// deepest of its dependencies.
export function layers(): Crate[][] {
    const depth = new Map<string, number>();
    const byName = new Map(CRATES.map((c) => [c.name, c]));
    const of = (name: string): number => {
        if (!depth.has(name)) {
            const deps = byName.get(name)!.deps;
            depth.set(name, deps.length ? 1 + Math.max(...deps.map(of)) : 0);
        }
        return depth.get(name)!;
    };
    const out: Crate[][] = [];
    for (const c of CRATES) (out[of(c.name)] ??= []).push(c);
    return out;
}
