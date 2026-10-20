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
        name: 'skoro-tensor',
        purpose: 'The layer over burn: tensors at an explicit precision on a supported device',
        deps: [],
        page: 'concepts/queries',
    },
    {
        name: 'skoro-core',
        purpose: 'Quantities, operations and their derivatives, models, terms and queries',
        deps: ['skoro-tensor'],
        page: 'concepts/spaces',
    },
    {
        name: 'skoro',
        purpose: 'The umbrella: re-exports the core and the tensor layer',
        deps: ['skoro-core'],
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
