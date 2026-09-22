// The one place that defines the site's structure. A page belongs to the track named by its
// directory under `src/content/`, and is placed within it by its `order` frontmatter.

export const SITE = {
    name: 'skoro',
    tagline:
        'Probabilistic modeling in Rust, with second-order, manifold-aware differentiation on burn backends.',
    description:
        'Skoro is a probabilistic modeling library designed for speed, modularity, and extensibility. ' +
        'It bridges hierarchical probabilistic modeling, a bespoke first- and second-order ' +
        'automatic-differentiation engine that is aware of non-linear manifolds, and high-performance ' +
        'tensor execution on burn backends (CPU via flex, GPU via cubecl).',
    repo: 'https://github.com/mvarble/skoro',
    branch: 'main',
    // TODO: bump when the crates are versioned for release.
    version: '0.0.0',
} as const;

export interface Track {
    id: string;
    title: string;
    /** A short label for the top bar. */
    short: string;
    blurb: string;
}

export const TRACKS: Track[] = [
    {
        id: 'start',
        title: 'Start here',
        short: 'Start',
        blurb: 'What skoro is, why it exists, and how to read the rest of these docs.',
    },
    {
        id: 'guide',
        title: 'Guide (top-down)',
        short: 'Guide',
        blurb:
            'Start from what a user writes, such as a model, a distribution or a density, and ' +
            'dig into the machinery only when you need it.',
    },
    {
        id: 'concepts',
        title: 'Concepts (bottom-up)',
        short: 'Concepts',
        blurb:
            'Build skoro from its foundations, one crate at a time and in dependency order, so ' +
            'that each idea rests only on the ones before it.',
    },
    {
        id: 'design',
        title: 'Design decisions',
        short: 'Design',
        blurb:
            'The choices that shaped the library, each with its context, the decision taken, and ' +
            'what it costs.',
    },
    {
        id: 'reference',
        title: 'Reference',
        short: 'Reference',
        blurb: 'Crate map, glossary, notation, and project status.',
    },
];
