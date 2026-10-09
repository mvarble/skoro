// The one place that defines the site's structure. `vite.config.ts` hands the tracks and the
// entry documents to the content database, and the navigation is built from the same lists.

export const SITE = {
    name: 'skoro',
    tagline: 'Probabilistic modeling in Rust.',
    description:
        'Skoro is a probabilistic modeling library in Rust. This site tells what it is, from the ' +
        'top down and from the bottom up, keeps the mathematics it rests on, and archives how each ' +
        'part of it came to be.',
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

// The tracks of pages, in reading order. A page belongs to the track named by its folder under
// `content/`, and is placed within it by its `order` frontmatter. Pages say what skoro *is*: they
// are rewritten as the library changes, and what they used to say lives on in the archive.
export const TRACKS: Track[] = [
    {
        id: 'start',
        title: 'Start here',
        short: 'Start',
        blurb: 'What skoro is, who it is for, and how to read the rest of these docs.',
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
            'Build skoro from its foundations, one idea at a time and in dependency order, so ' +
            'that each rests only on the ones before it.',
    },
    {
        id: 'reference',
        title: 'Reference',
        short: 'Reference',
        blurb: 'Crate map, glossary, notation, and project status.',
    },
];

// The two collections beside the tracks.
export const MATH = {
    id: 'math',
    title: 'Mathematics',
    short: 'Math',
    blurb:
        'The mathematics skoro rests on, one idea to a note, linked by what each builds on. The ' +
        'archive’s writeups lean on these notes rather than repeating them.',
};

export const ARCHIVE = {
    id: 'archive',
    title: 'Archive',
    short: 'Archive',
    blurb:
        'How each part of skoro came to be: the mathematics worked out for it, the implementation ' +
        'it was given, and the plan it was built by, kept as they were approved.',
};

// The documents an archive entry may hold, in the order they are written.
export const DOCUMENTS = [
    { name: 'math', title: 'Mathematics' },
    { name: 'writeup', title: 'Writeup' },
    { name: 'implementation', title: 'Implementation' },
    { name: 'plan', title: 'Plan' },
] as const;

export const KIND_LABELS: Record<string, string> = {
    feature: 'Feature',
    decision: 'Decision',
    chore: 'Chore',
};

export const STATUS_LABELS: Record<string, string> = {
    active: 'In progress',
    done: 'Done',
    abandoned: 'Abandoned',
};
