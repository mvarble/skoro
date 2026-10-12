// Project-wide KaTeX configuration.
//  - Macros every page may use: here.
//  - Macros for one page: its `katex_macros` frontmatter, which wins on a clash.
// Every macro defined here should also be listed on `reference/notation`.

import type { KatexOptions } from 'katex';
import type { Settings } from 'unified';

const alphabet = 'abcdefghijklmnopqrstuvwxyz';
const greeks = [
    ['α', 'alpha'],
    ['β', 'beta'],
    ['γ', 'gamma'],
    ['δ', 'delta'],
    ['ϵ', 'epsilon'],
    ['ζ', 'zeta'],
    ['η', 'eta'],
    ['θ', 'theta'],
    ['ι', 'iota'],
    ['κ', 'kappa'],
    ['λ', 'lambda'],
    ['μ', 'mu'],
    ['ν', 'nu'],
    ['ξ', 'xi'],
    ['π', 'pi'],
    ['ρ', 'rho'],
    ['σ', 'sigma'],
    ['τ', 'tau'],
    ['ϕ', 'phi'],
    ['ψ', 'psi'],
    ['χ', 'chi'],
    ['ω', 'omega'],
];

const titleCase = (str: string) => (str.length ? str.at(0)!.toUpperCase() + str.slice(1) : '');

const config: KatexOptions & Settings = {
    macros: {
        // Font shorthands, as on the blog: `\bbR`, `\calI`, `\bfx`, `\rmd`, ...
        ...Object.fromEntries([
            ...alphabet.split('').flatMap((char) => [
                [`\\rm${char}`, `{\\rm ${char}}`],
                [`\\bf${char}`, `\\boldsymbol{${char}}`],
                [`\\rm${char.toUpperCase()}`, `\\mathrm{${char.toUpperCase()}}`],
                [`\\cal${char.toUpperCase()}`, `\\mathcal{${char.toUpperCase()}}`],
                [`\\scr${char.toUpperCase()}`, `\\mathscr{${char.toUpperCase()}}`],
                [`\\bb${char.toUpperCase()}`, `\\mathbb{${char.toUpperCase()}}`],
                [`\\bf${char.toUpperCase()}`, `\\mathbf{${char.toUpperCase()}}`],
            ]),
            ...greeks.flatMap(([char, ident]) => [
                [`\\rm${ident}`, `{\\rm ${char}}`],
                [`\\bf${ident}`, `\\boldsymbol{${char}}`],
                [`\\rm${titleCase(ident)}`, `\\mathrm{${char.toUpperCase()}}`],
                [`\\cal${titleCase(ident)}`, `\\mathcal{${char.toUpperCase()}}`],
                [`\\bf${titleCase(ident)}`, `\\mathbf{${char.toUpperCase()}}`],
            ]),
        ]),
        '\\defeq': '\\coloneqq',
        '\\eqdef': '\\eqqcolon',

        // Skoro-wide notation.
        '\\R': '\\mathbb{R}',
        '\\E': '\\mathbb{E}',
        '\\T': '\\mathrm{T}',
        '\\retr': '\\operatorname{retract}',
        '\\invretr': '\\operatorname{retract}^{-1}',

        // Typography.
        '\\D': '\\mathrm{D}', // the derivative in tangent charts
        '\\dd': '\\mathrm{d}', // the differential: `\dd x`, `\frac{\dd}{\dd t}`
        '\\hvp': '\\operatorname{hvp}', // an operation's second-order map

        // Points, tangents and maps. `\tangent` is a tangent vector or a frame of them alike.
        '\\manifold': '\\bbX',
        '\\point': 'x',
        '\\tangent': 'v',
        '\\dimension': 'd',
        '\\otherManifold': '\\bbY',
        '\\otherPoint': 'y',
        '\\otherTangent': 'v',
        '\\map': 'F',
        '\\discretePoint': 'k',
        // The order of derivative an operation supports, and the order a pass requires of a step.
        '\\order': '\\operatorname{order}',
        '\\req': '\\operatorname{req}',

        // Densities.
        '\\density': 'p', // a model's density, which its terms' log-densities sum to
        '\\logterm': '\\ell', // a term's log-density
        '\\proposal': 'q', // the density of a sampler's draws
        '\\weight': 'w', // the importance weight
        '\\kernel': '\\kappa', // a probability kernel, such as a regular conditional distribution
        '\\key': '\\xi', // a random key, from which every draw is derived
        '\\transpose': '^{\\top}',
    },
};

export default config;
