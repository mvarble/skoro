# skoro documentation site

A static SvelteKit site whose pages are `.svx` (Markdown + Svelte, via mdsvex) with KaTeX math.

```sh
pnpm install
pnpm dev                          # http://localhost:5173
pnpm build                        # static site in build/
BASE_PATH=/docs pnpm build        # when served under a subpath, not at the root of a domain
SITE_URL=https://skoro.dev pnpm build  # makes the links in llms.txt absolute, as CI does
pnpm check && pnpm lint && pnpm test
pnpm todos                        # every `[!TODO]` left in the content
SITE_CONTENT=test/content pnpm build   # the fixture, with a document of every kind
```

Run every command from `site/`: file references resolve against the repository root, which is
taken to be the parent of the working directory.

## Layout

| Path                                     | What it is                                                                                                                            |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `content/landing.md`                     | The prose on the home page.                                                                                                           |
| `content/<track>/<slug>/index.{md,svx}`  | A page: what skoro _is_. The folder is its track (`start`, `guide`, `concepts`, `reference`), and `order` places it within the track. |
| `content/math/<slug>/index.{md,svx}`     | A note of mathematics, one idea to a note.                                                                                            |
| `content/archive/<slug>/index.md`        | An archive entry: one piece of development, opened by `bin/wf start`.                                                                 |
| `content/archive/<slug>/<name>.{md,svx}` | Its documents: `math`, `implementation`, `plan` (a feature), `writeup` (a decision). Frozen once approved.                            |
| `content/archive/<slug>/workflow.json`   | Its stage record, written by `bin/wf` alone and shown as the entry's history.                                                         |
| `content/**/*.bib`                       | Bibliography entries, for `[](cite:key)`.                                                                                             |
| `src/lib/site.ts`                        | Site name, repo URL, the tracks, and the names of an entry's documents. The navigation comes from here.                               |
| `src/lib/crates.ts`                      | The crate map's data. Keep it in step with `cargo metadata`.                                                                          |
| `src/styles/`                            | CSS, split as `tokens/{colors,dimensions,typography}.css` + element and utility sheets. Colors exist only as tokens.                  |
| `plugins/`                               | skoro's own Markdown plugins: callouts, quoted source, heading anchors.                                                               |
| `katex.config.ts`                        | Site-wide KaTeX macros.                                                                                                               |
| `test/content/`                          | A fixture with a document of every kind. `SITE_CONTENT=test/content pnpm build` builds it instead of `content/`.                      |

The content goes through [`@mvarble/mesearch-cms`](https://github.com/mvarble/mesearch/tree/main/packages/cms)
with its `skoroPreset`: it reads every document, resolves links, numbers equations and statements,
collects citations, and serves all of it to the routes as `$cms`. A link to a document that does
not exist, or a `depends_on` that names nothing, is reported when the site builds. KaTeX and the
scroll box around display math come from
[`@mvarble/mesearch-markdown`](https://github.com/mvarble/mesearch/tree/main/packages/markdown),
and the map on the Mathematics page is `@mvarble/mesearch-ui`'s, in skoro's colours.

## Writing a document

```yaml
---
title: Fisher scoring # the first heading, or the folder name, otherwise
summary: One sentence. # the first paragraph otherwise: sidebar, lists, <meta>, llms.txt
order: 11 # pages only: position within the track
depends_on: [retraction, concepts/graph] # a bare slug is a math note
katex_macros: # this document only; wins over katex.config.ts
    \fisher: \mathcal{I}
---
```

An entry's `index.md` also lists the `pages` it changed (`[guide/first-model, concepts/graph]`),
the `crates` it changed (`[skoro-core]`), and the earlier entries it `revises`. Each page then shows
the entries that changed it, newest first, under "How this came to be"; an entry links each of its
crates to that crate's reading order; and an entry that a later one revises says so at its top.

- **Math:** `$inline$` and `$$display$$`. A KaTeX error does not fail the build. The formula
  renders in red, and the build prints a `katex:` warning naming the file.
- **Numbered equations:** `$$ ... @tag(slug) $$` numbers a display equation, and `[](eq:slug)`
  refers to it as `(1)`. From another document: `[](eq:math/<slug>/<eq-slug>)`.
- **Statements:** a theorem or definition is a small document of its own beside the one that
  shows it, with `type: statement` and a `kind`:

    ```svelte
    <script>
        import Statement from '$lib/components/Statement.svelte';
        import * as whitening from './statements/whitening.md';
    </script>

    <Statement {...whitening} />
    ```

    It is numbered along with the document's equations, and `[%full](statement:whitening)` refers
    to it as "Lemma 2". `$lib/components/Proof.svelte` frames a proof.

- **Citations:** `[](cite:key)` renders as `[Foll99]`, and the page ends with the references it
  cites.
- **Callouts:** GitHub-style blockquotes, which stay plain Markdown in the `.md` exports:
  `> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`, `> [!DECISION]`, `> [!TODO]`. Text after the marker
  on the first line becomes the title. `pnpm todos` lists every `[!TODO]`, and `bin/wf approve`
  refuses a document that still has one.
- **Quoting source:** a fence whose only line is `{path}` or `{path:start..end}` is replaced by
  that file's lines. The path is relative to the document, and line numbers are **1-indexed and
  inclusive**. The block gets a caption linking to GitHub. If every quoted line is a `///` or
  `//!` doc comment, the markers are stripped and the block is highlighted as Markdown, unless the
  fence names a language (use `rust` to quote a doctest's code). Pages quote the code that exists;
  the archive's documents, written before it did, sketch it inline.
- **Links:** write them the way the files sit on disk: `../../math/retraction/`, `plan.md`,
  `../../concepts/`. They are rewritten to the site's URLs, and to `.md` URLs in the exports.
- **Svelte:** a `.svx` document can import components (see Model-graph figures below). Outside
  code and math, `{` and `<` are Svelte syntax, so put generics like `Q<B, M>` in backticks.
- `h2`/`h3` headings get anchors and appear in "On this page".

## KaTeX in components

`$…$` in a `.svx` page is rendered by the build. For math inside Svelte markup or props, use
`<Tex>`, or `renderTex` / `renderMixed` from `$lib/tex`:

```svelte
<script>
    import Tex from '$lib/components/Tex.svelte';
</script>

<Tex src={String.raw`\calI(\theta) = \E[s s^\top]`} />
<Tex display src={String.raw`\int_0^1 f`} macros={{ '\\f': 'g' }} />
```

- It renders synchronously, so prerendering writes the math into the static HTML and it displays
  without JavaScript. Svelte keeps those nodes when it hydrates.
- A client-side navigation renders the page in the browser, so KaTeX (about 76 kB gzipped) ships in
  the JS chunk of any page that uses these components, and only those pages.
- The macros are the site-wide ones in `katex.config.ts`, plus any you pass. Frontmatter
  `katex_macros` belong to the Markdown pipeline and don't reach components.
- A bad formula renders in KaTeX's red, and the build prints a `katex: component` warning.

## Graph figures

`<ModelGraph spec={…} />` draws a hand-placed figure of a graph as SVG: one of skoro's graphs (a
compilation graph, a probabilistic graph, or a program's computation or backpropagation graph), or
any other. The component knows nothing of what a node means. Each figure declares its own kinds of
node and edge, with the name the legend and the Markdown export give each, and how each is drawn: a
hue from a fixed palette (`gray`, `blue`, `orange`, `green`, `purple`, `teal`, `red`, `sky`,
`amber`, `olive`, `magenta`), a shape, a fill and an outline for nodes, a line style for edges. It
adds a legend of the kinds the figure uses, in the order it declares them, and a caption.

Put each spec in a `*.graph.ts` file beside its page, and import it by name:

```svelte
<script>
    import ModelGraph from '$lib/components/ModelGraph.svelte';
    import { normal } from './normal.graph';
</script>

<ModelGraph spec={normal} />
```

```ts
import { defineGraph } from '$lib/graph/spec';

export const normal = defineGraph({
    title: 'A Gaussian quantity', // accessible name; starts the caption
    caption: 'Its mean, and the term that scores it.', // optional
    width: 760,
    height: 430, // the viewport
    nodeKinds: {
        quantity: { name: 'Quantity', hue: 'orange' },
        observed: { name: 'Observed quantity', hue: 'orange', fill: 'shaded' },
        term: { name: 'Term', hue: 'purple', shape: 'square' },
    },
    edgeKinds: {
        conditions: { name: 'conditions a term' },
        scores: { name: 'scores a quantity', line: 'dotted' },
    },
    nodes: [
        {
            id: 'mu',
            x: 70,
            y: 300,
            kind: 'quantity',
            label: 'μ',
            annotation: 'mean',
            annotationAngle: 180,
        },
        { id: 'x', x: 400, y: 300, kind: 'observed', label: 'x' },
        { id: 't', x: 650, y: 175, kind: 'term', label: 't' },
    ],
    edges: [
        { from: 'mu', to: 't', kind: 'conditions' }, // straight
        { from: 't', to: 'x', kind: 'scores', bend: 30 },
        { from: 'mu', to: 'x', via: [[400, 400]], curve: 'bezier', toAngle: 90 }, // no kind: plain, unlisted
    ],
});
```

- **Kinds are the figure's own.** A node's `kind` and an edge's `kind` are keys of the figure's
  `nodeKinds` and `edgeKinds`, and a key the figure does not define fails the build. An edge with no
  kind is a plain solid line in the neutral colour, and the legend leaves it out. Kind names may
  contain TeX.
- **Coordinates are mathematical:** the origin is the bottom-left corner and +Y points up. Angles are
  degrees counter-clockwise from +X, so 90 is straight up.
- **Endpoints are computed for you.** An edge meets a node's boundary (circle, or square for
  `log-likelihood`) where the line toward its next point crosses it. Set `fromAngle` / `toAngle` to
  pick the spot yourself.
- **Routing:** `via` adds intermediate points. `curve: 'bezier'` draws a smooth curve through them
  that leaves and meets each node along its normal. `bend: θ` curves a direct edge like TikZ's
  `bend left=θ`.
- **Text:** node `label`s, `annotation`s, edge `label`s and the `caption` may mix plain text with
  `$…$` TeX (`\\$` is a literal dollar). Write TeX in `String.raw` or double the backslashes. A
  spec's `katexMacros` are added over the site-wide macros for that figure. Text is laid out in an
  HTML layer over the SVG rather than inside it, so it needs no `<foreignObject>` and no JavaScript.
- **Styling:** `arrow` (`end`, `start`, `both`, `none`), `label` / `labelAt`, `muted`, and the
  escape hatches `class` / `style` on nodes and edges. The hues are tokens `--graph-<hue>` in
  `src/styles/tokens/colors.css`, and the rules are in `src/styles/utils/graph.css`.
- A mistake, such as an edge to a missing node or a kind the figure does not define, fails the
  build with the figure's title in the message.
- In the `.md` exports, the tag is replaced by a list of the figure's nodes and edges with the
  figure's names for their kinds, so a language model reads the same figure.
- The geometry is `src/lib/graph/geometry.ts`, tested by `pnpm test`.

## Reading orders

`<ReadingOrder spec={…} />` draws the order in which to read a crate's modules: the modules top to
bottom, each use of an earlier module as an arc on the left, and each use of a later one as a dashed
orange arc on the right, with the items it uses named beside the module. A band marks each cycle,
and hovering, focusing or tapping a module shows all of its uses and what uses it. Uses that a
longer chain of earlier uses already implies are hidden until the checkbox shows them.

```ts
import { defineReadingOrder } from '$lib/reading/spec';

export const order = defineReadingOrder({
    title: 'The order in which to read skoro-core',
    caption: 'Optional: shown under the figure.',
    modules: [{ id: 'address' }, { id: 'key' }, { id: 'testing', apart: true }],
    uses: [
        { from: 'key', to: 'address', items: ['Address', 'Segment'] },
        { from: 'testing', to: 'key', items: ['Key'] },
    ],
});
```

- **The order is the spec's.** The component draws the modules in the order given; it does not
  choose one. `bin/reading-order` (in `tools/reading-order/`) reads a crate's sources and computes
  an order with as few items used ahead as can be.
- **Items:** a name with a lower-case last segment (`narrow`, `Lowering.compile`) is a function or
  method, and is shown with `()`.
- A use of a module the spec does not list, a module listed twice, or a module set `apart` before
  one that is not, fails with the figure's title in the message.
- The geometry is `src/lib/reading/layout.ts`, tested by `pnpm test`; the styles are
  `src/styles/utils/reading.css`, from the graph palette's tokens.

`/reading-order/<crate>/` draws the reading order of `crates/<crate>`, one page for each crate in
the workspace, computed from its sources when the site is built (and on every request under
`pnpm dev`). The crate map (`$lib/crates.ts`) links each crate to its page, so a crate it lists
must be one under `crates/`. `$lib/server/reading.ts` reads the crates.

## For language models

Every document is also served as Markdown at its URL with `.md` in place of the trailing slash
(`/math/retraction.md`), with its frontmatter kept, file references expanded, statements inlined
and links pointed at other `.md` files. `/llms.txt` indexes them ([llmstxt.org](https://llmstxt.org)),
and `/llms-full.txt` concatenates all of them in reading order.
