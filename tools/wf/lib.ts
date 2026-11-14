// The workflow's model: entries, their stages and documents, and the rules for
// which files may change at which stage. `cli.ts` is the command line over it.
//
// Everything here takes the repository root explicitly, so that the tests can
// run it against throwaway repositories.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

// ---------------------------------------------------------------------------
// Layout

export const ARCHIVE = 'site/content/archive';
export const MATH = 'site/content/math';
export const CONTENT = 'site/content';
export const ACTIVE = '.workflow/active';
// What counts as the library's code: changed only once an entry reaches `build`.
export const CODE = [/^crates\//, /^Cargo\.(toml|lock)$/, /^rust-toolchain(\.toml)?$/];

// ---------------------------------------------------------------------------
// Kinds and stages

export type Kind = 'feature' | 'decision' | 'chore';
export type Stage = 'math' | 'implementation' | 'plan' | 'build' | 'publish' | 'writeup';
export type Terminal = 'done' | 'abandoned';

export const KINDS: Record<Kind, Stage[]> = {
    // An algorithm, a distribution, an example: the mathematics first, then how
    // it is built, then the plan, the code, and the documentation.
    feature: ['math', 'implementation', 'plan', 'build', 'publish'],
    // A position the library takes, argued in prose and then published.
    decision: ['writeup', 'publish'],
    // Work that needs no new mathematics or design: a port, a refactor.
    chore: ['plan', 'build', 'publish'],
};

// The stages that are written as a document of their own, by file name.
export const DOCUMENT_STAGES: Stage[] = ['math', 'implementation', 'plan', 'writeup'];

// What each stage opens for editing, for the rest of the entry's life.
// `entry` (the entry's own folder) is open from the start.
export type Area = 'math' | 'code' | 'pages';
export const UNLOCKS: Partial<Record<Stage, Area[]>> = {
    math: ['math'],
    writeup: ['math'],
    build: ['code'],
    publish: ['pages'],
};

export const isKind = (value: string): value is Kind => value in KINDS;

// ---------------------------------------------------------------------------
// Records

export interface Approval {
    date: string;
    commit: string;
    // The approved document's hash, for the stages that have one.
    hash?: string;
    // Set when the entry was reopened at or before this stage; the stage has
    // to be approved again before the entry moves past it.
    stale?: boolean;
}

export interface WorkflowEvent {
    date: string;
    action: 'start' | 'approve' | 'reopen' | 'abandon' | 'resume' | 'amend';
    // The stage the event is about. An amendment of a statement has none.
    stage?: Stage;
    reason?: string;
    commit?: string;
    // An amendment: what was changed, as the amending entry lists it under `amends`, and the
    // entry that changed it.
    document?: string;
    by?: string;
}

export interface Workflow {
    kind: Kind;
    stage: Stage | Terminal;
    started: string;
    finished?: string;
    events: WorkflowEvent[];
    approvals: Partial<Record<Stage, Approval>>;
}

export class WorkflowError extends Error {}

export const today = () => new Date().toISOString().slice(0, 10);

export const entryDir = (slug: string) => `${ARCHIVE}/${slug}`;
export const workflowFile = (slug: string) => `${entryDir(slug)}/workflow.json`;

export function readWorkflow(root: string, slug: string): Workflow {
    const file = path.join(root, workflowFile(slug));
    let parsed: Workflow;
    try {
        parsed = JSON.parse(fs.readFileSync(file, 'utf8')) as Workflow;
    } catch (error) {
        throw new WorkflowError(
            `${workflowFile(slug)}: cannot be read (${(error as Error).message}).`,
        );
    }
    if (!isKind(parsed.kind)) {
        throw new WorkflowError(`${workflowFile(slug)}: unknown kind '${parsed.kind}'.`);
    }
    const stages: string[] = [...KINDS[parsed.kind], 'done', 'abandoned'];
    if (!stages.includes(parsed.stage)) {
        throw new WorkflowError(
            `${workflowFile(slug)}: '${parsed.stage}' is not a stage of a ${parsed.kind}.`,
        );
    }
    parsed.events ??= [];
    parsed.approvals ??= {};
    return parsed;
}

export function writeWorkflow(root: string, slug: string, workflow: Workflow) {
    const file = path.join(root, workflowFile(slug));
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify(workflow, null, 4) + '\n');
}

export function activeSlug(root: string): string | undefined {
    const file = path.join(root, ACTIVE);
    if (!fs.existsSync(file)) return undefined;
    const slug = fs.readFileSync(file, 'utf8').trim();
    return slug || undefined;
}

export function setActive(root: string, slug: string | undefined) {
    const file = path.join(root, ACTIVE);
    if (slug === undefined) {
        fs.rmSync(file, { force: true });
        return;
    }
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, slug + '\n');
}

export function listEntries(root: string): string[] {
    const dir = path.join(root, ARCHIVE);
    if (!fs.existsSync(dir)) return [];
    return fs
        .readdirSync(dir, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => entry.name)
        .sort();
}

// The file holding a stage's document: `.svx` when there is one, else `.md`.
export function documentPath(root: string, slug: string, stage: Stage): string {
    const svx = `${entryDir(slug)}/${stage}.svx`;
    return fs.existsSync(path.join(root, svx)) ? svx : `${entryDir(slug)}/${stage}.md`;
}

export const hasDocument = (stage: Stage) => DOCUMENT_STAGES.includes(stage);

// A plan's checkboxes are ticked during `build`. The plan is otherwise frozen,
// so its hash is taken with every box unticked.
export function documentHash(root: string, file: string, stage: Stage): string | undefined {
    const abs = path.join(root, file);
    if (!fs.existsSync(abs)) return undefined;
    let text = fs.readFileSync(abs, 'utf8');
    if (stage == 'plan') text = text.replace(/^(\s*[-*] )\[[xX]\]/gm, '$1[ ]');
    return createHash('sha256').update(text).digest('hex');
}

export function checkboxes(text: string) {
    const ticked = (text.match(/^\s*[-*] \[[xX]\]/gm) ?? []).length;
    const open = (text.match(/^\s*[-*] \[ \]/gm) ?? []).length;
    return { ticked, open, total: ticked + open };
}

// The stages of an entry reached so far, in order: those approved at some
// point (stale or not), then the current one.
export function reached(workflow: Workflow): Stage[] {
    const stages = KINDS[workflow.kind];
    if (workflow.stage == 'done') return stages;
    const out = stages.filter((stage) => workflow.approvals[stage]);
    if (workflow.stage != 'abandoned' && !out.includes(workflow.stage)) out.push(workflow.stage);
    return out;
}

export function unlocked(workflow: Workflow): Set<Area> {
    return new Set(reached(workflow).flatMap((stage) => UNLOCKS[stage] ?? []));
}

// ---------------------------------------------------------------------------
// Paths

export type Place =
    | { area: 'workflow' }
    | { area: 'entry'; slug: string; stage?: Stage }
    | { area: Area }
    | { area: 'free' };

// What a repository-relative path is, as far as the rules are concerned.
export function placeOf(rel: string): Place {
    rel = rel.split(path.sep).join('/');
    if (rel == ACTIVE || rel.startsWith('.workflow/')) return { area: 'workflow' };
    if (rel.startsWith(ARCHIVE + '/')) {
        const [slug, file, ...rest] = rel.slice(ARCHIVE.length + 1).split('/');
        if (!slug || !file) return { area: 'free' };
        if (file == 'workflow.json' && !rest.length) return { area: 'workflow' };
        const name = /^([a-z]+)\.(md|svx)$/.exec(file)?.[1] as Stage | undefined;
        const stage = !rest.length && name && hasDocument(name) ? name : undefined;
        return { area: 'entry', slug, ...(stage ? { stage } : {}) };
    }
    if (rel.startsWith(MATH + '/')) return { area: 'math' };
    if (rel.startsWith(CONTENT + '/')) return { area: 'pages' };
    if (CODE.some((pattern) => pattern.test(rel))) return { area: 'code' };
    return { area: 'free' };
}

// What an entry may amend in another: a done entry's stage documents and its statements, to keep
// true a reference into the math notes. The key is how the amending entry lists the file under
// `amends` in its `index.md`: its path under the content directory, without its extension.
export function amendKey(rel: string): string | undefined {
    rel = rel.split(path.sep).join('/');
    if (!rel.startsWith(ARCHIVE + '/')) return undefined;
    const [slug, file, ...rest] = rel.slice(ARCHIVE.length + 1).split('/');
    if (!slug || !file) return undefined;
    if (!rest.length) {
        const name = /^([a-z]+)\.(md|svx)$/.exec(file)?.[1] as Stage | undefined;
        return name && hasDocument(name) ? `archive/${slug}/${name}` : undefined;
    }
    const statement = file == 'statements' && rest.length == 1 && /^([a-z0-9-]+)\.md$/.exec(rest[0]!);
    return statement ? `archive/${slug}/statements/${statement[1]}` : undefined;
}

// The entry an `amends` key names, and the stage of the document, when it is one.
export function amendTarget(key: string): { slug: string; stage?: Stage; name?: string } | undefined {
    const document = /^archive\/([a-z0-9-]+)\/([a-z]+)$/.exec(key);
    if (document && hasDocument(document[2] as Stage)) {
        return { slug: document[1]!, stage: document[2] as Stage };
    }
    const statement = /^archive\/([a-z0-9-]+)\/statements\/([a-z0-9-]+)$/.exec(key);
    return statement ? { slug: statement[1]!, name: statement[2]! } : undefined;
}

// The file an `amends` key names.
export function amendPath(root: string, key: string): string | undefined {
    const target = amendTarget(key);
    if (!target) return undefined;
    return target.stage
        ? documentPath(root, target.slug, target.stage)
        : `${entryDir(target.slug)}/statements/${target.name}.md`;
}

// What the entry being worked on may change in the others.
export interface Amending {
    // Whether an entry is done: only a done entry is amended.
    isDone(slug: string): boolean;
    // What the entry lists under `amends`. An edit about to be made is not held to the list,
    // which is written as the publish goes; a branch's history is.
    amends: string[];
}

// Why `rel` may not be changed while `slug` (if any) is the entry being worked
// on, or undefined when it may. `editing` is set for a single edit about to be
// made, as opposed to a change already in a branch's history: an edit may only
// touch the current stage's document, while the history may hold any stage's.
export function refusal(
    rel: string,
    subject: { slug: string; workflow: Workflow } | undefined,
    editing: boolean,
    amending?: Amending,
): string | undefined {
    const place = placeOf(rel);
    switch (place.area) {
        case 'free':
            return undefined;
        case 'workflow':
            return editing ? `${rel} is written by \`bin/wf\` alone.` : undefined;
        case 'entry': {
            if (!subject) {
                return editing
                    ? `${rel} belongs to entry '${place.slug}', which is not active on this branch.`
                    : undefined;
            }
            if (place.slug != subject.slug) {
                const key = amendKey(rel);
                const publishing = reached(subject.workflow).includes('publish');
                if (key && amending?.isDone(place.slug) && publishing) {
                    if (editing || amending.amends.includes(key)) return undefined;
                    return (
                        `${rel} belongs to entry '${place.slug}', which is done; list it as ` +
                        `'${key}' under \`amends\` in ${entryDir(subject.slug)}/index.md.`
                    );
                }
                if (key && amending?.isDone(place.slug)) {
                    return (
                        `${rel} belongs to entry '${place.slug}', which is done; '${subject.slug}' ` +
                        'may amend it only at its publish stage, to keep a reference into the math notes true.'
                    );
                }
                return `${rel} belongs to entry '${place.slug}', but this branch works on '${subject.slug}'.`;
            }
            if (!editing || !place.stage) return undefined;
            const { workflow } = subject;
            if (!KINDS[workflow.kind].includes(place.stage)) {
                return `a ${workflow.kind} has no ${place.stage} stage.`;
            }
            if (place.stage == workflow.stage) return undefined;
            if (place.stage == 'plan' && workflow.stage == 'build') return undefined;
            return (
                `${rel} is the ${place.stage} document, and the entry is at ${workflow.stage}. ` +
                `To change it, ask for \`/wf-reopen ${place.stage}\`.`
            );
        }
        default: {
            const area = place.area;
            if (!subject) {
                return area == 'code'
                    ? `${rel} is code, and no entry is active: start one with \`/wf-start\`.`
                    : undefined;
            }
            if (unlocked(subject.workflow).has(area)) return undefined;
            const opener = Object.entries(UNLOCKS)
                .filter(([stage, areas]) =>
                    KINDS[subject.workflow.kind].includes(stage as Stage) && areas.includes(area),
                )
                .map(([stage]) => stage);
            return (
                `${rel} is ${describe(area)}, which opens at the ` +
                `${opener.join(' or ') || 'no'} stage; '${subject.slug}' is at ${subject.workflow.stage}.`
            );
        }
    }
}

const describe = (area: Area) =>
    ({ math: 'a math note', code: 'code', pages: 'a documentation page' })[area];

// ---------------------------------------------------------------------------
// Git

export function git(root: string, ...args: string[]): string {
    return execFileSync('git', args, {
        cwd: root,
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'pipe'],
    }).trim();
}

export function tryGit(root: string, ...args: string[]): string | undefined {
    try {
        return git(root, ...args);
    } catch {
        return undefined;
    }
}

export function repoRoot(cwd: string): string {
    const root = tryGit(cwd, 'rev-parse', '--show-toplevel');
    if (!root) throw new WorkflowError('not inside a git repository.');
    return root;
}

export const currentBranch = (root: string) => tryGit(root, 'branch', '--show-current') ?? '';
export const headCommit = (root: string) => tryGit(root, 'rev-parse', '--short', 'HEAD') ?? '';

export function isCommitted(root: string, rel: string): boolean {
    return git(root, 'status', '--porcelain', '--', rel) == '';
}

// Every path that differs from `base`: committed since the merge base, staged,
// unstaged, or untracked. Undefined when `base` does not resolve.
export function changedSince(root: string, base: string): string[] | undefined {
    const mergeBase = tryGit(root, 'merge-base', base, 'HEAD');
    if (!mergeBase) return undefined;
    const lines = [
        git(root, 'diff', '--name-only', mergeBase),
        git(root, 'ls-files', '--others', '--exclude-standard'),
    ].join('\n');
    return [...new Set(lines.split('\n').filter(Boolean))].sort();
}

// The default base: the remote's main branch, else the local one.
export function defaultBase(root: string): string | undefined {
    for (const ref of ['origin/main', 'main']) {
        if (tryGit(root, 'rev-parse', '--verify', '--quiet', ref)) return ref;
    }
    return undefined;
}

// ---------------------------------------------------------------------------
// Frontmatter, read only as far as the workflow needs: a list field such as
// `pages: [guide/first-model]`, written inline, as a block, or as a bracketed list spread over
// several lines, as Prettier writes one too long for a line.

export function frontmatterList(text: string, field: string): string[] | undefined {
    const block = /^---\r?\n([\s\S]*?)\r?\n---/.exec(text)?.[1];
    if (!block) return undefined;
    const lines = block.split(/\r?\n/);
    const at = lines.findIndex((line) => line.startsWith(`${field}:`));
    if (at < 0) return undefined;
    const value = lines[at]!.slice(field.length + 1).trim();
    const unquote = (item: string) => item.trim().replace(/^(['"])(.*)\1$/, '$2');
    // A bracketed list opens on the field's line or, indented, on the next one, and runs to the
    // line that closes it.
    const opens = value.startsWith('[') ? at : !value && /^\s+\[/.test(lines[at + 1] ?? '') ? at + 1 : -1;
    if (opens >= 0) {
        let flow = opens == at ? value : lines[opens]!.trim();
        for (let i = opens + 1; !flow.includes(']') && i < lines.length; i++) flow += ' ' + lines[i]!.trim();
        const inner = flow.replace(/^\[/, '').replace(/\].*$/, '').trim();
        return inner ? inner.split(',').map(unquote).filter(Boolean) : [];
    }
    if (value) return [unquote(value)];
    const items: string[] = [];
    for (const line of lines.slice(at + 1)) {
        const item = /^\s+-\s+(.*)$/.exec(line);
        if (!item) break;
        items.push(unquote(item[1]!));
    }
    return items;
}

// The page a documentation path belongs to, as `<track>/<slug>`, if it is one.
export function pageKey(rel: string): string | undefined {
    const match = new RegExp(`^${CONTENT}/([^/]+)/([^/]+)/`).exec(rel);
    if (!match || match[1] == 'math' || match[1] == 'archive') return undefined;
    return `${match[1]}/${match[2]}`;
}

export function pageExists(root: string, key: string): boolean {
    return ['md', 'svx'].some((ext) =>
        fs.existsSync(path.join(root, CONTENT, key, `index.${ext}`)),
    );
}

// The crate a path belongs to, as its directory under `crates/`, if it is one.
export function crateOf(rel: string): string | undefined {
    return /^crates\/([^/]+)\//.exec(rel)?.[1];
}

export function crateExists(root: string, name: string): boolean {
    return fs.existsSync(path.join(root, 'crates', name, 'Cargo.toml'));
}

// A crate's description, which its page in the reference shows above its reading order.
export const crateDescription = (name: string) => `${CONTENT}/reference/crates/${name}/index.md`;

export const hasTodo = (text: string) => /^>\s*\[!TODO\]/m.test(text);
