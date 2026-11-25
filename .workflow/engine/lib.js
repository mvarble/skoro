// The workflow's model: entries, their stages and documents, and the rules for which files may
// change at which stage. `cli.ts` is the command line over it.
//
// Everything here takes the repository root explicitly, so that the tests can run it against
// throwaway repositories. What differs between projects (kinds, stages, the code, the layout)
// comes from the project's `workflow.config.ts`, which `useConfig` sets before anything runs: the
// command line sets it once per command.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
let current;
export function useConfig(config) {
    current = config;
}
export function config() {
    if (!current)
        throw new WorkflowError('no workflow config is in use.');
    return current;
}
// The layout, from the config.
export const CONTENT = () => config().content ?? 'site/content';
export const ARCHIVE = () => `${CONTENT()}/archive`;
export const MATH = () => {
    const math = config().math ?? 'math';
    return math === false ? undefined : `${CONTENT()}/${math}`;
};
export const ACTIVE = '.workflow/active';
export const MANAGED = '.workflow/managed.json';
export const SKILLS = '.agents/skills';
export const kinds = () => Object.fromEntries(Object.entries(config().kinds).map(([kind, { stages }]) => [kind, stages]));
export const stagesOf = (kind) => config().kinds[kind]?.stages ?? [];
export const stageConfig = (stage) => config().stages[stage] ?? {};
export const isKind = (value) => Object.hasOwn(config().kinds, value);
export const hasDocument = (stage) => !!stageConfig(stage).document;
export const skillOf = (stage) => stageConfig(stage).skill ?? `/wf-${stage}`;
export const unlocksOf = (stage) => stageConfig(stage).unlocks ?? [];
export class WorkflowError extends Error {
}
export const today = () => new Date().toISOString().slice(0, 10);
export const entryDir = (slug) => `${ARCHIVE()}/${slug}`;
export const workflowFile = (slug) => `${entryDir(slug)}/workflow.json`;
export function readWorkflow(root, slug) {
    const file = path.join(root, workflowFile(slug));
    let parsed;
    try {
        parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
    }
    catch (error) {
        throw new WorkflowError(`${workflowFile(slug)}: cannot be read (${error.message}).`);
    }
    if (!isKind(parsed.kind)) {
        throw new WorkflowError(`${workflowFile(slug)}: unknown kind '${parsed.kind}'.`);
    }
    const stages = [...stagesOf(parsed.kind), 'done', 'abandoned'];
    if (!stages.includes(parsed.stage)) {
        throw new WorkflowError(`${workflowFile(slug)}: '${parsed.stage}' is not a stage of a ${parsed.kind}.`);
    }
    parsed.events ??= [];
    parsed.approvals ??= {};
    return parsed;
}
export function writeWorkflow(root, slug, workflow) {
    const file = path.join(root, workflowFile(slug));
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify(workflow, null, 4) + '\n');
}
export function activeSlug(root) {
    const file = path.join(root, ACTIVE);
    if (!fs.existsSync(file))
        return undefined;
    const slug = fs.readFileSync(file, 'utf8').trim();
    return slug || undefined;
}
export function setActive(root, slug) {
    const file = path.join(root, ACTIVE);
    if (slug === undefined) {
        fs.rmSync(file, { force: true });
        return;
    }
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, slug + '\n');
}
export function listEntries(root) {
    const dir = path.join(root, ARCHIVE());
    if (!fs.existsSync(dir))
        return [];
    return fs
        .readdirSync(dir, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => entry.name)
        .sort();
}
// The file holding a stage's document: `.svx` when there is one, else `.md`.
export function documentPath(root, slug, stage) {
    const svx = `${entryDir(slug)}/${stage}.svx`;
    return fs.existsSync(path.join(root, svx)) ? svx : `${entryDir(slug)}/${stage}.md`;
}
// A checklist's boxes are ticked at a later stage. The checklist is otherwise frozen, so its hash
// is taken with every box unticked.
export function documentHash(root, file, stage) {
    const abs = path.join(root, file);
    if (!fs.existsSync(abs))
        return undefined;
    let text = fs.readFileSync(abs, 'utf8');
    if (stageConfig(stage).checklist)
        text = text.replace(/^(\s*[-*] )\[[xX]\]/gm, '$1[ ]');
    return createHash('sha256').update(text).digest('hex');
}
export function checkboxes(text) {
    const ticked = (text.match(/^\s*[-*] \[[xX]\]/gm) ?? []).length;
    const open = (text.match(/^\s*[-*] \[ \]/gm) ?? []).length;
    return { ticked, open, total: ticked + open };
}
// The stages of an entry reached so far, in order: those approved at some point (stale or not),
// then the current one.
export function reached(workflow) {
    const stages = stagesOf(workflow.kind);
    if (workflow.stage == 'done')
        return stages;
    const out = stages.filter((stage) => workflow.approvals[stage]);
    if (workflow.stage != 'abandoned' && !out.includes(workflow.stage))
        out.push(workflow.stage);
    return out;
}
export function unlocked(workflow) {
    return new Set(reached(workflow).flatMap(unlocksOf));
}
export const contentHash = (text) => createHash('sha256').update(text.replaceAll('\r\n', '\n')).digest('hex');
export function readManifest(root) {
    const file = path.join(root, MANAGED);
    if (!fs.existsSync(file))
        return undefined;
    try {
        return JSON.parse(fs.readFileSync(file, 'utf8'));
    }
    catch {
        return undefined;
    }
}
// What a repository-relative path is, as far as the rules are concerned.
export function placeOf(rel) {
    rel = rel.split(path.sep).join('/');
    const archive = ARCHIVE();
    if (rel == ACTIVE || rel.startsWith('.workflow/'))
        return { area: 'workflow' };
    if (rel.startsWith(archive + '/')) {
        const [slug, file, ...rest] = rel.slice(archive.length + 1).split('/');
        if (!slug || !file)
            return { area: 'free' };
        if (file == 'workflow.json' && !rest.length)
            return { area: 'workflow' };
        const name = /^([a-z]+)\.(md|svx)$/.exec(file)?.[1];
        const stage = !rest.length && name && hasDocument(name) ? name : undefined;
        return { area: 'entry', slug, ...(stage ? { stage } : {}) };
    }
    if ((config().free ?? []).some((prefix) => rel.startsWith(prefix)))
        return { area: 'free' };
    const math = MATH();
    if (math && rel.startsWith(math + '/'))
        return { area: 'math' };
    if (rel.startsWith(CONTENT() + '/'))
        return { area: 'pages' };
    if (config().code.some((pattern) => pattern.test(rel)))
        return { area: 'code' };
    return { area: 'free' };
}
// What an entry may amend in another: a done entry's stage documents and its statements, to keep
// a reference in them true. The key is how the amending entry lists the file under `amends` in its
// `index.md`: its path under the content directory, without its extension.
export function amendKey(rel) {
    rel = rel.split(path.sep).join('/');
    const archive = ARCHIVE();
    if (!rel.startsWith(archive + '/'))
        return undefined;
    const [slug, file, ...rest] = rel.slice(archive.length + 1).split('/');
    if (!slug || !file)
        return undefined;
    if (!rest.length) {
        const name = /^([a-z]+)\.(md|svx)$/.exec(file)?.[1];
        return name && hasDocument(name) ? `archive/${slug}/${name}` : undefined;
    }
    const statement = file == 'statements' && rest.length == 1 && /^([a-z0-9-]+)\.md$/.exec(rest[0]);
    return statement ? `archive/${slug}/statements/${statement[1]}` : undefined;
}
// The entry an `amends` key names, and the stage of the document, when it is one.
export function amendTarget(key) {
    const document = /^archive\/([a-z0-9-]+)\/([a-z]+)$/.exec(key);
    if (document && hasDocument(document[2])) {
        return { slug: document[1], stage: document[2] };
    }
    const statement = /^archive\/([a-z0-9-]+)\/statements\/([a-z0-9-]+)$/.exec(key);
    return statement ? { slug: statement[1], name: statement[2] } : undefined;
}
// The file an `amends` key names.
export function amendPath(root, key) {
    const target = amendTarget(key);
    if (!target)
        return undefined;
    return target.stage
        ? documentPath(root, target.slug, target.stage)
        : `${entryDir(target.slug)}/statements/${target.name}.md`;
}
// Why `rel` may not be changed while `slug` (if any) is the entry being worked on, or undefined
// when it may. `editing` is set for a single edit about to be made, as opposed to a change already
// in a branch's history: an edit may only touch the current stage's document, while the history
// may hold any stage's. `managed` is what `mesearch-wf sync` wrote, which no edit changes.
export function refusal(rel, subject, editing, amending, managed) {
    rel = rel.split(path.sep).join('/');
    // Claude Code reaches the skills through `.claude/skills`, a link to `.agents/skills`.
    const canonical = rel.replace(/^\.claude\/skills\//, `${SKILLS}/`);
    if (editing && managed && Object.hasOwn(managed.files, canonical)) {
        return (`${canonical} is managed by ${managed.package} ${managed.version}: change it there, then ` +
            'run `mesearch-wf sync`.');
    }
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
                    if (editing || amending.amends.includes(key))
                        return undefined;
                    return (`${rel} belongs to entry '${place.slug}', which is done; list it as ` +
                        `'${key}' under \`amends\` in ${entryDir(subject.slug)}/index.md.`);
                }
                if (key && amending?.isDone(place.slug)) {
                    return (`${rel} belongs to entry '${place.slug}', which is done; '${subject.slug}' ` +
                        'may amend it only at its publish stage, to keep a reference in it true.');
                }
                return `${rel} belongs to entry '${place.slug}', but this branch works on '${subject.slug}'.`;
            }
            if (!editing || !place.stage)
                return undefined;
            const { workflow } = subject;
            if (!stagesOf(workflow.kind).includes(place.stage)) {
                return `a ${workflow.kind} has no ${place.stage} stage.`;
            }
            if (place.stage == workflow.stage)
                return undefined;
            if (stageConfig(place.stage).editableDuring?.includes(workflow.stage))
                return undefined;
            return (`${rel} is the ${place.stage} document, and the entry is at ${workflow.stage}. ` +
                `To change it, ask for \`/wf-reopen ${place.stage}\`.`);
        }
        default: {
            const area = place.area;
            if (!subject) {
                return area == 'code'
                    ? `${rel} is code, and no entry is active: start one with \`/wf-start\`.`
                    : undefined;
            }
            if (unlocked(subject.workflow).has(area))
                return undefined;
            const opener = stagesOf(subject.workflow.kind).filter((stage) => unlocksOf(stage).includes(area));
            if (!opener.length) {
                return `${rel} is ${describe(area)}, which a ${subject.workflow.kind} does not change.`;
            }
            return (`${rel} is ${describe(area)}, which opens at the ` +
                `${opener.join(' or ')} stage; '${subject.slug}' is at ${subject.workflow.stage}.`);
        }
    }
}
const describe = (area) => ({ math: 'a math note', code: 'code', pages: 'a documentation page' })[area];
// ---------------------------------------------------------------------------
// Git
export function git(root, ...args) {
    return execFileSync('git', args, {
        cwd: root,
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'pipe'],
    }).trim();
}
export function tryGit(root, ...args) {
    try {
        return git(root, ...args);
    }
    catch {
        return undefined;
    }
}
export function repoRoot(cwd) {
    const root = tryGit(cwd, 'rev-parse', '--show-toplevel');
    if (!root)
        throw new WorkflowError('not inside a git repository.');
    return root;
}
export const currentBranch = (root) => tryGit(root, 'branch', '--show-current') ?? '';
export const headCommit = (root) => tryGit(root, 'rev-parse', '--short', 'HEAD') ?? '';
export function isCommitted(root, rel) {
    return git(root, 'status', '--porcelain', '--', rel) == '';
}
// Every path that differs from `base`: committed since the merge base, staged, unstaged, or
// untracked. Undefined when `base` does not resolve.
export function changedSince(root, base) {
    const mergeBase = tryGit(root, 'merge-base', base, 'HEAD');
    if (!mergeBase)
        return undefined;
    const lines = [
        git(root, 'diff', '--name-only', mergeBase),
        git(root, 'ls-files', '--others', '--exclude-standard'),
    ].join('\n');
    return [...new Set(lines.split('\n').filter(Boolean))].sort();
}
// The default base: the remote's main branch, else the local one.
export function defaultBase(root) {
    for (const ref of ['origin/main', 'main']) {
        if (tryGit(root, 'rev-parse', '--verify', '--quiet', ref))
            return ref;
    }
    return undefined;
}
// ---------------------------------------------------------------------------
// Frontmatter, read only as far as the workflow needs: a list field such as
// `pages: [guide/first-model]`, written inline, as a block, or as a bracketed list spread over
// several lines, as Prettier writes one too long for a line.
export function frontmatterList(text, field) {
    const block = /^---\r?\n([\s\S]*?)\r?\n---/.exec(text)?.[1];
    if (!block)
        return undefined;
    const lines = block.split(/\r?\n/);
    const at = lines.findIndex((line) => line.startsWith(`${field}:`));
    if (at < 0)
        return undefined;
    const value = lines[at].slice(field.length + 1).trim();
    const unquote = (item) => item.trim().replace(/^(['"])(.*)\1$/, '$2');
    // A bracketed list opens on the field's line or, indented, on the next one, and runs to the
    // line that closes it.
    const opens = value.startsWith('[')
        ? at
        : !value && /^\s+\[/.test(lines[at + 1] ?? '')
            ? at + 1
            : -1;
    if (opens >= 0) {
        let flow = opens == at ? value : lines[opens].trim();
        for (let i = opens + 1; !flow.includes(']') && i < lines.length; i++) {
            flow += ' ' + lines[i].trim();
        }
        const inner = flow.replace(/^\[/, '').replace(/\].*$/, '').trim();
        return inner ? inner.split(',').map(unquote).filter(Boolean) : [];
    }
    if (value)
        return [unquote(value)];
    const items = [];
    for (const line of lines.slice(at + 1)) {
        const item = /^\s+-\s+(.*)$/.exec(line);
        if (!item)
            break;
        items.push(unquote(item[1]));
    }
    return items;
}
// The page a documentation path belongs to, as `<track>/<slug>`, if it is one.
export function pageKey(rel) {
    const match = new RegExp(`^${CONTENT()}/([^/]+)/([^/]+)/`).exec(rel);
    const math = config().math ?? 'math';
    if (!match || match[1] == math || match[1] == 'archive')
        return undefined;
    return `${match[1]}/${match[2]}`;
}
export function pageExists(root, key) {
    return ['md', 'svx'].some((ext) => fs.existsSync(path.join(root, CONTENT(), key, `index.${ext}`)));
}
export const hasTodo = (text) => /^>\s*\[!TODO\]/m.test(text);
