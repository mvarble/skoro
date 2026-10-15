// `bin/wf`: the workflow's command line. See `.agents/reference/workflow.md`.
import fs from 'node:fs';
import path from 'node:path';

import {
    ACTIVE,
    activeSlug,
    changedSince,
    checkboxes,
    currentBranch,
    defaultBase,
    documentHash,
    documentPath,
    entryDir,
    frontmatterList,
    git,
    hasDocument,
    hasTodo,
    headCommit,
    isCommitted,
    isKind,
    KINDS,
    listEntries,
    pageExists,
    pageKey,
    placeOf,
    readWorkflow,
    refusal,
    repoRoot,
    setActive,
    today,
    tryGit,
    type Kind,
    type Stage,
    type Workflow,
    WorkflowError,
    workflowFile,
    writeWorkflow,
} from './lib.ts';

export interface Result {
    code: number;
    stdout: string;
    stderr: string;
}

const USAGE = `usage: bin/wf <command>

  status [--expect <stage>]       where the active entry stands, and what comes next
  path [<stage>]                  the file of a stage's document (the current stage's by default)
  start <slug> --kind <kind> [--title <title>] [--revises <slug>,...]
                                  open an entry on this branch; kinds: ${Object.keys(KINDS).join(', ')}
  approve                         accept the current stage and move to the next
  reopen <stage> --reason <text>  go back to an earlier stage
  abandon --reason <text>         set the active entry aside; its documents stay
  resume <slug>                   take up an abandoned entry again
  check [--base <ref>] [--merge]  verify the rules (run by CI); --merge also requires no active entry
  guard                           a Claude Code PreToolUse hook: refuses edits the stage does not allow

approve, reopen, abandon and resume are the human's to ask for.`;

class Output {
    out: string[] = [];
    err: string[] = [];
    say = (line = '') => void this.out.push(line);
    warn = (line: string) => void this.err.push(line);
    result = (code: number): Result => ({
        code,
        stdout: this.out.length ? this.out.join('\n') + '\n' : '',
        stderr: this.err.length ? this.err.join('\n') + '\n' : '',
    });
}

export function run(argv: string[], options: { cwd: string; stdin?: string }): Result {
    const io = new Output();
    try {
        const [command, ...rest] = argv;
        const args = parseArgs(rest);
        const root = repoRoot(options.cwd);
        switch (command) {
            case 'status':
                return status(root, args, io);
            case 'path':
                return pathOf(root, args, io);
            case 'start':
                return start(root, args, io);
            case 'approve':
                return approve(root, args, io);
            case 'reopen':
                return reopen(root, args, io);
            case 'abandon':
                return abandon(root, args, io);
            case 'resume':
                return resume(root, args, io);
            case 'check':
                return check(root, args, io);
            case 'guard':
                return guard(root, options.stdin ?? '', io);
            case undefined:
            case 'help':
            case '--help':
                io.say(USAGE);
                return io.result(0);
            default:
                io.warn(`wf: no command '${command}'.\n\n${USAGE}`);
                return io.result(2);
        }
    } catch (error) {
        if (!(error instanceof WorkflowError)) throw error;
        io.warn(`wf: ${error.message}`);
        return io.result(1);
    }
}

interface Args {
    positional: string[];
    flags: Record<string, string | true>;
}

function parseArgs(argv: string[]): Args {
    const args: Args = { positional: [], flags: {} };
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i]!;
        if (!arg.startsWith('--')) {
            args.positional.push(arg);
            continue;
        }
        const [name, inline] = arg.slice(2).split(/=(.*)/s);
        const next = argv[i + 1];
        if (inline !== undefined) args.flags[name!] = inline;
        else if (next !== undefined && !next.startsWith('--')) {
            args.flags[name!] = next;
            i++;
        } else args.flags[name!] = true;
    }
    return args;
}

const flag = (args: Args, name: string) =>
    typeof args.flags[name] == 'string' ? (args.flags[name] as string) : undefined;

function requireActive(root: string): { slug: string; workflow: Workflow } {
    const slug = activeSlug(root);
    if (!slug) {
        throw new WorkflowError('no entry is active on this branch; start one with `/wf-start`.');
    }
    return { slug, workflow: readWorkflow(root, slug) };
}

function stageOf(workflow: Workflow): Stage {
    if (workflow.stage == 'done' || workflow.stage == 'abandoned') {
        throw new WorkflowError(`the entry is ${workflow.stage}.`);
    }
    return workflow.stage;
}

// The skill that works on each stage (`.agents/skills/wf-*`).
const SKILL: Record<Stage, string> = {
    math: '/wf-math',
    implementation: '/wf-implementation',
    plan: '/wf-plan',
    build: '/wf-build',
    publish: '/wf-publish',
    writeup: '/wf-writeup',
};

// ---------------------------------------------------------------------------
// status, path

function status(root: string, args: Args, io: Output): Result {
    const expect = flag(args, 'expect');
    const slug = activeSlug(root);
    if (!slug) {
        if (expect) {
            io.warn(
                `wf: this step needs an entry at the ${expect} stage, and none is active on this ` +
                    'branch. Start one with `/wf-start`.',
            );
            return io.result(3);
        }
        io.say('No entry is active on this branch. Start one with `/wf-start <slug> <kind>`.');
        const entries = listEntries(root);
        if (entries.length) {
            io.say();
            io.say('Entries:');
            for (const entry of entries) {
                try {
                    const workflow = readWorkflow(root, entry);
                    io.say(`  ${entry.padEnd(28)} ${workflow.kind.padEnd(9)} ${workflow.stage}`);
                } catch (error) {
                    io.say(`  ${entry.padEnd(28)} (${(error as Error).message})`);
                }
            }
        }
        return io.result(0);
    }

    const workflow = readWorkflow(root, slug);
    if (expect && workflow.stage != expect) {
        io.warn(
            `wf: this step is for the ${expect} stage, but '${slug}' is at ${workflow.stage}. ` +
                (workflow.stage in SKILL
                    ? `The step for it is ${SKILL[workflow.stage as Stage]}.`
                    : ''),
        );
        return io.result(3);
    }

    const stages = KINDS[workflow.kind];
    const line = stages
        .map((stage) => {
            if (stage == workflow.stage) return `[${stage}]`;
            const approval = workflow.approvals[stage];
            if (approval && !approval.stale) return `${stage} ✓`;
            if (approval?.stale) return `${stage} (reopened)`;
            return stage;
        })
        .join(' → ');
    io.say(`Entry:    ${slug} (${workflow.kind}), ${entryDir(slug)}/`);
    io.say(`Stage:    ${line}`);
    const stage = workflow.stage as Stage;
    if (hasDocument(stage)) io.say(`Document: ${documentPath(root, slug, stage)}`);
    const plan = path.join(root, documentPath(root, slug, 'plan'));
    if (stages.includes('plan') && fs.existsSync(plan)) {
        const boxes = checkboxes(fs.readFileSync(plan, 'utf8'));
        io.say(`Plan:     ${boxes.ticked} of ${boxes.total} done`);
    }
    io.say(
        `Next:     work with ${SKILL[stage]}; when it is right, \`/wf-approve\` moves on` +
            (stage == stages.at(-1) ? ' and finishes the entry.' : ` to ${nextStage(workflow)}.`),
    );
    return io.result(0);
}

function nextStage(workflow: Workflow): string {
    const stages = KINDS[workflow.kind];
    return stages[stages.indexOf(workflow.stage as Stage) + 1] ?? 'done';
}

function pathOf(root: string, args: Args, io: Output): Result {
    const { slug, workflow } = requireActive(root);
    const stage = (args.positional[0] ?? workflow.stage) as Stage;
    if (!hasDocument(stage) || !KINDS[workflow.kind].includes(stage)) {
        throw new WorkflowError(`the ${stage} stage of a ${workflow.kind} has no document.`);
    }
    io.say(documentPath(root, slug, stage));
    return io.result(0);
}

// ---------------------------------------------------------------------------
// start

function start(root: string, args: Args, io: Output): Result {
    const [slug] = args.positional;
    const kind = flag(args, 'kind');
    if (!slug || !kind) throw new WorkflowError('usage: wf start <slug> --kind <kind>');
    if (!isKind(kind)) {
        throw new WorkflowError(`'${kind}' is not a kind; one of ${Object.keys(KINDS).join(', ')}.`);
    }
    if (!/^[a-z0-9]+(-[a-z0-9]+)*$/.test(slug)) {
        throw new WorkflowError(`'${slug}' is not a slug: lowercase words joined by hyphens.`);
    }
    const active = activeSlug(root);
    if (active) {
        const workflow = readWorkflow(root, active);
        throw new WorkflowError(
            `'${active}' is active on this branch (at ${workflow.stage}). Finish it, abandon it, ` +
                'or start the new entry on another branch.',
        );
    }
    const branch = currentBranch(root);
    if (branch == 'main') {
        throw new WorkflowError('entries are worked on in a branch of their own, not on main.');
    }
    if (fs.existsSync(path.join(root, entryDir(slug)))) {
        throw new WorkflowError(
            `${entryDir(slug)} exists already. An abandoned entry is taken up with \`wf resume\`; ` +
                'a finished one is revised by a new entry with `--revises`.',
        );
    }
    const revises = (flag(args, 'revises') ?? '').split(',').filter(Boolean);
    for (const earlier of revises) {
        if (!fs.existsSync(path.join(root, workflowFile(earlier)))) {
            throw new WorkflowError(`--revises names '${earlier}', which is not an entry.`);
        }
    }

    const title = flag(args, 'title') ?? slug;
    const first = KINDS[kind][0]!;
    const frontmatter = [
        '---',
        `title: ${JSON.stringify(title)}`,
        'pages: []',
        ...(revises.length ? [`revises: [${revises.join(', ')}]`] : []),
        '---',
    ];
    const body = ['> [!TODO]', '> Say in a paragraph what this entry sets out to do, and why.'];
    fs.mkdirSync(path.join(root, entryDir(slug)), { recursive: true });
    fs.writeFileSync(
        path.join(root, entryDir(slug), 'index.md'),
        [...frontmatter, '', ...body, ''].join('\n'),
    );
    writeWorkflow(root, slug, {
        kind,
        stage: first,
        started: today(),
        events: [{ date: today(), action: 'start', stage: first }],
        approvals: {},
    });
    setActive(root, slug);

    io.say(`Started '${slug}', a ${kind}, at the ${first} stage.`);
    io.say(`  ${entryDir(slug)}/index.md`);
    io.say(`  ${workflowFile(slug)}`);
    io.say(`  ${ACTIVE}`);
    io.say(`Next: ${SKILL[first]}.`);
    return io.result(0);
}

// ---------------------------------------------------------------------------
// approve

function approve(root: string, args: Args, io: Output): Result {
    const { slug, workflow } = requireActive(root);
    const stage = stageOf(workflow);
    const problems = [
        ...lockProblems(root, slug, workflow),
        ...readiness(root, slug, workflow, stage, flag(args, 'base')),
    ];
    if (problems.length) {
        throw new WorkflowError(
            `the ${stage} stage of '${slug}' is not ready:\n${problems.map((p) => `  - ${p}`).join('\n')}`,
        );
    }

    const commit = headCommit(root);
    const doc = hasDocument(stage) ? documentPath(root, slug, stage) : undefined;
    workflow.approvals[stage] = {
        date: today(),
        commit,
        ...(doc ? { hash: documentHash(root, doc, stage) } : {}),
    };
    workflow.events.push({ date: today(), action: 'approve', stage, commit });
    const next = nextStage(workflow);
    if (next == 'done') {
        workflow.stage = 'done';
        workflow.finished = today();
        setActive(root, undefined);
    } else {
        workflow.stage = next as Stage;
    }
    writeWorkflow(root, slug, workflow);

    io.say(`Approved the ${stage} stage of '${slug}' at ${commit}.`);
    if (next == 'done') {
        io.say(`'${slug}' is done, and no entry is active on this branch.`);
        io.say(`Commit ${workflowFile(slug)} and the removal of ${ACTIVE}.`);
    } else {
        io.say(`Now at ${next}: ${SKILL[next as Stage]}. Commit ${workflowFile(slug)}.`);
    }
    return io.result(0);
}

// What keeps a stage from being approved, beyond the locks.
function readiness(
    root: string,
    slug: string,
    workflow: Workflow,
    stage: Stage,
    base: string | undefined,
): string[] {
    const problems: string[] = [];
    const read = (rel: string) => fs.readFileSync(path.join(root, rel), 'utf8');

    if (hasDocument(stage)) {
        const doc = documentPath(root, slug, stage);
        if (!fs.existsSync(path.join(root, doc))) return [`${doc} does not exist.`];
        const text = read(doc);
        if (!isCommitted(root, doc)) problems.push(`${doc} has uncommitted changes.`);
        if (!text.replace(/^---[\s\S]*?\n---/, '').trim()) problems.push(`${doc} is empty.`);
        if (hasTodo(text)) problems.push(`${doc} still has a [!TODO]: open questions remain.`);
        if (stage == 'implementation' && !/^#{2,3} Tests\b/m.test(text)) {
            problems.push(`${doc} has no "## Tests" section.`);
        }
        if (stage == 'plan' && checkboxes(text).total == 0) {
            problems.push(`${doc} has no checklist (\`- [ ] ...\`).`);
        }
    }

    if (stage == 'build') {
        const plan = documentPath(root, slug, 'plan');
        if (fs.existsSync(path.join(root, plan))) {
            const boxes = checkboxes(read(plan));
            if (boxes.open) problems.push(`${plan} has ${boxes.open} unticked items.`);
        }
        const dirty = uncommitted(root);
        if (dirty) problems.push(`the working tree has uncommitted changes:\n${dirty}`);
    }

    if (stage == 'publish') {
        const index = `${entryDir(slug)}/index.md`;
        const text = read(index);
        if (hasTodo(text)) problems.push(`${index} still has a [!TODO].`);
        if (!isCommitted(root, index)) problems.push(`${index} has uncommitted changes.`);
        const listed = frontmatterList(text, 'pages') ?? [];
        for (const key of listed) {
            if (!pageExists(root, key)) problems.push(`${index} lists page '${key}', which does not exist.`);
        }
        const ref = base ?? defaultBase(root);
        const changed = ref ? changedSince(root, ref) : undefined;
        if (changed) {
            const pages = new Set(changed.map(pageKey).filter((key) => key !== undefined));
            for (const key of pages) {
                if (!listed.includes(key) && pageExists(root, key)) {
                    problems.push(`page '${key}' changed on this branch; list it under \`pages\` in ${index}.`);
                }
            }
            for (const key of listed) {
                if (!pages.has(key)) problems.push(`${index} lists page '${key}', which this branch did not change.`);
            }
            const landing = changed.some((rel) => /^site\/content\/landing\.(md|svx)$/.test(rel));
            if (workflow.kind != 'chore' && !pages.size && !landing) {
                problems.push('no documentation page changed: say what this entry made true in the docs.');
            }
        }
    }
    return problems;
}

// What `git status` lists, indented to sit under a problem.
function uncommitted(root: string): string {
    return git(root, 'status', '--porcelain')
        .split('\n')
        .filter(Boolean)
        .map((line) => `    ${line}`)
        .join('\n');
}

// ---------------------------------------------------------------------------
// reopen, abandon, resume

function reopen(root: string, args: Args, io: Output): Result {
    const { slug, workflow } = requireActive(root);
    const current = stageOf(workflow);
    const stage = args.positional[0] as Stage | undefined;
    const reason = flag(args, 'reason');
    const stages = KINDS[workflow.kind];
    if (!stage || !reason) throw new WorkflowError('usage: wf reopen <stage> --reason <text>');
    if (!stages.includes(stage) || stages.indexOf(stage) >= stages.indexOf(current)) {
        throw new WorkflowError(
            `'${stage}' is not a stage of this ${workflow.kind} before ${current}.`,
        );
    }
    for (const earlier of stages.slice(stages.indexOf(stage), stages.indexOf(current))) {
        const approval = workflow.approvals[earlier];
        if (approval) approval.stale = true;
    }
    workflow.stage = stage;
    workflow.events.push({ date: today(), action: 'reopen', stage, reason, commit: headCommit(root) });
    writeWorkflow(root, slug, workflow);
    io.say(`Reopened '${slug}' at ${stage}: ${reason}`);
    if (hasDocument(stage)) {
        io.say(
            `Revise ${documentPath(root, slug, stage)} with ${SKILL[stage]}, and say what changed ` +
                'and why under its "## Revisions" heading.',
        );
    }
    io.say(`Every stage from ${stage} to ${current} is approved again on the way back.`);
    return io.result(0);
}

function abandon(root: string, args: Args, io: Output): Result {
    const { slug, workflow } = requireActive(root);
    const stage = stageOf(workflow);
    const reason = flag(args, 'reason');
    if (!reason) throw new WorkflowError('usage: wf abandon --reason <text>');
    workflow.events.push({ date: today(), action: 'abandon', stage, reason, commit: headCommit(root) });
    workflow.stage = 'abandoned';
    writeWorkflow(root, slug, workflow);
    setActive(root, undefined);
    io.say(`Set '${slug}' aside at ${stage}. Its documents stay in the archive; \`wf resume ${slug}\` takes it up again.`);
    return io.result(0);
}

function resume(root: string, args: Args, io: Output): Result {
    const [slug] = args.positional;
    if (!slug) throw new WorkflowError('usage: wf resume <slug>');
    const active = activeSlug(root);
    if (active) throw new WorkflowError(`'${active}' is active on this branch.`);
    const workflow = readWorkflow(root, slug);
    if (workflow.stage != 'abandoned') {
        throw new WorkflowError(`'${slug}' is ${workflow.stage}, not abandoned.`);
    }
    const left = workflow.events.findLast((event) => event.action == 'abandon')?.stage ?? KINDS[workflow.kind][0]!;
    workflow.stage = left;
    workflow.events.push({ date: today(), action: 'resume', stage: left });
    writeWorkflow(root, slug, workflow);
    setActive(root, slug);
    io.say(`Took up '${slug}' again at ${left}: ${SKILL[left]}.`);
    return io.result(0);
}

// ---------------------------------------------------------------------------
// check, guard

// Documents changed after they were approved, documents written before their
// stage, and drafts changed while the entry was reopened at an earlier stage.
function lockProblems(root: string, slug: string, workflow: Workflow): string[] {
    const problems: string[] = [];
    const stages = KINDS[workflow.kind];
    const current = workflow.stage;
    for (const stage of stages.filter(hasDocument)) {
        const doc = documentPath(root, slug, stage);
        const exists = fs.existsSync(path.join(root, doc));
        const approval = workflow.approvals[stage];
        if (stage == current) continue;
        if (approval?.hash) {
            if (documentHash(root, doc, stage) != approval.hash) {
                problems.push(
                    `${doc} changed after it was approved at ${approval.commit}; reopen ${stage} to change it.`,
                );
            }
        } else if (exists && current != 'done') {
            // A draft the entry had reached before it was reopened waits, unchanged, for its
            // stage to come round again.
            const reopened = workflow.events.findLast((event) => event.action == 'reopen')?.commit;
            const kept = reopened && tryGit(root, 'cat-file', '-e', `${reopened}:${doc}`) != undefined;
            if (!kept) {
                problems.push(`${doc} was written before its stage; '${slug}' is at ${current}.`);
            } else if (tryGit(root, 'diff', '--quiet', reopened, '--', doc) == undefined) {
                problems.push(
                    `${doc} changed after '${slug}' was reopened at ${reopened}; it waits for its stage.`,
                );
            }
        }
        if (current == 'done' && (!approval || approval.stale)) {
            problems.push(`'${slug}' is done, but its ${stage} stage was never approved.`);
        }
    }
    return problems;
}

// Whether `rel` was in the tree at `commit` and is unchanged since.
function keptSince(root: string, commit: string, rel: string): boolean {
    return (
        tryGit(root, 'cat-file', '-e', `${commit}:${rel}`) != undefined &&
        tryGit(root, 'diff', '--quiet', commit, '--', rel) != undefined
    );
}

function check(root: string, args: Args, io: Output): Result {
    const problems: string[] = [];
    const active = activeSlug(root);
    const workflows = new Map<string, Workflow>();
    for (const slug of listEntries(root)) {
        try {
            const workflow = readWorkflow(root, slug);
            workflows.set(slug, workflow);
            problems.push(...lockProblems(root, slug, workflow));
        } catch (error) {
            problems.push((error as Error).message);
        }
    }
    if (active) {
        const workflow = workflows.get(active);
        if (!workflow) problems.push(`${ACTIVE} names '${active}', which is not an entry.`);
        else if (workflow.stage == 'done' || workflow.stage == 'abandoned') {
            problems.push(`${ACTIVE} names '${active}', which is ${workflow.stage}.`);
        }
    }

    const base = flag(args, 'base') ?? defaultBase(root);
    const changed = base ? changedSince(root, base) : undefined;
    if (!changed) {
        io.say(`Skipped the branch rules: no base to compare with${base ? ` ('${base}')` : ''}.`);
    } else {
        const touched = [
            ...new Set(
                changed
                    .map(placeOf)
                    .flatMap((place) => (place.area == 'entry' ? [place.slug] : [])),
            ),
        ];
        if (touched.length > 1) {
            problems.push(`this branch changes ${touched.length} entries (${touched.join(', ')}); one branch works on one.`);
        }
        const slug = active ?? touched[0];
        const workflow = slug ? workflows.get(slug) : undefined;
        const subject = slug && workflow ? { slug, workflow } : undefined;
        // Code, math notes and pages written before a reopening wait, unchanged, for the stage
        // that opens them to come round again, as a draft does.
        const reopened = subject?.workflow.events.findLast((event) => event.action == 'reopen')?.commit;
        for (const rel of changed) {
            const why = refusal(rel, subject, false);
            if (!why) continue;
            if (reopened && placeOf(rel).area != 'entry' && keptSince(root, reopened, rel)) continue;
            problems.push(why);
        }
    }
    if (args.flags.merge && active) {
        const stage = workflows.get(active)?.stage;
        problems.push(
            `'${active}' is still at ${stage}: finish it with \`/wf-approve\` or set it aside with ` +
                '`/wf-abandon` before merging.',
        );
    }

    if (problems.length) {
        io.warn(`wf check found ${problems.length} problem${problems.length == 1 ? '' : 's'}:`);
        for (const problem of problems) io.warn(`  - ${problem}`);
        return io.result(1);
    }
    io.say(`wf check: ok${active ? ` ('${active}' at ${workflows.get(active)?.stage})` : ''}.`);
    return io.result(0);
}

// A Claude Code PreToolUse hook. The tool call arrives on stdin as JSON; exit
// code 2 refuses it and shows stderr to the agent.
function guard(root: string, stdin: string, io: Output): Result {
    let input: { cwd?: string; tool_input?: Record<string, unknown> };
    try {
        input = JSON.parse(stdin || '{}');
    } catch {
        return io.result(0);
    }
    const target = input.tool_input?.file_path ?? input.tool_input?.notebook_path;
    if (typeof target != 'string') return io.result(0);
    const rel = path.relative(root, path.resolve(input.cwd ?? root, target));
    if (rel.startsWith('..') || path.isAbsolute(rel)) return io.result(0);

    let subject: { slug: string; workflow: Workflow } | undefined;
    const slug = activeSlug(root);
    if (slug) {
        try {
            subject = { slug, workflow: readWorkflow(root, slug) };
        } catch (error) {
            io.warn(`wf guard: ${(error as Error).message}`);
            return io.result(0);
        }
    }
    const why = refusal(rel, subject, true);
    if (!why) return io.result(0);
    io.warn(`wf: ${why}`);
    return io.result(2);
}
