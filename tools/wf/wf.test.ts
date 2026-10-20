// The workflow tool, driven through its command line in throwaway repositories.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { run } from './cli.ts';
import { frontmatterList, placeOf } from './lib.ts';

function repo() {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'wf-'));
    const sh = (...args: string[]) =>
        execFileSync('git', args, { cwd: root, encoding: 'utf8', stdio: 'pipe' });
    sh('init', '-q', '-b', 'main');
    sh('config', 'user.email', 'test@example.org');
    sh('config', 'user.name', 'Test');
    write(root, 'README.md', 'skoro\n');
    write(root, 'site/content/guide/first-model/index.md', '---\ntitle: First\norder: 1\n---\n\nHi.\n');
    sh('add', '-A');
    sh('commit', '-qm', 'root');
    sh('checkout', '-qb', 'work');
    const wf = (...argv: string[]) => run(argv, { cwd: root });
    const commit = (message = 'step') => {
        sh('add', '-A');
        sh('commit', '-qm', message);
    };
    return { root, wf, commit, sh };
}

function write(root: string, rel: string, text: string) {
    fs.mkdirSync(path.dirname(path.join(root, rel)), { recursive: true });
    fs.writeFileSync(path.join(root, rel), text);
}

const read = (root: string, rel: string) => fs.readFileSync(path.join(root, rel), 'utf8');
const entry = 'site/content/archive/normal';
const workflow = (root: string) => JSON.parse(read(root, `${entry}/workflow.json`));

// Takes a feature through math and implementation, ready to plan.
function toPlan(r: ReturnType<typeof repo>) {
    r.wf('start', 'normal', '--kind', 'feature', '--title', 'The Normal');
    write(r.root, `${entry}/index.md`, '---\ntitle: The Normal\npages: []\n---\n\nA Gaussian.\n');
    write(r.root, `${entry}/math.md`, '# Math\n\n$x$\n');
    r.commit();
    assert.equal(r.wf('approve').code, 0);
    write(r.root, `${entry}/implementation.md`, 'How.\n\n## Tests\n\nThese.\n');
    r.commit();
    assert.equal(r.wf('approve').code, 0);
}

test('start opens one entry per branch, never on main', () => {
    const r = repo();
    const started = r.wf('start', 'normal', '--kind', 'feature');
    assert.equal(started.code, 0, started.stderr);
    assert.equal(read(r.root, '.workflow/active'), 'normal\n');
    assert.equal(workflow(r.root).stage, 'math');
    assert.match(read(r.root, `${entry}/index.md`), /\[!TODO\]/);

    const second = r.wf('start', 'other', '--kind', 'chore');
    assert.equal(second.code, 1);
    assert.match(second.stderr, /'normal' is active on this branch/);

    const main = repo();
    main.sh('checkout', '-q', 'main');
    assert.match(main.wf('start', 'x', '--kind', 'chore').stderr, /not on main/);
    assert.match(main.wf('start', 'Bad_Slug', '--kind', 'chore').stderr, /not a slug/);
    assert.match(main.wf('start', 'x', '--kind', 'epic').stderr, /'epic' is not a kind/);
});

test('status --expect refuses the wrong stage', () => {
    const r = repo();
    assert.equal(r.wf('status', '--expect', 'math').code, 3);
    r.wf('start', 'normal', '--kind', 'feature');
    assert.equal(r.wf('status', '--expect', 'math').code, 0);
    const wrong = r.wf('status', '--expect', 'plan');
    assert.equal(wrong.code, 3);
    assert.match(wrong.stderr, /'normal' is at math\. The step for it is \/wf-math/);
    assert.match(r.wf('status').stdout, /\[math\] → implementation/);
});

test('approve wants the document committed and free of open questions', () => {
    const r = repo();
    r.wf('start', 'normal', '--kind', 'feature');
    assert.match(r.wf('approve').stderr, /math\.md does not exist/);
    write(r.root, `${entry}/math.md`, '> [!TODO]\n> Which chart?\n');
    let result = r.wf('approve');
    assert.match(result.stderr, /uncommitted changes/);
    assert.match(result.stderr, /still has a \[!TODO\]/);
    write(r.root, `${entry}/math.md`, 'The chart is the exponential map.\n');
    r.commit();
    result = r.wf('approve');
    assert.equal(result.code, 0, result.stderr);
    const record = workflow(r.root);
    assert.equal(record.stage, 'implementation');
    assert.match(record.approvals.math.hash, /^[0-9a-f]{64}$/);

    write(r.root, `${entry}/implementation.md`, 'No tests yet.\n');
    r.commit();
    assert.match(r.wf('approve').stderr, /no "## Tests" section/);
});

test('an approved document is locked until its stage is reopened', () => {
    const r = repo();
    toPlan(r);
    assert.equal(r.wf('check').code, 0);
    write(r.root, `${entry}/math.md`, '# Math\n\n$y$\n');
    const locked = r.wf('check');
    assert.equal(locked.code, 1);
    assert.match(locked.stderr, /math\.md changed after it was approved/);

    assert.match(r.wf('reopen', 'math').stderr, /usage/);
    const reopened = r.wf('reopen', 'math', '--reason', 'the chart was wrong');
    assert.equal(reopened.code, 0, reopened.stderr);
    const record = workflow(r.root);
    assert.equal(record.stage, 'math');
    assert.equal(record.approvals.math.stale, true);
    assert.equal(record.approvals.implementation.stale, true);
    assert.equal(r.wf('check').code, 0, 'the reopened document may change');
    // The implementation stays locked until the entry comes back to it.
    write(r.root, `${entry}/implementation.md`, 'Changed.\n\n## Tests\n');
    assert.match(r.wf('check').stderr, /implementation\.md changed after it was approved/);
});

test('a document cannot be written before its stage', () => {
    const r = repo();
    r.wf('start', 'normal', '--kind', 'feature');
    write(r.root, `${entry}/plan.md`, '- [ ] Early.\n');
    assert.match(r.wf('check').stderr, /plan\.md was written before its stage/);
});

test('a draft written before a reopening waits, unchanged, for its stage', () => {
    const r = repo();
    r.wf('start', 'normal', '--kind', 'feature');
    write(r.root, `${entry}/math.md`, '# Math\n\n$x$\n');
    r.commit();
    assert.equal(r.wf('approve').code, 0);
    write(r.root, `${entry}/implementation.md`, 'A draft.\n');
    r.commit();
    assert.equal(r.wf('reopen', 'math', '--reason', 'the chart was wrong').code, 0);
    r.commit();
    assert.equal(r.wf('check').code, 0, 'the draft may stay');

    write(r.root, `${entry}/implementation.md`, 'A changed draft.\n');
    assert.match(r.wf('check').stderr, /implementation\.md changed after 'normal' was reopened/);
    write(r.root, `${entry}/implementation.md`, 'A draft.\n');
    write(r.root, `${entry}/plan.md`, '- [ ] Early.\n');
    assert.match(r.wf('check').stderr, /plan\.md was written before its stage/);
    fs.rmSync(path.join(r.root, `${entry}/plan.md`));

    write(r.root, `${entry}/math.md`, '# Math\n\n$y$\n');
    r.commit();
    assert.equal(r.wf('approve').code, 0);
    write(r.root, `${entry}/implementation.md`, 'A changed draft.\n');
    assert.equal(r.wf('check').code, 0, 'back at its stage, the draft may change');
});

test('code waits for the build stage', () => {
    const r = repo();
    write(r.root, 'crates/skoro/src/lib.rs', '// code\n');
    assert.match(r.wf('check').stderr, /is code, and no entry is active/);
    toPlan(r);
    assert.match(r.wf('check').stderr, /is code, which opens at the build stage/);
    write(r.root, `${entry}/plan.md`, '- [ ] Add it.\n- [ ] Test it.\n');
    r.commit();
    assert.equal(r.wf('approve').code, 0);
    assert.equal(r.wf('check').code, 0, 'code is open once the plan is approved');
});

test('code written before a reopening waits, unchanged, for the build stage', () => {
    const r = repo();
    toPlan(r);
    write(r.root, `${entry}/plan.md`, '- [ ] Add it.\n- [ ] Test it.\n');
    r.commit();
    assert.equal(r.wf('approve').code, 0);
    write(r.root, 'crates/skoro/src/lib.rs', '// code\n');
    r.commit();
    assert.equal(r.wf('reopen', 'implementation', '--reason', 'a test cannot be built').code, 0);
    r.commit();
    assert.equal(r.wf('check', '--base', 'main').code, 0, 'the code may stay');
    write(r.root, 'crates/skoro/src/lib.rs', '// changed code\n');
    assert.match(r.wf('check', '--base', 'main').stderr, /is code, which opens at the build stage/);
});

test('the plan is ticked during build, and build needs every box ticked', () => {
    const r = repo();
    toPlan(r);
    write(r.root, `${entry}/plan.md`, '- [ ] Add it.\n- [ ] Test it.\n');
    r.commit();
    r.wf('approve');
    write(r.root, `${entry}/plan.md`, '- [x] Add it.\n- [ ] Test it.\n');
    r.commit();
    assert.equal(r.wf('check').code, 0, 'ticking a box is not an edit');
    assert.match(r.wf('approve').stderr, /1 unticked items/);
    write(r.root, `${entry}/plan.md`, '- [x] Add it.\n- [x] Test it, more.\n');
    assert.match(r.wf('check').stderr, /plan\.md changed after it was approved/);
});

test('publish needs the changed pages listed, and finishes the entry', () => {
    const r = repo();
    toPlan(r);
    write(r.root, `${entry}/plan.md`, '- [x] Done.\n');
    r.commit();
    r.wf('approve');
    r.commit();
    r.wf('approve');
    r.commit();
    assert.equal(workflow(r.root).stage, 'publish');

    write(r.root, 'site/content/guide/first-model/index.md', '---\ntitle: First\norder: 1\n---\n\nNew.\n');
    r.commit();
    const unlisted = r.wf('approve', '--base', 'main');
    assert.match(unlisted.stderr, /page 'guide\/first-model' changed on this branch/);
    assert.match(r.wf('check', '--base', 'main', '--merge').stderr, /still at publish/);

    write(r.root, 'site/content/guide/other/index.md', '---\ntitle: Other\norder: 2\n---\n\nOld.\n');
    r.sh('add', '-A');
    r.sh('commit', '-qm', 'other');
    write(r.root, `${entry}/index.md`, '---\ntitle: The Normal\npages: [guide/first-model, guide/gone]\n---\n\nA Gaussian.\n');
    r.commit();
    assert.match(r.wf('approve', '--base', 'main').stderr, /'guide\/gone', which does not exist/);
    write(
        r.root,
        `${entry}/index.md`,
        '---\ntitle: The Normal\npages:\n  - guide/first-model\n  - guide/other\n---\n\nA Gaussian.\n',
    );
    r.commit();
    const done = r.wf('approve', '--base', 'main');
    assert.equal(done.code, 0, done.stderr);
    assert.equal(workflow(r.root).stage, 'done');
    assert.ok(!fs.existsSync(path.join(r.root, '.workflow/active')));
    r.commit();
    const merge = r.wf('check', '--base', 'main', '--merge');
    assert.equal(merge.code, 0, merge.stderr);
});

test('publish needs the changed crates listed', () => {
    const r = repo();
    toPlan(r);
    write(r.root, `${entry}/plan.md`, '- [x] Done.\n');
    r.commit();
    r.wf('approve');
    write(r.root, 'crates/normal/Cargo.toml', '[package]\nname = "normal"\n');
    write(r.root, 'crates/normal/src/lib.rs', 'pub struct Normal;\n');
    r.commit();
    r.wf('approve');
    r.commit();
    assert.equal(workflow(r.root).stage, 'publish');

    write(r.root, 'site/content/guide/first-model/index.md', '---\ntitle: First\norder: 1\n---\n\nNew.\n');
    const index = (crates: string, pages = 'guide/first-model') =>
        write(r.root, `${entry}/index.md`, `---\ntitle: The Normal\npages: [${pages}]\ncrates: ${crates}\n---\n\nA Gaussian.\n`);
    index('[]');
    r.commit();
    const unlisted = r.wf('approve', '--base', 'main').stderr;
    assert.match(unlisted, /crate 'normal' changed on this branch; list it/);
    assert.match(unlisted, /describe it in site\/content\/reference\/crates\/normal\/index\.md/);
    write(r.root, 'site/content/reference/crates/index.md', '---\ntitle: Crates\norder: 1\n---\n\nAll.\n');
    write(r.root, 'site/content/reference/crates/normal/index.md', 'The normal distribution.\n');
    index('[normal, gone]', 'guide/first-model, reference/crates');
    r.commit();
    const extra = r.wf('approve', '--base', 'main').stderr;
    assert.match(extra, /lists crate 'gone', which does not exist/);
    assert.match(extra, /lists crate 'gone', which this branch did not change/);
    index('[normal]', 'guide/first-model, reference/crates');
    r.commit();
    const done = r.wf('approve', '--base', 'main');
    assert.equal(done.code, 0, done.stderr);
});

test('publish refuses a listed page the branch did not change, and wants some page changed', () => {
    const r = repo();
    r.wf('start', 'charter', '--kind', 'decision');
    write(r.root, 'site/content/archive/charter/index.md', '---\ntitle: Charter\npages: []\n---\n\nWhy.\n');
    write(r.root, 'site/content/archive/charter/writeup.md', 'Because.\n');
    r.commit();
    assert.equal(r.wf('approve').code, 0);
    r.commit();
    assert.match(r.wf('approve', '--base', 'main').stderr, /no documentation page changed/);
    write(r.root, 'site/content/archive/charter/index.md', '---\ntitle: Charter\npages: [guide/first-model]\n---\n\nWhy.\n');
    r.commit();
    assert.match(
        r.wf('approve', '--base', 'main').stderr,
        /lists page 'guide\/first-model', which this branch did not change/,
    );
    write(r.root, 'site/content/archive/charter/index.md', '---\ntitle: Charter\npages: []\n---\n\nWhy.\n');
    write(r.root, 'site/content/landing.md', 'What skoro is.\n');
    r.commit();
    const done = r.wf('approve', '--base', 'main');
    assert.equal(done.code, 0, done.stderr);
});

test('a branch works on one entry', () => {
    const r = repo();
    r.wf('start', 'normal', '--kind', 'feature');
    write(r.root, 'site/content/archive/other/index.md', 'Other.\n');
    const result = r.wf('check', '--base', 'main');
    assert.match(result.stderr, /changes 2 entries/);
    assert.match(result.stderr, /belongs to entry 'other', but this branch works on 'normal'/);
});

test('abandon sets an entry aside, and resume takes it up where it stopped', () => {
    const r = repo();
    toPlan(r);
    assert.match(r.wf('abandon').stderr, /usage/);
    assert.equal(r.wf('abandon', '--reason', 'not now').code, 0);
    assert.equal(workflow(r.root).stage, 'abandoned');
    assert.ok(!fs.existsSync(path.join(r.root, '.workflow/active')));
    assert.equal(r.wf('check').code, 0, 'abandoned documents stay as approved');
    assert.match(r.wf('start', 'normal', '--kind', 'feature').stderr, /wf resume/);
    assert.equal(r.wf('resume', 'normal').code, 0);
    assert.equal(workflow(r.root).stage, 'plan');
});

test('a decision is written up and published', () => {
    const r = repo();
    const result = r.wf('start', 'design', '--kind', 'decision');
    assert.equal(result.code, 0, result.stderr);
    assert.match(result.stdout, /Next: \/wf-writeup/);
    assert.equal(r.wf('path').stdout.trim(), 'site/content/archive/design/writeup.md');
    assert.match(r.wf('path', 'math').stderr, /has no document/);
});

test('guard refuses edits the stage does not allow', () => {
    const r = repo();
    const guard = (file: string) =>
        run(['guard'], {
            cwd: r.root,
            stdin: JSON.stringify({ cwd: r.root, tool_input: { file_path: file } }),
        });
    assert.equal(guard('crates/skoro/src/lib.rs').code, 2);
    assert.equal(guard('site/content/guide/first-model/index.md').code, 0);
    assert.equal(guard('/etc/hosts').code, 0, 'outside the repository');
    r.wf('start', 'normal', '--kind', 'feature');
    assert.equal(guard(`${entry}/math.md`).code, 0);
    assert.equal(guard(`${entry}/index.md`).code, 0);
    const early = guard(`${entry}/plan.md`);
    assert.equal(early.code, 2);
    assert.match(early.stderr, /ask for `\/wf-reopen plan`/);
    assert.equal(guard(`${entry}/workflow.json`).code, 2);
    assert.equal(guard('.workflow/active').code, 2);
    assert.equal(guard('site/content/guide/first-model/index.md').code, 2, 'pages open at publish');
    assert.equal(guard('site/content/math/retraction/index.md').code, 0);
    assert.equal(guard(path.join(r.root, 'crates/skoro/src/lib.rs')).code, 2);
    assert.equal(guard('site/src/lib/site.ts').code, 0, 'the site itself is not gated');
});

test('frontmatter lists and paths are read as the rules need', () => {
    assert.deepEqual(frontmatterList('---\npages: [a/b, "c/d"]\n---\n', 'pages'), ['a/b', 'c/d']);
    assert.deepEqual(frontmatterList('---\npages:\n  - a/b\n  - c/d\nx: 1\n---\n', 'pages'), ['a/b', 'c/d']);
    assert.deepEqual(frontmatterList('---\npages: []\n---\n', 'pages'), []);
    assert.equal(frontmatterList('no frontmatter', 'pages'), undefined);
    assert.deepEqual(placeOf('site/content/archive/normal/math.svx'), {
        area: 'entry',
        slug: 'normal',
        stage: 'math',
    });
    assert.deepEqual(placeOf('site/content/archive/normal/statements/math.md'), {
        area: 'entry',
        slug: 'normal',
    });
    assert.deepEqual(placeOf('Cargo.toml'), { area: 'code' });
    assert.deepEqual(placeOf('site/content/landing.md'), { area: 'pages' });
});
