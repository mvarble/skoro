// skoro's crate rules, run through the vendored engine with skoro's own config, in a throwaway
// repository.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// @ts-expect-error: the engine is vendored as JavaScript, without its declarations.
import { run } from '../../.workflow/engine/cli.js';
import config from '../../workflow.config.ts';
import { crateOf } from './crates.ts';

function write(root: string, rel: string, text: string) {
    fs.mkdirSync(path.dirname(path.join(root, rel)), { recursive: true });
    fs.writeFileSync(path.join(root, rel), text);
}

function repo() {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'skoro-crates-'));
    const sh = (...args: string[]) =>
        execFileSync('git', args, { cwd: root, encoding: 'utf8', stdio: 'pipe' });
    sh('init', '-q', '-b', 'main');
    sh('config', 'user.email', 'test@example.org');
    sh('config', 'user.name', 'Test');
    write(root, 'site/content/guide/first-model/index.md', '---\ntitle: First\norder: 1\n---\n\nHi.\n');
    sh('add', '-A');
    sh('commit', '-qm', 'root');
    sh('checkout', '-qb', 'work');
    const wf = (...argv: string[]): { code: number; stderr: string } =>
        run(argv, { cwd: root, config });
    const commit = () => {
        sh('add', '-A');
        sh('commit', '-qm', 'step');
    };
    return { root, wf, commit };
}

const entry = 'site/content/archive/normal';

test('a path belongs to the crate whose directory holds it', () => {
    assert.equal(crateOf('crates/skoro-tensor/src/lib.rs'), 'skoro-tensor');
    assert.equal(crateOf('Cargo.toml'), undefined);
});

test('publish needs the changed crates listed, and each described', () => {
    const r = repo();
    r.wf('start', 'normal', '--kind', 'feature', '--title', 'The Normal');
    write(r.root, `${entry}/index.md`, '---\ntitle: The Normal\npages: []\n---\n\nA Gaussian.\n');
    write(r.root, `${entry}/math.md`, '# Math\n\n$x$\n');
    r.commit();
    assert.equal(r.wf('approve').code, 0);
    write(r.root, `${entry}/implementation.md`, 'How.\n\n## Tests\n\nThese.\n');
    r.commit();
    assert.equal(r.wf('approve').code, 0);
    write(r.root, `${entry}/plan.md`, '- [x] Done.\n');
    r.commit();
    assert.equal(r.wf('approve').code, 0);
    write(r.root, 'crates/normal/Cargo.toml', '[package]\nname = "normal"\n');
    write(r.root, 'crates/normal/src/lib.rs', 'pub struct Normal;\n');
    r.commit();
    assert.equal(r.wf('approve').code, 0);
    r.commit();

    write(r.root, 'site/content/guide/first-model/index.md', '---\ntitle: First\norder: 1\n---\n\nNew.\n');
    const index = (crates: string, pages = 'guide/first-model') =>
        write(
            r.root,
            `${entry}/index.md`,
            `---\ntitle: The Normal\npages: [${pages}]\ncrates: ${crates}\n---\n\nA Gaussian.\n`,
        );
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
