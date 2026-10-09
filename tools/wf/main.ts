// The entry point `bin/wf` runs.
import fs from 'node:fs';

import { run } from './cli.ts';

const argv = process.argv.slice(2);
// Only the hook reads stdin, and it is always given one.
const stdin = argv[0] == 'guard' ? fs.readFileSync(0, 'utf8') : undefined;
const result = run(argv, { cwd: process.cwd(), stdin });
process.stdout.write(result.stdout);
process.stderr.write(result.stderr);
process.exitCode = result.code;
