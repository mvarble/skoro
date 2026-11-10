// The entry point `bin/figures` runs: `bin/figures <data directory> [<output directory>]`. It reads
// `pose_graph.json` and `sliding_window.json`, which the examples write when run with
// `SKORO_FIGURES=<data directory>`, and writes the figures, by default to `site/static/figures/`.
import fs from 'node:fs';
import path from 'node:path';

import { calibrationFigure, poseGraphFigure, slidingWindowFigure } from './lib.ts';
import type { PoseGraph, SlidingWindow } from './lib.ts';

const [data, out = path.join(import.meta.dirname, '../../site/static/figures')] = process.argv.slice(2);
if (!data) {
    process.stderr.write('usage: bin/figures <data directory> [<output directory>]\n');
    process.exit(2);
}
const read = <T>(name: string): T => JSON.parse(fs.readFileSync(path.join(data, `${name}.json`), 'utf8')) as T;
fs.mkdirSync(out, { recursive: true });
const write = (name: string, svg: string) => {
    const file = path.join(out, `${name}.svg`);
    fs.writeFileSync(file, svg + '\n');
    process.stdout.write(`wrote ${file}\n`);
};
const poseGraph = read<PoseGraph>('pose_graph');
write('pose-graph', poseGraphFigure(poseGraph));
write('calibration', calibrationFigure(poseGraph));
write('sliding-window', slidingWindowFigure(read<SlidingWindow>('sliding_window')));
