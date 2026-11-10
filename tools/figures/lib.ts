// The examples' figures, drawn as self-contained SVG cards from the data the examples write when
// `SKORO_FIGURES` names a directory: the pose graph's map, the calibration of its marginals, and
// the sliding window's path and errors. `main.ts` is the command line over it; the README and the
// site show what it writes to `site/static/figures/`.
//
// Each card carries its own light surface and a hairline border, so that it reads the same on a
// light page and a dark one. The series take the first three slots of a categorical palette
// validated on that surface for every pair: blue for the estimate, orange for dead reckoning, aqua
// for the landmarks and the batch. Aqua is below 3:1 against the surface, so every aqua mark is
// named by a legend or a direct label. The truth is drawn in neutral ink, dashed, never in a
// series colour.

// ---------------------------------------------------------------------------
// The data the examples write

export type Vec3 = [number, number, number];

/** A quantity of one row: its truth, its start, its mode, and its position's covariance. */
export interface Estimated {
    truth: Vec3;
    start: Vec3;
    mode: Vec3;
    /** 3 × 3, by rows, in the world's axes. */
    covariance: number[];
}

export interface Calibration {
    coverage: number;
    tolerance: number;
}

export interface PoseGraph {
    rows: number;
    poses: Estimated[];
    landmarks: Estimated[];
    /** `(k, j)` pairs. */
    sightings: [number, number][];
    pose_errors: number[];
    landmark_errors: number[];
    pose_calibration: Calibration;
    landmark_calibration: Calibration;
}

export interface SlidingWindow {
    rows: number;
    window: number;
    truth: Vec3[];
    start: Vec3[];
    batch: Vec3[];
    filtered: { mode: Vec3; covariance: number[] }[];
    landmark_truth: Vec3[];
    landmark_batch: Vec3[];
    filtering_errors: number[];
    filtering_calibration: Calibration;
    gaps: number[];
}

// ---------------------------------------------------------------------------
// Numbers

/** `Γ(d/2)` for a positive integer `d`. */
export function halfGamma(d: number): number {
    let value = d % 2 == 0 ? 1 : Math.sqrt(Math.PI);
    for (let k = d % 2 == 0 ? 1 : 0.5; k < d / 2; k += 1) value *= k;
    return value;
}

/** The chi-squared density with `d` degrees of freedom. */
export function chiSquaredDensity(d: number, x: number): number {
    if (x <= 0) return 0;
    return (x ** (d / 2 - 1) * Math.exp(-x / 2)) / (2 ** (d / 2) * halfGamma(d));
}

/** `χ²_d(0.9)` for the degrees of freedom the examples use. */
export const CHI2_90: Record<number, number> = {
    2: 4.605_170_185_988_091,
    3: 6.251_388_631_170_324,
    6: 10.644_640_675_668_418,
};

/** The ellipse of a covariance's `xy` block that holds probability 0.9: radii and angle. */
export function ellipse90(covariance: number[]): { rx: number; ry: number; angle: number } {
    const [a, b, c] = [covariance[0]!, covariance[1]!, covariance[4]!];
    const mean = (a + c) / 2;
    const spread = Math.hypot((a - c) / 2, b);
    const [l1, l2] = [mean + spread, Math.max(mean - spread, 0)];
    const angle = (Math.atan2(l1 - a, b) * 180) / Math.PI;
    const k = CHI2_90[2]!;
    return {
        rx: Math.sqrt(k * l1),
        ry: Math.sqrt(k * l2),
        angle: Number.isFinite(angle) && b != 0 ? angle : a >= c ? 0 : 90,
    };
}

/** Counts of `values` in `bins` equal bins over `[0, max)`; values past `max` are left out. */
export function histogram(values: number[], bins: number, max: number): number[] {
    const counts = new Array<number>(bins).fill(0);
    for (const v of values) {
        const i = Math.floor((v / max) * bins);
        if (i >= 0 && i < bins) counts[i]! += 1;
    }
    return counts;
}

export function distance(a: Vec3, b: Vec3): number {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/** The largest eigenvalue of a symmetric 3 × 3 matrix, by rows, by power iteration. */
export function largestEigenvalue(m: number[]): number {
    let v = [1, 1, 1];
    let lambda = 0;
    for (let i = 0; i < 100; i++) {
        const w = [0, 1, 2].map((r) => m[3 * r]! * v[0]! + m[3 * r + 1]! * v[1]! + m[3 * r + 2]! * v[2]!);
        const norm = Math.hypot(...w) || 1;
        lambda = norm;
        v = w.map((x) => x / norm);
    }
    return lambda;
}

// ---------------------------------------------------------------------------
// Drawing

const INK = {
    surface: '#fcfcfb',
    panel: '#f3f2ee',
    border: '#e4e3de',
    primary: '#0b0b0b',
    secondary: '#52514e',
    muted: '#7a7974',
    grid: '#e1e0da',
};

export const SERIES = {
    estimate: '#2a78d6',
    reckoning: '#eb6834',
    landmark: '#1baf7a',
};

const FONT = `font-family="ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"`;

const f = (x: number) => (Math.round(x * 100) / 100).toString();

function escape(text: string): string {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function text(
    x: number,
    y: number,
    content: string,
    { size = 14, fill = INK.secondary, weight = 400, anchor = 'start' } = {},
): string {
    return `<text x="${f(x)}" y="${f(y)}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}">${escape(content)}</text>`;
}

/** A card: the surface, a title and a subtitle, and the body. */
function card(width: number, height: number, title: string, subtitle: string, body: string[], label: string) {
    return [
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="${escape(label)}" ${FONT}>`,
        `<title>${escape(label)}</title>`,
        `<rect x="0.5" y="0.5" width="${width - 1}" height="${height - 1}" rx="18" fill="${INK.surface}" stroke="${INK.border}"/>`,
        text(36, 52, title, { size: 26, fill: INK.primary, weight: 650 }),
        text(36, 80, subtitle, { size: 15, fill: INK.muted }),
        ...body,
        `</svg>`,
    ].join('\n');
}

/** A map from world `(x, y)` to the screen, equal in both axes, `y` up. */
interface Projection {
    (p: Vec3): [number, number];
    scale: number;
}

function project(points: Vec3[], box: { x: number; y: number; w: number; h: number }, pad = 0.04): Projection {
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const scale = Math.min(box.w / ((x1 - x0) * (1 + 2 * pad)), box.h / ((y1 - y0) * (1 + 2 * pad)));
    const cx = box.x + box.w / 2;
    const cy = box.y + box.h / 2;
    const [mx, my] = [(x0 + x1) / 2, (y0 + y1) / 2];
    const map = ((p: Vec3) => [cx + (p[0] - mx) * scale, cy - (p[1] - my) * scale]) as Projection;
    map.scale = scale;
    return map;
}

function polyline(points: [number, number][], stroke: string, width: number, extra = ''): string {
    const d = points.map(([x, y], i) => `${i ? 'L' : 'M'}${f(x)} ${f(y)}`).join('');
    return `<path d="${d}" fill="none" stroke="${stroke}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round" ${extra}/>`;
}

function ellipse(at: [number, number], covariance: number[], scale: number, color: string, opacity = 0.22): string {
    const { rx, ry, angle } = ellipse90(covariance);
    return `<ellipse cx="${f(at[0])}" cy="${f(at[1])}" rx="${f(Math.max(rx * scale, 0.6))}" ry="${f(Math.max(ry * scale, 0.6))}" transform="rotate(${f(-angle)} ${f(at[0])} ${f(at[1])})" fill="${color}" fill-opacity="${opacity}" stroke="${color}" stroke-width="1" stroke-opacity="0.9"/>`;
}

function cross(at: [number, number], size: number, color: string): string {
    const [x, y] = at;
    return `<path d="M${f(x - size)} ${f(y - size)}L${f(x + size)} ${f(y + size)}M${f(x - size)} ${f(y + size)}L${f(x + size)} ${f(y - size)}" stroke="${color}" stroke-width="1.6" stroke-linecap="round"/>`;
}

/** A legend: one row per entry, a swatch drawn by `mark` and its label. */
function legend(x: number, y: number, entries: { label: string; mark: (x: number, y: number) => string }[], gap = 26): string[] {
    return entries.flatMap(({ label, mark }, i) => [
        mark(x, y + i * gap),
        text(x + 34, y + i * gap + 5, label, { size: 14, fill: INK.secondary }),
    ]);
}

const swatch = {
    line: (color: string, dash = '') => (x: number, y: number) =>
        polyline([[x, y], [x + 24, y]], color, 2.5, dash ? `stroke-dasharray="${dash}"` : ''),
    ellipse: (color: string) => (x: number, y: number) =>
        `<ellipse cx="${x + 12}" cy="${y}" rx="11" ry="6" fill="${color}" fill-opacity="0.22" stroke="${color}"/>` +
        `<circle cx="${x + 12}" cy="${y}" r="2.6" fill="${color}"/>`,
    cross: (color: string) => (x: number, y: number) => cross([x + 12, y], 4.5, color),
};

/** A tile: a large figure and what it is. */
function tile(x: number, y: number, w: number, value: string, label: string, note: string): string[] {
    return [
        `<rect x="${x}" y="${y}" width="${w}" height="92" rx="12" fill="${INK.panel}"/>`,
        text(x + 18, y + 42, value, { size: 27, fill: INK.primary, weight: 650 }),
        text(x + 18, y + 66, label, { size: 13, fill: INK.secondary }),
        text(x + 18, y + 83, note, { size: 12, fill: INK.muted }),
    ];
}

function scaleBar(x: number, y: number, metres: number, scale: number, label: string): string[] {
    const w = metres * scale;
    return [
        `<path d="M${f(x)} ${f(y - 5)}V${f(y)}H${f(x + w)}V${f(y - 5)}" fill="none" stroke="${INK.muted}" stroke-width="1.5"/>`,
        text(x + w / 2, y + 17, label, { size: 12, fill: INK.muted, anchor: 'middle' }),
    ];
}

const percent = (x: number) => `${(100 * x).toFixed(1)}%`;

// ---------------------------------------------------------------------------
// The figures

/** The pose graph's map: one dataset's truth, dead reckoning and estimate, and a zoom at true
 * scale on where the circuit closes. */
export function poseGraphFigure(data: PoseGraph): string {
    const W = 1200;
    const H = 760;
    const poses = data.poses;
    const all = [...poses.flatMap((p) => [p.truth, p.start, p.mode]), ...data.landmarks.flatMap((l) => [l.truth, l.mode])];
    const mapBox = { x: 36, y: 104, w: 620, h: 620 };
    const map = project(all, mapBox);
    const body: string[] = [
        `<rect x="${mapBox.x}" y="${mapBox.y}" width="${mapBox.w}" height="${mapBox.h}" rx="14" fill="${INK.panel}"/>`,
    ];
    // The sightings, faint, from each estimated pose to its estimated landmark.
    const sightings = data.sightings
        .map(([k, j]) => {
            const [a, b] = [map(poses[k]!.mode), map(data.landmarks[j]!.mode)];
            return `M${f(a[0])} ${f(a[1])}L${f(b[0])} ${f(b[1])}`;
        })
        .join('');
    body.push(`<path d="${sightings}" stroke="${INK.primary}" stroke-opacity="0.07" stroke-width="1"/>`);
    body.push(polyline(poses.map((p) => map(p.start)), SERIES.reckoning, 2.2, 'stroke-opacity="0.95"'));
    body.push(polyline(poses.map((p) => map(p.truth)), INK.secondary, 1.6, 'stroke-dasharray="5 5"'));
    body.push(polyline(poses.map((p) => map(p.mode)), SERIES.estimate, 2.4));
    for (const p of poses) {
        body.push(ellipse(map(p.mode), p.covariance, map.scale, SERIES.estimate, 0.25));
    }
    for (const l of data.landmarks) {
        body.push(cross(map(l.truth), 4.5, INK.secondary));
        body.push(ellipse(map(l.mode), l.covariance, map.scale, SERIES.landmark, 0.28));
        const [x, y] = map(l.mode);
        body.push(`<circle cx="${f(x)}" cy="${f(y)}" r="3.4" fill="${SERIES.landmark}" stroke="${INK.panel}" stroke-width="1.5"/>`);
    }
    // Where dead reckoning ends against where the estimate does.
    const end = poses[poses.length - 1]!;
    const [ex, ey] = map(end.start);
    body.push(`<circle cx="${f(ex)}" cy="${f(ey)}" r="4" fill="${SERIES.reckoning}" stroke="${INK.panel}" stroke-width="1.5"/>`);
    body.push(text(ex - 10, ey - 12, `dead reckoning ends ${distance(end.start, end.truth).toFixed(2)} m off`, { size: 13, fill: INK.primary, anchor: 'end' }));
    body.push(...scaleBar(mapBox.x + 24, mapBox.y + mapBox.h - 24, 5, map.scale, '5 m'));

    // The zoom: the last poses, where the circuit closes, at true scale.
    const last = poses.slice(-9);
    const zoomPoints = [...last.flatMap((p) => [p.truth, p.mode])];
    const zoomBox = { x: 690, y: 132, w: 474, h: 280 };
    const zoom = project(zoomPoints, zoomBox, 0.18);
    // Its footprint on the map: the zoom's corners, back in the world, then onto the map.
    const toWorld = (sx: number, sy: number): Vec3 => {
        const z0 = zoom([0, 0, 0]);
        return [(sx - z0[0]) / zoom.scale, -(sy - z0[1]) / zoom.scale, 0];
    };
    const w0 = map(toWorld(zoomBox.x, zoomBox.y));
    const w1 = map(toWorld(zoomBox.x + zoomBox.w, zoomBox.y + zoomBox.h));
    body.push(
        `<rect x="${f(Math.min(w0[0], w1[0]))}" y="${f(Math.min(w0[1], w1[1]))}" width="${f(Math.abs(w1[0] - w0[0]))}" height="${f(Math.abs(w1[1] - w0[1]))}" fill="none" stroke="${INK.primary}" stroke-opacity="0.75" stroke-width="1.2" rx="3"/>`,
    );
    body.push(`<rect x="${zoomBox.x}" y="${zoomBox.y}" width="${zoomBox.w}" height="${zoomBox.h}" rx="14" fill="${INK.panel}"/>`);
    body.push(`<clipPath id="zoom"><rect x="${zoomBox.x}" y="${zoomBox.y}" width="${zoomBox.w}" height="${zoomBox.h}" rx="14"/></clipPath>`);
    const clipped: string[] = [];
    clipped.push(polyline(poses.map((p) => zoom(p.truth)), INK.secondary, 1.6, 'stroke-dasharray="5 5"'));
    clipped.push(polyline(poses.map((p) => zoom(p.mode)), SERIES.estimate, 2.2));
    for (const p of poses) {
        clipped.push(ellipse(zoom(p.mode), p.covariance, zoom.scale, SERIES.estimate, 0.25));
        const [x, y] = zoom(p.mode);
        clipped.push(`<circle cx="${f(x)}" cy="${f(y)}" r="2.6" fill="${SERIES.estimate}"/>`);
        clipped.push(cross(zoom(p.truth), 3.5, INK.primary));
    }
    body.push(`<g clip-path="url(#zoom)">${clipped.join('')}</g>`);
    body.push(text(zoomBox.x + 4, zoomBox.y - 12, 'Where the circuit closes, at true scale', { size: 15, fill: INK.primary, weight: 600 }));
    body.push(...scaleBar(zoomBox.x + 20, zoomBox.y + zoomBox.h - 22, 0.25, zoom.scale, '25 cm'));

    // The legend and the figures.
    body.push(
        ...legend(704, 440, [
            { label: 'Truth (drawn from the model)', mark: swatch.line(INK.secondary, '5 5') },
            { label: 'Dead reckoning on the odometry: the start', mark: swatch.line(SERIES.reckoning) },
            { label: 'Estimated poses, with 90% ellipses', mark: swatch.ellipse(SERIES.estimate) },
            { label: 'Estimated landmarks, with 90% ellipses', mark: swatch.ellipse(SERIES.landmark) },
            { label: 'True landmarks', mark: swatch.cross(INK.secondary) },
        ]),
    );
    const startError = Math.max(...poses.map((p) => distance(p.start, p.truth)));
    const modeError = Math.max(...poses.map((p) => distance(p.mode, p.truth)));
    body.push(...tile(690, 590, 232, `${startError.toFixed(2)} m → ${(100 * modeError).toFixed(0)} cm`, 'largest position error', 'dead reckoning → estimate'));
    body.push(
        ...tile(
            932,
            590,
            232,
            percent(data.pose_calibration.coverage),
            'of true poses in their 90% ellipsoid',
            `over ${data.rows} datasets, ±${percent(data.pose_calibration.tolerance)} allowed`,
        ),
    );
    return card(
        W,
        H,
        'A pose graph, estimated',
        `One of ${data.rows} datasets: ${poses.length} poses on SE(3) around a closed circuit, ${data.landmarks.length} landmarks, ${data.sightings.length} sightings`,
        body,
        'A pose graph on a closed circuit: the truth, the drifting dead-reckoned start, and the Levenberg–Marquardt estimate with its 90% ellipses',
    );
}

/** One panel of the calibration figure: a histogram of normalized errors against the chi-squared
 * density they follow when the marginals are calibrated. */
function calibrationPanel(
    box: { x: number; y: number; w: number; h: number },
    errors: number[],
    d: number,
    calibration: Calibration,
    what: string,
): string[] {
    const max = d == 6 ? 28 : 20;
    const bins = 40;
    const counts = histogram(errors, bins, max);
    const width = max / bins;
    const densities = counts.map((c) => c / (errors.length * width));
    const top = Math.max(...densities, ...Array.from({ length: 200 }, (_, i) => chiSquaredDensity(d, ((i + 0.5) * max) / 200))) * 1.12;
    const plot = { x: box.x + 30, y: box.y + 96, w: box.w - 60, h: box.h - 160 };
    const sx = (x: number) => plot.x + (x / max) * plot.w;
    const sy = (y: number) => plot.y + plot.h - (y / top) * plot.h;
    const out: string[] = [
        `<rect x="${box.x}" y="${box.y}" width="${box.w}" height="${box.h}" rx="14" fill="${INK.panel}"/>`,
        text(box.x + 22, box.y + 34, what, { size: 17, fill: INK.primary, weight: 600 }),
    ];
    // The grid and the axis.
    for (let t = 0; t <= max; t += d == 6 ? 4 : 4) {
        out.push(`<path d="M${f(sx(t))} ${f(plot.y)}V${f(plot.y + plot.h)}" stroke="${INK.grid}" stroke-width="1"/>`);
        out.push(text(sx(t), plot.y + plot.h + 20, `${t}`, { size: 12, fill: INK.muted, anchor: 'middle' }));
    }
    out.push(text(plot.x + plot.w / 2, plot.y + plot.h + 42, 'normalized error |W (q* ⊖ q̂)|²', { size: 13, fill: INK.muted, anchor: 'middle' }));
    // The bars, with a gap of the panel's colour between them, rounded at their tops.
    densities.forEach((v, i) => {
        if (v <= 0) return;
        const x = sx(i * width) + 1;
        const w = sx((i + 1) * width) - sx(i * width) - 2;
        const y = sy(v);
        const h = plot.y + plot.h - y;
        const r = Math.min(3, w / 2, h);
        out.push(
            `<path d="M${f(x)} ${f(plot.y + plot.h)}V${f(y + r)}Q${f(x)} ${f(y)} ${f(x + r)} ${f(y)}H${f(x + w - r)}Q${f(x + w)} ${f(y)} ${f(x + w)} ${f(y + r)}V${f(plot.y + plot.h)}Z" fill="${SERIES.estimate}" fill-opacity="${(i + 0.5) * width <= CHI2_90[d]! ? 0.9 : 0.45}"/>`,
        );
    });
    // The density.
    const curve: [number, number][] = Array.from({ length: 241 }, (_, i) => {
        const x = (i / 240) * max;
        return [sx(x), sy(chiSquaredDensity(d, x))];
    });
    out.push(polyline(curve, INK.primary, 2.2));
    // The 90% quantile.
    const q = sx(CHI2_90[d]!);
    out.push(`<path d="M${f(q)} ${f(plot.y - 6)}V${f(plot.y + plot.h)}" stroke="${INK.primary}" stroke-width="1.4" stroke-dasharray="4 4"/>`);
    out.push(text(q + 8, plot.y + 8, `χ²${d == 6 ? '₆' : '₃'}(0.9) = ${CHI2_90[d]!.toFixed(2)}`, { size: 13, fill: INK.primary }));
    out.push(text(q + 8, plot.y + 26, 'the 90% ellipsoid’s edge', { size: 12, fill: INK.muted }));
    // The verdict.
    out.push(
        text(box.x + 22, box.y + 60, `${percent(calibration.coverage)} inside their 90% ellipsoid: 90.0% ± ${percent(calibration.tolerance)} expected`, {
            size: 14,
            fill: INK.secondary,
        }),
    );
    out.push(text(box.x + box.w - 22, box.y + 34, `${errors.length.toLocaleString('en-US')} errors`, { size: 13, fill: INK.muted, anchor: 'end' }));
    return out;
}

/** The calibration of the pose graph's marginals: its normalized errors against chi-squared. */
export function calibrationFigure(data: PoseGraph): string {
    const W = 1200;
    const H = 580;
    const body: string[] = [
        ...calibrationPanel({ x: 36, y: 104, w: 556, h: 400 }, data.pose_errors, 6, data.pose_calibration, 'Poses (6 dimensions)'),
        ...calibrationPanel({ x: 608, y: 104, w: 556, h: 400 }, data.landmark_errors, 3, data.landmark_calibration, 'Landmarks (3 dimensions)'),
        ...legend(
            48,
            534,
            [
                { label: 'How often each normalized error occurred', mark: (x, y) => `<rect x="${x + 4}" y="${y - 7}" width="16" height="14" rx="3" fill="${SERIES.estimate}"/>` },
            ],
        ),
        ...legend(470, 534, [{ label: 'The chi-squared density it follows when the marginals are calibrated', mark: swatch.line(INK.primary) }]),
    ];
    return card(
        W,
        H,
        'Are the uncertainties honest?',
        `The truth's distance from every estimate, in its own marginal's units, over ${data.rows} datasets`,
        body,
        'Histograms of the normalized errors of the poses and the landmarks against the chi-squared densities they follow when the Laplace marginals are calibrated',
    );
}

/** The sliding window: one dataset's path, and each tick's position error. */
export function slidingWindowFigure(data: SlidingWindow): string {
    const W = 1200;
    const H = 800;
    const all = [...data.truth, ...data.start, ...data.batch, ...data.filtered.map((x) => x.mode), ...data.landmark_truth];
    const mapBox = { x: 36, y: 104, w: 1128, h: 250 };
    const map = project(all, mapBox, 0.02);
    const body: string[] = [`<rect x="${mapBox.x}" y="${mapBox.y}" width="${mapBox.w}" height="${mapBox.h}" rx="14" fill="${INK.panel}"/>`];
    for (const l of data.landmark_truth) body.push(cross(map(l), 4, INK.muted));
    // The window at the last tick.
    const lastWindow = data.filtered.slice(-data.window).map((x) => map(x.mode));
    const xs = lastWindow.map((p) => p[0]);
    const ys = lastWindow.map((p) => p[1]);
    body.push(
        `<rect x="${f(Math.min(...xs) - 10)}" y="${f(Math.min(...ys) - 14)}" width="${f(Math.max(...xs) - Math.min(...xs) + 20)}" height="${f(Math.max(...ys) - Math.min(...ys) + 28)}" rx="8" fill="${SERIES.estimate}" fill-opacity="0.12" stroke="${SERIES.estimate}" stroke-opacity="0.6"/>`,
    );
    body.push(text(Math.max(...xs) + 6, Math.min(...ys) - 22, `the window: last ${data.window} poses`, { size: 13, fill: INK.primary, anchor: 'end' }));
    body.push(polyline(data.start.map(map), SERIES.reckoning, 2.2));
    body.push(polyline(data.truth.map(map), INK.secondary, 1.6, 'stroke-dasharray="5 5"'));
    body.push(polyline(data.batch.map(map), SERIES.landmark, 2.2));
    body.push(polyline(data.filtered.map((x) => map(x.mode)), SERIES.estimate, 2.2));
    for (const x of data.filtered) {
        const [cx, cy] = map(x.mode);
        body.push(`<circle cx="${f(cx)}" cy="${f(cy)}" r="2.4" fill="${SERIES.estimate}"/>`);
    }
    body.push(...scaleBar(mapBox.x + 24, mapBox.y + mapBox.h - 22, 10, map.scale, '10 m'));
    // Direct labels at the end of the path.
    const label = (p: Vec3, content: string, color: string, dy: number) => {
        const [x, y] = map(p);
        return [`<circle cx="${f(x)}" cy="${f(y)}" r="3.5" fill="${color}"/>`, text(x + 10, y + dy, content, { size: 13, fill: INK.secondary })];
    };
    body.push(...label(data.start[data.start.length - 1]!, 'dead reckoning', SERIES.reckoning, 4));

    // Each tick's position error, on a logarithmic axis, with the filter's 90% radius.
    const plot = { x: 110, y: 430, w: 920, h: 240 };
    body.push(`<rect x="36" y="380" width="1128" height="340" rx="14" fill="${INK.panel}"/>`);
    body.push(text(58, 408, 'Position error at each tick', { size: 16, fill: INK.primary, weight: 600 }));
    const ticks = data.truth.length;
    const [lo, hi] = [0.003, 5];
    const sx = (k: number) => plot.x + (k / (ticks - 1)) * plot.w;
    const sy = (e: number) => plot.y + plot.h - ((Math.log10(Math.max(e, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * plot.h;
    for (const e of [0.01, 0.1, 1]) {
        body.push(`<path d="M${plot.x} ${f(sy(e))}H${plot.x + plot.w}" stroke="${INK.grid}" stroke-width="1"/>`);
        body.push(text(plot.x - 10, sy(e) + 4, e >= 1 ? `${e} m` : `${e * 100} cm`, { size: 12, fill: INK.muted, anchor: 'end' }));
    }
    for (let k = 0; k < ticks; k += 10) {
        body.push(text(sx(k), plot.y + plot.h + 20, `${k}`, { size: 12, fill: INK.muted, anchor: 'middle' }));
    }
    body.push(text(plot.x + plot.w / 2, plot.y + plot.h + 40, 'tick', { size: 13, fill: INK.muted, anchor: 'middle' }));
    // The filter's 90% radius along its longest axis, as a band from the floor.
    const radius = data.filtered.map((x) => Math.sqrt(CHI2_90[3]! * largestEigenvalue(x.covariance)));
    const band = [
        ...radius.map((r, k) => [sx(k), sy(r)] as [number, number]),
        [sx(ticks - 1), sy(lo)] as [number, number],
        [sx(0), sy(lo)] as [number, number],
    ];
    body.push(`<path d="${band.map(([x, y], i) => `${i ? 'L' : 'M'}${f(x)} ${f(y)}`).join('')}Z" fill="${SERIES.estimate}" fill-opacity="0.16"/>`);
    const series = (path: Vec3[], color: string, width: number) =>
        polyline(
            path.map((p, k) => [sx(k), sy(distance(p, data.truth[k]!))]),
            color,
            width,
        );
    body.push(series(data.start, SERIES.reckoning, 2.2));
    body.push(series(data.batch, SERIES.landmark, 2));
    body.push(series(data.filtered.map((x) => x.mode), SERIES.estimate, 2.4));
    // Direct labels at the right end.
    const endLabel = (path: Vec3[], content: string, dy = 4) => {
        const e = distance(path[ticks - 1]!, data.truth[ticks - 1]!);
        return text(sx(ticks - 1) + 8, sy(e) + dy, content, { size: 12, fill: INK.secondary });
    };
    body.push(endLabel(data.start, 'dead reckoning'));
    body.push(endLabel(data.filtered.map((x) => x.mode), 'window', 12));
    body.push(endLabel(data.batch, 'batch', -6));

    body.push(
        ...legend(58, 744, [{ label: 'Truth', mark: swatch.line(INK.secondary, '5 5') }]),
        ...legend(170, 744, [{ label: 'Dead reckoning', mark: swatch.line(SERIES.reckoning) }]),
        ...legend(350, 744, [{ label: 'Window, filtering: each pose as it arrives', mark: swatch.line(SERIES.estimate) }]),
        ...legend(700, 744, [{ label: 'Batch, every measurement', mark: swatch.line(SERIES.landmark) }]),
        ...legend(950, 744, [
            {
                label: 'Window’s 90% radius',
                mark: (x, y) => `<rect x="${x}" y="${y - 7}" width="24" height="14" rx="3" fill="${SERIES.estimate}" fill-opacity="0.3"/>`,
            },
        ]),
    );
    const meanGap = data.gaps.reduce((a, b) => a + b, 0) / data.gaps.length;
    body.push(
        text(
            1164,
            80,
            `filtering coverage ${percent(data.filtering_calibration.coverage)} against 90% · gap to the batch at the end ${meanGap.toFixed(2)} σ`,
            { size: 14, fill: INK.secondary, anchor: 'end' },
        ),
    );
    return card(
        W,
        H,
        'A sliding window, retiring as it goes',
        `One of ${data.rows} datasets: ${ticks} poses along an open path`,
        body,
        'A sliding-window filter over an open path: its estimates against the truth, dead reckoning and the batch, and each tick’s position error',
    );
}
