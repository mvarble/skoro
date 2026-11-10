import { defineGraph } from '$lib/graph/spec';

// The stages of an entry and the commands that move it between them. The three kinds share their
// last stages: a chore joins a feature's route at the plan, and every kind ends in publishing. Each
// stage is labelled with the command that does its work.

export const workflow = defineGraph({
    title: 'The workflow',
    caption:
        '/wf-start opens an entry at the first stage of its kind. Each stage is worked on with the ' +
        'command written beside it, and /wf-approve moves the entry on to the next; /wf-reopen ' +
        'takes it back to any earlier stage, of which only the step back to the stage before is ' +
        'drawn. Approving the publishing finishes the entry, and its branch can merge.',
    width: 1080,
    height: 520,
    nodeRadius: 58,
    fontSize: 15,
    nodeKinds: {
        endpoint: { name: 'Opening and finishing an entry', hue: 'gray' },
        document: {
            name: 'A stage that writes a document, frozen once approved',
            hue: 'blue',
            shape: 'square',
        },
        code: { name: 'The stage that writes the code', hue: 'green', shape: 'square' },
        pages: { name: 'The stage that writes the pages', hue: 'orange', shape: 'square' },
    },
    edgeKinds: {
        start: { name: '/wf-start <slug> <kind>', hue: 'gray' },
        approve: { name: '/wf-approve' },
        reopen: { name: '/wf-reopen <stage>', hue: 'red', line: 'dashed' },
    },
    nodes: [
        {
            id: 'start',
            x: 50,
            y: 260,
            r: 36,
            kind: 'endpoint',
            label: 'Start',
            annotation: '/wf-start',
        },
        {
            id: 'math',
            x: 210,
            y: 420,
            kind: 'document',
            label: 'Mathematics',
            annotation: '/wf-math',
            annotationAngle: 90,
        },
        {
            id: 'implementation',
            x: 400,
            y: 420,
            kind: 'document',
            label: 'Implementation',
            annotation: '/wf-implementation',
            annotationAngle: 90,
        },
        { id: 'plan', x: 580, y: 260, kind: 'document', label: 'Plan', annotation: '/wf-plan' },
        { id: 'build', x: 770, y: 260, kind: 'code', label: 'Build', annotation: '/wf-build' },
        {
            id: 'publish',
            x: 960,
            y: 260,
            kind: 'pages',
            label: 'Publish',
            annotation: '/wf-publish',
            annotationAngle: 0,
        },
        {
            id: 'writeup',
            x: 580,
            y: 80,
            kind: 'document',
            label: 'Writeup',
            annotation: '/wf-writeup',
            annotationAngle: 180,
        },
        {
            id: 'done',
            x: 960,
            y: 420,
            r: 36,
            kind: 'endpoint',
            label: 'Done',
            annotation: 'ready to merge',
            annotationAngle: 90,
        },
    ],
    edges: [
        { from: 'start', to: 'math', kind: 'start', label: 'feature' },
        { from: 'start', to: 'plan', kind: 'start', label: 'chore' },
        { from: 'start', to: 'writeup', kind: 'start', label: 'decision' },
        { from: 'math', to: 'implementation', kind: 'approve' },
        { from: 'implementation', to: 'plan', kind: 'approve' },
        { from: 'plan', to: 'build', kind: 'approve' },
        { from: 'build', to: 'publish', kind: 'approve' },
        { from: 'writeup', to: 'publish', kind: 'approve', via: [[960, 80]], toAngle: -90 },
        { from: 'publish', to: 'done', kind: 'approve' },
        { from: 'implementation', to: 'math', kind: 'reopen', bend: 40 },
        { from: 'plan', to: 'implementation', kind: 'reopen', bend: -35 },
        { from: 'build', to: 'plan', kind: 'reopen', bend: -40 },
        { from: 'publish', to: 'build', kind: 'reopen', bend: -40 },
        { from: 'publish', to: 'writeup', kind: 'reopen', bend: 25 },
    ],
});
