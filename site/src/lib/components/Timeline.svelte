<script lang="ts">
    import type { WorkflowEvent } from '$lib/documents';

    // An entry's stage record, as `bin/wf` wrote it: when each stage was approved, and when and
    // why the entry went back to an earlier one.
    let { events }: { events: WorkflowEvent[] } = $props();

    const VERBS: Record<string, string> = {
        start: 'Started at',
        approve: 'Approved',
        reopen: 'Reopened',
        abandon: 'Set aside at',
        resume: 'Taken up again at',
    };
</script>

<ol class="timeline">
    {#each events as event, i (i)}
        <li class="event-{event.action}">
            <time datetime={event.date}>{event.date}</time>
            <span class="what">{VERBS[event.action] ?? event.action} <b>{event.stage}</b></span>
            {#if event.commit}<code class="commit">{event.commit}</code>{/if}
            {#if event.reason}<span class="reason">{event.reason}</span>{/if}
        </li>
    {/each}
</ol>
