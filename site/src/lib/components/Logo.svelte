<script lang="ts">
    // Inlined rather than linked, so the logo's ink and paper can follow the site's theme toggle
    // (an `<img>` never sees the `.dark` class). The rules live under `.logo` in `layout.css`.
    import icon from '../../../static/favicon.svg?raw';
    import lockup from '../../../logo/skoro-lockup.svg?raw';

    let { variant }: { variant: 'icon' | 'lockup' } = $props();
</script>

<span class="logo logo-{variant}" aria-hidden="true">
    <!-- eslint-disable-next-line svelte/no-at-html-tags -- a build-time file from logo/generate.py -->
    {@html variant === 'icon' ? icon : lockup}
</span>
