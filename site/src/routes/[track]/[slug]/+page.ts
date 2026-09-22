import { loadComponent } from '$lib/content';
import type { PageLoad } from './$types';

// The server load finds the page; this one attaches its compiled component, which cannot travel as
// serialized data.
export const load: PageLoad = async ({ data }) => ({
    ...data,
    component: await loadComponent(data.page),
});
