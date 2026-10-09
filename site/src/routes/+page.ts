import loaders from '$cms/loaders';
import type { PageLoad } from './$types';

// The landing prose is a document like any other, attached as a component.
export const load: PageLoad = async ({ data }) => ({
    ...data,
    component: data.landing ? (await loaders[data.landing]!()).default : undefined,
});
