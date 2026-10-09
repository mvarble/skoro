import { withComponent } from '$lib/load';
import type { PageLoad } from './$types';

export const load: PageLoad = ({ data }) => withComponent(data);
