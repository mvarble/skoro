import { PAGES } from '$lib/server/content';
import type { LayoutServerLoad } from './$types';

export const load: LayoutServerLoad = () => ({ pages: PAGES });
