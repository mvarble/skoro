import { MediaQuery } from 'svelte/reactivity';
import { Persisted } from './persisted.svelte';

// Copied from the blog. The key must match the inline script in `app.html`, which applies the
// theme before first paint.
export const THEME_KEY = 'skoro:theme';

class Theme {
    #preference = new Persisted<'system' | 'light' | 'dark'>(THEME_KEY, 'system');
    #query = new MediaQuery('prefers-color-scheme: dark');
    #system = $derived<'dark' | 'light'>(this.#query.current ? 'dark' : 'light');

    get current() {
        return this.#preference.current === 'system' ? this.#system : this.#preference.current;
    }

    set current(value: 'light' | 'dark') {
        this.#preference.current = value === this.#system ? 'system' : value;
    }
}

export const theme = new Theme();
