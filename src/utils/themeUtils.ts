
export type ThemeMode = 'auto' | 'system' | 'light' | 'dark';
export const THEME_KEY = 'themeMode';
export const THEME_CYCLE: ThemeMode[] = ['auto', 'system', 'light', 'dark'];

export function nextThemeMode(current: ThemeMode | null): ThemeMode {
  const cur = current ?? 'auto';
  const idx = THEME_CYCLE.indexOf(cur);
  return THEME_CYCLE[(idx + 1) % THEME_CYCLE.length];
}

export function loadStoredTheme(): ThemeMode | null {
  try {
    const v = localStorage.getItem(THEME_KEY);
    if (!v) return null;
    if (THEME_CYCLE.includes(v as ThemeMode)) return v as ThemeMode;
    return null;
  } catch {
    return null;
  }
}

export function storeTheme(mode: ThemeMode | null) {
  try {
    if (mode == null) localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, mode);
  } catch {}
}

/**
 * Возвращает true, если по заданному mode нужно применить "dark" внешний вид.
 * prefersDark — результат matchMedia('(prefers-color-scheme: dark)').matches
 */
export function effectiveIsDark(mode: ThemeMode, prefersDark: boolean): boolean {
  switch (mode) {
    case 'dark':
      return true;
    case 'light':
      return false;
    case 'system':
    case 'auto':
      return prefersDark;
    default:
      return prefersDark;
  }
}

export function applyHtmlDarkClass(shouldBeDark: boolean) {
  try {
    const el = document.documentElement;
    if (shouldBeDark) el.classList.add('dark');
    else el.classList.remove('dark');
  } catch {}
}
