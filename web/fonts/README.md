# Fonts

Self-hosted so the chat UI and the widget make no third-party requests. Variable fonts (one file per subset covers
every weight), Hebrew and Latin subsets, from the Fontsource builds `@fontsource-variable/heebo@5.3.0` and
`@fontsource-variable/frank-ruhl-libre@5.3.0`.

| File | Font | Used for |
|---|---|---|
| `heebo-hebrew-wght.woff2`, `heebo-latin-wght.woff2` | [Heebo](https://github.com/OdedEzer/heebo), weights 100–900 | all text |
| `frank-ruhl-libre-hebrew-wght.woff2`, `frank-ruhl-libre-latin-wght.woff2` | [Frank Ruhl Libre](https://github.com/fontef/frankruhllibre), weights 300–900 | headings |

Both are licensed under the SIL Open Font License 1.1: [`OFL-heebo.txt`](OFL-heebo.txt),
[`OFL-frank-ruhl-libre.txt`](OFL-frank-ruhl-libre.txt). The `@font-face` rules are at the top of `../widget.css`.
