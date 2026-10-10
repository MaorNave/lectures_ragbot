# Fonts

Self-hosted so the chat UI and the widget make no third-party requests. Variable fonts (one file per subset covers
every weight), Hebrew and Latin subsets, from the Fontsource builds `@fontsource-variable/heebo@5.3.0` and
`@fontsource-variable/rubik@5.2.5`.

| File | Font | Used for |
|---|---|---|
| `heebo-hebrew-wght.woff2`, `heebo-latin-wght.woff2` | [Heebo](https://github.com/OdedEzer/heebo), weights 100–900 | body text |
| `rubik-hebrew-wght.woff2`, `rubik-latin-wght.woff2` | [Rubik](https://github.com/googlefonts/rubik), weights 300–900 | headings, lecture numbers, source names |

Both are licensed under the SIL Open Font License 1.1: [`OFL-heebo.txt`](OFL-heebo.txt),
[`OFL-rubik.txt`](OFL-rubik.txt). The `@font-face` rules are at the top of `../widget.css`.
