# hugo-toha.github.io

[![Netlify Status](https://api.netlify.com/api/v1/badges/b1b93b02-f278-440b-ae1b-304e9f4c4ab5/deploy-status)](https://app.netlify.com/sites/toha/deploys) [![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fhugo-toha%2Fhugo-toha.github.io%2Fbadge%3Fref%3Dmain&style=flat)](https://actions-badge.atrox.dev/hugo-toha/hugo-toha.github.io/goto?ref=main) ![Repository Size](https://img.shields.io/github/repo-size/hugo-toha/hugo-toha.github.io) ![Contributor](https://img.shields.io/github/contributors/hugo-toha/hugo-toha.github.io) ![Last Commit](https://img.shields.io/github/last-commit/hugo-toha/hugo-toha.github.io) ![License](https://img.shields.io/github/license/hugo-toha/hugo-toha.github.io) ![Open Issues](https://img.shields.io/github/issues/hugo-toha/hugo-toha.github.io?color=important) ![Open Pull Requests](https://img.shields.io/github/issues-pr/hugo-toha/hugo-toha.github.io?color=yellowgreen) ![Security Headers](https://img.shields.io/security-headers?url=https%3A%2F%2Fhugo-toha.github.io%2F) [![This project is using Percy.io for visual regression testing.](https://percy.io/static/images/percy-badge.svg)](https://percy.io/b7cb60ab/hugo-toha.github.io)

An example hugo static site with Toha theme.

Attributions:
- <a href='https://www.freepik.com/vectors/business'>Business vector created by studiogstock - www.freepik.com</a>


## Legacy portfolio notice

This site remains available as an archive. Every HTML page displays a notice
linking to the current portfolio at `https://noegracia.eu/`. Publish these
changes only after the new portfolio is live. There is no automatic redirect.

The shared banner is in `layouts/partials/legacy-banner.html`; its responsive
styles are in `assets/styles/override.scss`. It appears below the navigation
and scrolls with the page, with no popup, dismissal state or JavaScript.
The home and base layouts are copied from Toha v4.9.0 with only the banner
partial added at the start of the body. Compare these overrides with upstream
when upgrading the theme. The old site's `baseURL` stays unchanged.
