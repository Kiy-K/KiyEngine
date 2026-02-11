---
description: How to update and deploy the KiyEngine download website
---

# KiyEngine Website Workflow

The website is a **pure static HTML/CSS/JS** site in the `website/` directory. No build step required.

## Structure

- `website/index.html` — Single-page site with downloads, features, setup guide
- `.github/workflows/pages.yml` — GitHub Pages deployment (auto on push to `website/**`)

## Local Preview

// turbo
1. Open `website/index.html` in a browser:
```bash
xdg-open website/index.html   # Linux
open website/index.html        # macOS
```

## Edit Content

2. Edit `website/index.html` directly. All CSS is inline in `<style>`, JS is inline in `<script>`.

## Deploy

3. Commit and push to `main`. GitHub Actions deploys automatically when files in `website/` change.
```bash
git add website/
git commit -m "website: update content"
git push origin main
```

4. The site will be live at: **https://kiy-k.github.io/KiyEngine/**

## First-time Setup

If GitHub Pages is not yet enabled:
1. Go to repo Settings → Pages
2. Set Source to **GitHub Actions**
3. Push any change to `website/` to trigger the first deployment

## Download Links

The website auto-fetches the latest release from GitHub API to populate download links and metadata (version, size, date, download count). If the API call fails, links fall back to `/releases/latest`.

## Adding New Downloads

To add a new download card, copy an existing `.dl-card` block in the HTML and update:
- The icon, title, description
- The `id` attributes for JS auto-population
- Add matching logic in the `<script>` section
