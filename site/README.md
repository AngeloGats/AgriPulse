# AgroPulse site

The website is organized by page type:

- `pages/` contains the homepage, About page, Algorithm Library, and Client
  Portal.
- `assets/css/`, `assets/js/`, and `assets/images/` contain shared site styles,
  scripts, and images.
- `ML Algorithms/` contains the interactive Linear Regression field lab,
  including its local Flask app and page-specific assets.

Start the local Flask app from the project root:

```powershell
.\.venv\Scripts\python.exe "site\ML Algorithms\app.py"
```

Or in VS Code, choose **Terminal > Run Task > Start AgroPulse model server**.
Keep the task running while using the model links from Live Server on port 5500.

The Flask app serves the main homepage at `/` (also available at `/index.html`),
along with `/about.html`,
`/analytics.html`, and `/demo.html`; open the model demonstration at
`/linear-regression`.

When using VS Code Live Server from the project root, the root `index.html`
forwards to `site/pages/index.html` so the homepage opens instead of a directory
listing.
