# AgroPulse Intelligence – Setup Instructions

# READ THIS OR I WILL KISS YOU - GATAN

## 📖 First: how to read this file nicely (Preview)

If you opened this file in **VS Code** and it looks like plain text with `#`,
`**` and `|` symbols everywhere, that is normal – it is a *Markdown* file, and
VS Code can show it as a nicely formatted page. This is called **Preview**.

**Open this file in VS Code first** (if needed: **File → Open Folder…** →
choose the `MachineLearning` folder → click `INSTRUCTIONS.md` in the file list
on the left). Then use any **one** of these methods:

### Method 1 – Keyboard shortcut (fastest)

| Computer | Press | What it does |
|---|---|---|
| Windows / Linux | `Ctrl` + `Shift` + `V` | Replaces this view with the formatted Preview |
| Windows / Linux | `Ctrl` + `K`, release, then `V` | Opens the Preview **side by side** with the text |
| macOS | `Cmd` + `Shift` + `V` | Replaces this view with the formatted Preview |
| macOS | `Cmd` + `K`, release, then `V` | Opens the Preview **side by side** with the text |

### Method 2 – The Preview button
1. Look at the **top-right corner** of the editor tab for this file.
2. Click the small icon that looks like a **page with a magnifying glass**
   (hover over it and it says **“Open Preview”** or **“Open Preview to the
   Side”**).

### Method 3 – Command Palette (works if the others don’t)
1. Press `Ctrl` + `Shift` + `P` (macOS: `Cmd` + `Shift` + `P`).
2. Type **`Markdown: Open Preview`** and press `Enter`.

### Tips
- **Copying commands:** in the Preview, select the text of a command with your
  mouse and press `Ctrl` + `C` (macOS: `Cmd` + `C`). Paste it into the
  terminal with `Ctrl` + `V` (macOS: `Cmd` + `V`).
- **Back to the plain text:** close the Preview tab, or click the original
  `INSTRUCTIONS.md` tab.
- **Nothing happens?** Make sure the `INSTRUCTIONS.md` tab is the one selected
  (click on it first), then try Method 3.
- **Not using VS Code?** Any Markdown viewer can open this file, but VS Code’s
  Preview is the easiest option.

---

## What this guide covers

This guide gets the AgroPulse website and its trained Linear Regression demo
running on a new computer. Follow the steps in order; the whole setup takes
about 10–15 minutes.

**What you are running:** a Flask (Python) web server on your own computer. It
serves the website pages and runs the trained Linear Regression model. Nothing
is deployed online, and no internet connection is needed after installation
(only the Google Fonts on the pages need internet; without it, the page falls
back to default fonts).

---

## 1. Install the required software

| Software | Required? | Why |
|---|---|---|
| **Python 3.12** | **Yes** | Runs the Flask server and the trained model |
| **VS Code** | Recommended | Editor used by the team; has the Python extension |
| **Live Server** (VS Code extension) | Optional | Only if you want to browse the static pages on port 5500 |
| **Git** | Optional | Only if the project is shared through a Git repository |

You do **not** need Node.js, a database, or any other language.

> Python 3.12 is what the project was built and tested with. Python 3.11 or
> newer should also work, but anything older will fail to install the packages.

### 1.1 Install Python

#### Windows
1. Download Python 3.12 from <https://www.python.org/downloads/windows/>.
2. Run the installer and **tick “Add python.exe to PATH”** on the first screen,
   then choose **Install Now**.
3. Close and reopen your terminal, then check it worked:
   ```powershell
   python --version
   ```
   It should print `Python 3.12.x`.

   Alternative (Windows 10/11 with winget):
   ```powershell
   winget install --id Python.Python.3.12 -e
   ```

#### macOS
1. Download the 3.12 installer from <https://www.python.org/downloads/macos/>,
   **or** install with Homebrew: `brew install python@3.12`
2. Check it worked (on macOS the command is `python3`):
   ```bash
   python3 --version
   ```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
python3 --version
```

### 1.2 Install VS Code (recommended)
1. Download from <https://code.visualstudio.com/> and install it.
2. Open VS Code → **Extensions** (`Ctrl+Shift+X`) → install **Python**
   (by Microsoft).
3. *Optional:* install **Live Server** (by Ritwick Dey) – see
   [section 5](#5-optional-browse-the-pages-with-live-server-port-5500).

### 1.3 Install Git (only if you will use a Git repository)
Download from <https://git-scm.com/downloads> and install with the default
options. Check with `git --version`.

---

## 2. Get the project

You need the **`MachineLearning`** project folder. Get it from whichever way
your team shares it:

- **ZIP / shared drive:** copy the folder to your computer and extract it. Do
  **not** copy the `.venv` folder from someone else’s computer – every person
  creates their own in step 3.
- **Git repository:** clone it (replace the URL with your team’s repository):
  ```bash
  git clone <repository-url>
  ```

### Files the app needs

The app only needs these parts of the project. If you are sending the project
to a teammate, make sure they are included:

| Path | Size | Purpose |
|---|---|---|
| `site/` | small | The website pages, images, and the Flask app |
| `trained_models/linear_regression/` | ~0.2 MB | The trained model and its test predictions |
| `data/LinearRegression/processed/` | ~15 MB | Prepared data the app reads on startup |
| `requirements.txt` | tiny | The exact Python packages to install |
| `index.html` (project root) | tiny | Redirect used with Live Server |
| `ml/` | small | Scripts that rebuild the data and model (optional) |

The folders inside `data/LinearRegression/` other than `processed/` hold the
**raw** datasets (over 500 MB). They are **not** needed to run the website.

---

## 3. Set up the Python environment

Open a terminal **in the `MachineLearning` folder** (the one that contains
`requirements.txt` and `site/`). In VS Code: **File → Open Folder…** and choose
`MachineLearning`, then **Terminal → New Terminal**.

> **Important:** open the `MachineLearning` folder itself, not a parent folder.
> The commands below assume you are inside it.

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation fails with *“running scripts is disabled on this system”*, run
this once, then activate again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Or skip activation entirely and call the environment’s Python directly:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

When the environment is active, your terminal prompt starts with `(.venv)`.

Installing the packages downloads a few hundred MB (NumPy, pandas,
scikit-learn, Flask). This only has to be done once.

> **Do not change the versions in `requirements.txt`.** The trained model file
> was saved with scikit-learn 1.9.1; other versions may refuse to load it or
> give different results.

In VS Code, if it asks which Python interpreter to use, choose the one inside
`.venv` (**Ctrl+Shift+P → “Python: Select Interpreter”**).

---

## 4. Run the application

From the `MachineLearning` folder:

### Windows (PowerShell)
```powershell
.\.venv\Scripts\python.exe "site\ML Algorithms\app.py"
```

### macOS / Linux
```bash
.venv/bin/python "site/ML Algorithms/app.py"
```

(If the environment is activated, `python "site/ML Algorithms/app.py"` also
works.) Keep the quotes: the folder name `ML Algorithms` contains a space.

You should see:
```
 * Running on http://127.0.0.1:5051
```

Now open these addresses in your browser:

| Page | Address |
|---|---|
| Homepage | <http://127.0.0.1:5051/> |
| Algorithm library | <http://127.0.0.1:5051/analytics.html> |
| About us | <http://127.0.0.1:5051/about.html> |
| Client portal | <http://127.0.0.1:5051/demo.html> |
| **Linear Regression model** | <http://127.0.0.1:5051/linear-regression> |

**To stop the server:** click the terminal and press `Ctrl+C`.
**The server must stay running** while you use the site. If you close the
terminal, the model page will stop working.

### Quick check that everything works
1. Open <http://127.0.0.1:5051/linear-regression>.
2. Click **Load example inputs**, then **Calculate yield estimate**.
3. You should see a yield estimate (about **6.03 t/ha**) and an orange line
   appear on the **Actual vs. predicted yield** chart.

### Optional: run the automated tests
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s "site\ML Algorithms" -p "test_*.py" -v
```
(macOS/Linux: `.venv/bin/python -m unittest discover -s "site/ML Algorithms" -p "test_*.py" -v`)

All 8 tests should pass (`OK`).

---

## 5. Optional: browse the pages with Live Server (port 5500)

You only need this if you prefer VS Code’s **Live Server**. The simplest option
is to skip it and use `http://127.0.0.1:5051/` from section 4.

1. Install the **Live Server** extension in VS Code.
2. **Start the Flask server first** (section 4). The model page and
   predictions are handled by Flask – Live Server cannot run them.
3. Right-click `index.html` in the project root → **Open with Live Server**.
   Your browser opens at `http://127.0.0.1:5500/` and forwards to the homepage.
4. The buttons that open the Linear Regression model automatically send you to
   the Flask server on port **5051**.

**VS Code shortcut (Windows):** **Terminal → Run Task → Start AgroPulse model
server** starts Flask for you. This task is set up for Windows. On macOS/Linux,
start Flask from the terminal as in section 4.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognized (Windows) | Reinstall Python and tick **Add python.exe to PATH**; reopen the terminal. You can also try `py -3.12` instead of `python`. |
| `python3: command not found` (macOS/Linux) | Install Python as in section 1.1. |
| `No module named flask` (or pandas, sklearn…) | The packages are not installed in the environment you are using. Activate `.venv` and run `pip install -r requirements.txt`, or run the app with `.venv` Python as shown in section 4. |
| `Activate.ps1 cannot be loaded… scripts disabled` | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or call `.\.venv\Scripts\python.exe` directly. |
| `FileNotFoundError … missing required trained artifacts` | `trained_models/` or `data/LinearRegression/processed/` was not copied. Get them from the project owner (see section 2). To rebuild them, see section 7. |
| Browser: **This site can’t be reached / ERR_CONNECTION_REFUSED** on port 5051 | The Flask server is not running. Start it (section 4) and keep the terminal open. |
| Browser: **Cannot GET /linear-regression** | You are on Live Server (port 5500) and the link did not switch to Flask. Open <http://127.0.0.1:5051/linear-regression> directly. |
| `Address already in use` / port 5051 busy | Another copy of the app is still running. Stop it with `Ctrl+C` in its terminal (or close that terminal), then start again. |
| Page looks unstyled or an old chart appears | Hard-refresh the browser with `Ctrl+Shift+R` (`Cmd+Shift+R` on Mac). |
| Warning about scikit-learn version when loading the model | Your packages differ from `requirements.txt`. Reinstall inside a fresh `.venv` with `pip install -r requirements.txt`. |
| `Cannot read properties of null` on the model page | The browser is showing an old cached page. Hard-refresh (`Ctrl+Shift+R`) and make sure you opened port **5051**. |

---

## 7. Optional: rebuild the prepared data and trained model

You normally **do not** need this – the prepared data and trained models are
already included. Only do it if the raw datasets are in
`data/LinearRegression/` (they are large and are not required to run the site).

From the `MachineLearning` folder, with the environment active:
```bash
python ml/prepare_linear_regression_data.py
python ml/train_linear_regression.py
```
The first script rebuilds `data/LinearRegression/processed/`; the second
retrains the five Linear Regression models and writes them, with their metrics
and test predictions, to `trained_models/linear_regression/`. Retraining may
change the results slightly, so agree as a team before replacing the shared
model files.

---

## 8. Project layout

```
MachineLearning/
├── INSTRUCTIONS.md             ← this file
├── index.html                  ← redirect to the homepage (for Live Server)
├── requirements.txt            ← exact Python package versions
├── .vscode/tasks.json          ← "Start AgroPulse model server" task (Windows)
├── site/
│   ├── pages/                  ← homepage, About, Algorithm library, Client portal
│   ├── assets/                 ← shared CSS, JavaScript, images
│   └── ML Algorithms/          ← Flask app + Linear Regression page and tests
├── ml/                         ← data-preparation and training scripts
├── data/LinearRegression/
│   └── processed/              ← prepared data used by the app
└── trained_models/
    └── linear_regression/      ← trained models, metrics, test predictions
```

## Notes

- The server is Flask’s built-in development server, intended for local use
  and demos only. It listens on `127.0.0.1` (your own computer), so other
  devices on your network cannot reach it.
- Predictions are educational estimates, not agricultural advice.
- Never commit the `.venv` folder; it is specific to each computer (it is
  already listed in `.gitignore`).
