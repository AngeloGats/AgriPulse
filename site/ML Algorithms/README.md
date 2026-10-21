# Linear Regression Field Lab

This folder contains a local Flask page that uses the trained CYCleSS
Linear Regression model from `trained_models/linear_regression/models/cycless.joblib`.
The model artifact and source data remain in their existing project folders.

From the project root, start the field lab with:

```powershell
.\.venv\Scripts\python.exe "site\ML Algorithms\app.py"
```

Open <http://127.0.0.1:5051/> for the AgroPulse homepage. The Flask app also
serves the pages from `site/pages/` and shared assets from `site/assets/` on
port 5051. The homepage is available at `/` and `/index.html`; the other main
pages are `/about.html`, `/analytics.html`, and `/demo.html`. Open
`/linear-regression` for the trained model demo.

If you browse the static pages using VS Code Live Server on port 5500, also
start this Flask app. The Algorithm Library's model links automatically open
the demo on port 5051; Live Server alone cannot run Flask routes or predictions.
In VS Code, run **Terminal > Run Task > Start AgroPulse model server** to start
it in a terminal that stays open. Keep that task running while using the model;
stop it with Ctrl+C when finished.

The calculator accepts crop/year and the mean, standard deviation, minimum,
maximum, and valid-observation count for the Ratio, VH, and VV Sentinel-1
summaries. Enter 0 observations when a sensor band has no valid dates; its
summary features are imputed using the corresponding training medians, matching
the prepared training data. Standard deviation can be left blank when it is
unavailable and is imputed the same way. “Load example inputs” fills the form
from a held-out CYCleSS test record; it does not expose that record's actual
yield. The page includes a scatter plot of actual versus predicted yield for
the held-out test fields, with a fitted trend line to illustrate model
performance. Each successful input calculation updates an orange reference on
the chart's predicted-yield axis; the chart does not imply an actual yield is
known for that new field. Inputs are checked for valid summary statistics,
passed to the saved model, and flagged if their satellite values fall outside
training ranges.

Predictions are educational estimates, not agricultural advice. The model was
trained with full-season satellite summaries and is not an early-season
forecast. It was evaluated on 192 held-out grid-group test records with R² of
about 0.66 and MAE of about 1.27 t/ha.
