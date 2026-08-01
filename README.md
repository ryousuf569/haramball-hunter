# haramball-hunter

## Setup (Windows / PowerShell)

From the `haramball-hunter` folder that contains this README:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The pinned versions in `requirements.txt` are the ones this was developed against on Python 3.13.
Once the venv is activated, plain `python` refers to it, so the commands below need no version flag.

If PowerShell blocks the activate script, allow it for the current session first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Run

```powershell
python render.py
```

Deactivate the venv when done:

```powershell
deactivate
```

## References

Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P. (2017). *Physics-Based Modeling of Pass Probabilities in Soccer.* MIT Sloan Sports Analytics Conference.
 
Spearman, W. (2018). *Beyond Expected Goals.* MIT Sloan Sports Analytics Conference.

https://rcsoccersim.readthedocs.io/en/latest/soccerserver.html