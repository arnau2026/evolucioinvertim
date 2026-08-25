# Dashboard YTD en Streamlit

## Ejecucion local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Coloca `USAstocks.xlsx` en la misma carpeta que `app.py`, o subelo desde el panel lateral.

## Formato minimo del Excel

- `Ticker`
- `BUY DATE`
- `SELL DATE` (vacia si la posicion sigue abierta)

La cartera se calcula equiponderada entre las posiciones activas y se rebalancea diariamente.
