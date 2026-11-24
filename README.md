# Footprint-targeted

Code of the article "Footprint-targeted field plots correct a regional bias in GEDI biomass and improve 25 m biomass maps in contrasting forests of China".

Data (25 m AGB maps and plot table): https://doi.org/10.5281/zenodo.24516351

| Script | Step |
|---|---|
| `gedi_extract.py`, `gedi_merge.py` | GEDI Version 3 footprints |
| `predictors.py`, `palsar_fill.py` | 2021 predictors at 25 m |
| `training_table.py` | footprints with predictors |
| `plots_l4a.py` | plots against GEDI L4A, footprint repeatability |
| `calibration.py` | RH-AGB calibration |
| `quality_control.py` | quality-control schemes |
| `map_agb.py` | 25 m AGB maps and uncertainty |
| `validate_plots.py` | leave-one-plot-out validation |
| `compare_products.py` | GEDI L4B and ESA CCI Biomass |

Python 3.12, packages in `requirements.txt`. Set `YOUR-EE-PROJECT` (Google Earth Engine); GEDI needs a NASA Earthdata account. Plot coordinates are not public.
