# Smart Warehouse Decision Support System

A Flask-based college project demonstrating warehouse inventory management and a **fixed seven-day simulation** comparing two replenishment policies:

- Fixed Threshold
- Reorder Point

The simulation uses 10 illustrative demo products. It is separate from the live product inventory and does not write simulation sales/orders into real product stock.

## Features

- Existing warehouse pages for products, suppliers, stock, inventory, replenishment, purchase orders and planning.
- Seven-day policy comparison with a repeatable random seed.
- Clear daily sales/lost-demand bars and graphical comparisons for units sold, lost units and revenue.
- Daily report-period filter (Day 1–Day 7) that updates the selected-period summary and download links.
- Editable Excel report with Management Summary, Daily Metrics, Product Stock & Restock, Daily Inventory Detail and Event Log sheets. Excel includes charts.
- CSV summary export.
- Inventory formula helpers for safety stock, reorder point, EOQ and inventory position.
- Unit tests for simulation repeatability, inventory conservation, ordering/purchase costs and validation.

## Requirements

- Python 3.11 or newer (Python 3.13/3.14 should work with compatible packages)
- Fedora/Linux, Windows, or macOS

## Run on Fedora/Linux

From the extracted project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import app; app.app.run(debug=False, use_reloader=False)"
```

Open <http://127.0.0.1:5000/simulation>, then click **Run Policy Comparison**. The simulation is fixed at seven days.

If you already have a working project and database, **back up the project first** and copy the updated source files into it. Do not replace your `instance/warehouse.db` file. The app calls SQLAlchemy `create_all()` to create missing tables only; it does not drop or replace existing tables.

## Tests

```bash
source .venv/bin/activate
python -m pytest -q
```

## Reports

After running a simulation, use the Excel and CSV download buttons. Select a report period between Day 1 and Day 7 before downloading. Excel is editable and contains multiple sheets and charts; CSV contains the selected-period summary and daily metrics.

## Interpretation notes

- Demo product demand, stock and prices are illustrative; they are not actual store records.
- Service level is fulfilled demand divided by total demand.
- Estimated lost-sales revenue is not net profit.
- Purchase, ordering and holding costs follow the configured simulation assumptions. They should not be described as audited real-world costs.
- This project compares inventory rules; it does not claim to train a machine-learning model.

## Project structure

smart-warehouse-management-system/

- `app.py` — Flask routes, database models, simulation page and exports
- `simulation_engine.py` — seven-day simulation mechanics and event log
- `warehouse_demo.py` — illustrative 10-product demo dataset
- `inventory_logic.py` — inventory formulas
- `templates/` — web pages
- `static/style.css` — shared styling
- `tests/` — automated tests

The ZIP intentionally excludes `.venv`, database files, `.git`, caches and historical-data experiments so that it is smaller and does not carry over a local database or environment-specific binaries.


## Controlled seven-day business scenarios

The Simulation page offers five illustrative scenarios for comparing the Fixed Threshold and Reorder Point policies under the same random seed:

- **Normal demand**: baseline demo assumptions.
- **Demand spike**: average demand is increased by 80% and demand variation by 50%.
- **Supplier delay**: supplier lead times are increased by two days.
- **Low opening stock**: starting stock is reduced to 50% (rounded down, minimum one unit).
- **Combined stress test**: applies all three stress changes.

These are controlled teaching scenarios, not forecasts of a real business. The scenario changes are applied to temporary copies of demo product data; live inventory and the base demo definitions are not modified.

### Recommended verification

1. Run Normal demand with seed 42 and note the summary.
2. Run Demand spike with the same seed. Compare units sold, lost units, service level and revenue.
3. Run Supplier delay with the same seed and compare outstanding quantities and stockouts.
4. Run Combined stress test to demonstrate the interaction of demand, opening stock and lead time.
5. Download Excel and CSV after each run if you need to keep the scenario results.
