import math
import os

import json
import csv
import io
from simulation_engine import run_simulation
from warehouse_demo import DEMO_PRODUCTS, get_demo_products
from ml_forecasting import train_and_forecast_sales, create_ml_forecast_chart
from simulation_charts import (
    generate_all_simulation_charts,
    generate_simulation_pdf_report,
)

from inventory_logic import (
    safety_stock,
    reorder_point,
    economic_order_quantity,
    inventory_position,
)
from flask import Flask, render_template, request, redirect, url_for, send_file, abort
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///warehouse.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Create database object
db = SQLAlchemy(app)


class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(100))
    price = db.Column(db.Float, nullable=False)
    current_stock = db.Column(db.Integer, default=0)
    reorder_level = db.Column(db.Integer, default=10)

    supplier_id = db.Column(
        db.Integer,
        db.ForeignKey("supplier.id"),
        nullable=True
    )

    supplier = db.relationship("Supplier", backref="products")


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    product_id = db.Column(
        db.Integer,
        db.ForeignKey("product.id"),
        nullable=False
    )

    transaction_type = db.Column(
        db.String(20),
        nullable=False
    )

    quantity = db.Column(
        db.Integer,
        nullable=False
    )

    date = db.Column(
        db.DateTime,
        default=db.func.now()
    )
    product = db.relationship("Product", backref="transactions")

class Supplier(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20))
    email = db.Column(db.String(120))


# --------------------------------------------------
# Inventory planning settings
# --------------------------------------------------

class ProductPlanning(db.Model):
    __tablename__ = "product_planning"

    id = db.Column(db.Integer, primary_key=True)

    product_id = db.Column(
        db.Integer,
        db.ForeignKey("product.id"),
        nullable=False,
        unique=True
    )

    unit_cost = db.Column(db.Numeric(10, 2), nullable=False, default=0)
    avg_daily_demand = db.Column(db.Float, nullable=False, default=1.0)
    demand_stddev = db.Column(db.Float, nullable=False, default=0.0)
    lead_time_days = db.Column(db.Integer, nullable=False, default=3)
    ordering_cost = db.Column(db.Numeric(10, 2), nullable=False, default=50)
    holding_cost_per_unit_year = db.Column(
        db.Numeric(10, 2), nullable=False, default=10
    )
    service_level = db.Column(db.Float, nullable=False, default=0.95)
    minimum_order_qty = db.Column(db.Integer, nullable=False, default=1)

    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=db.func.now(),
        onupdate=db.func.now()
    )

    product = db.relationship("Product", backref="planning")


# --------------------------------------------------
# Purchase orders and their items
# --------------------------------------------------

class PurchaseOrder(db.Model):
    __tablename__ = "purchase_order"

    id = db.Column(db.Integer, primary_key=True)

    supplier_id = db.Column(
        db.Integer,
        db.ForeignKey("supplier.id"),
        nullable=True
    )

    order_date = db.Column(
        db.DateTime, nullable=False, default=db.func.now()
    )
    expected_delivery_date = db.Column(db.DateTime)
    status = db.Column(db.String(20), nullable=False, default="PENDING")
    total_cost = db.Column(db.Numeric(12, 2), nullable=False, default=0)

    supplier = db.relationship("Supplier")
    items = db.relationship(
        "PurchaseOrderItem",
        back_populates="purchase_order",
        cascade="all, delete-orphan"
    )


class PurchaseOrderItem(db.Model):
    __tablename__ = "purchase_order_item"

    id = db.Column(db.Integer, primary_key=True)

    purchase_order_id = db.Column(
        db.Integer,
        db.ForeignKey("purchase_order.id"),
        nullable=False
    )

    product_id = db.Column(
        db.Integer,
        db.ForeignKey("product.id"),
        nullable=False
    )

    quantity = db.Column(db.Integer, nullable=False)
    received_quantity = db.Column(db.Integer, nullable=False, default=0)
    unit_cost = db.Column(db.Numeric(10, 2), nullable=False)

    purchase_order = db.relationship(
        "PurchaseOrder", back_populates="items"
    )
    product = db.relationship("Product")


# --------------------------------------------------
# Multi-day simulation runs
# --------------------------------------------------

class SimulationRun(db.Model):
    __tablename__ = "simulation_run"

    id = db.Column(db.Integer, primary_key=True)

    store_type = db.Column(
        db.String(50), nullable=False, default="electronics"
    )
    duration_days = db.Column(db.Integer, nullable=False, default=7)
    random_seed = db.Column(db.Integer, nullable=False, default=42)
    policy_name = db.Column(
        db.String(50), nullable=False, default="reorder_point"
    )
    status = db.Column(db.String(20), nullable=False, default="CREATED")

    config_json = db.Column(db.Text)
    results_json = db.Column(db.Text)

    created_at = db.Column(
        db.DateTime, nullable=False, default=db.func.now()
    )
    started_at = db.Column(db.DateTime)
    finished_at = db.Column(db.DateTime)


# --------------------------------------------------
# Inventory event history
# --------------------------------------------------

class InventoryEvent(db.Model):
    __tablename__ = "inventory_event"

    id = db.Column(db.Integer, primary_key=True)

    simulation_run_id = db.Column(
        db.Integer,
        db.ForeignKey("simulation_run.id"),
        nullable=True
    )

    product_id = db.Column(
        db.Integer,
        db.ForeignKey("product.id"),
        nullable=True
    )

    event_type = db.Column(db.String(30), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=0)
    stock_before = db.Column(db.Integer)
    stock_after = db.Column(db.Integer)
    unit_price = db.Column(db.Numeric(10, 2))
    occurred_at = db.Column(
        db.DateTime, nullable=False, default=db.func.now()
    )
    notes = db.Column(db.Text)

    simulation_run = db.relationship("SimulationRun")
    product = db.relationship("Product")


@app.route("/")
def dashboard():

    total_products = Product.query.count()

    total_stock = db.session.query(
        db.func.sum(Product.current_stock)
    ).scalar() or 0

    low_stock = Product.query.filter(
        Product.current_stock > 0,
        Product.current_stock <= Product.reorder_level
    ).count()

    out_of_stock = Product.query.filter(
        Product.current_stock == 0
    ).count()

    alerts = Product.query.filter(
        Product.current_stock <= Product.reorder_level
    ).all()

    return render_template(
        "dashboard.html",
        total_products=total_products,
        total_stock=total_stock,
        low_stock=low_stock,
        out_of_stock=out_of_stock,
        alerts=alerts
    )

@app.route("/products", methods=["GET", "POST"])
def products():

    if request.method == "POST":

        product = Product(
            name=request.form["name"],
            category=request.form["category"],
            price=float(request.form["price"]),
            current_stock=int(request.form["current_stock"]),
            reorder_level=int(request.form["reorder_level"]),
            supplier_id=int(request.form["supplier_id"])
        )

        db.session.add(product)
        db.session.commit()

        return redirect(url_for("products"))

    all_products = Product.query.all()
    all_suppliers = Supplier.query.all()

    return render_template(
        "products.html",
        products=all_products,
        suppliers=all_suppliers
    )

@app.route("/products/edit/<int:product_id>", methods=["GET", "POST"])
def edit_product(product_id):

    product = db.session.get(Product, product_id)

    if not product:
        return "Product not found", 404

    if request.method == "POST":

        product.name = request.form["name"]
        product.category = request.form["category"]
        product.price = float(request.form["price"])
        product.current_stock = int(request.form["current_stock"])
        product.reorder_level = int(request.form["reorder_level"])
        product.supplier_id = int(request.form["supplier_id"])

        db.session.commit()

        return redirect(url_for("products"))

    suppliers = Supplier.query.all()

    return render_template(
        "edit_product.html",
        product=product,
        suppliers=suppliers
    )


@app.route("/products/delete/<int:product_id>", methods=["POST"])
def delete_product(product_id):

    product = db.session.get(Product, product_id)

    if not product:
        return "Product not found", 404

    db.session.delete(product)
    db.session.commit()

    return redirect(url_for("products"))


@app.route("/stock", methods=["GET", "POST"])
def stock():

    if request.method == "POST":

        product_id = int(request.form["product_id"])
        transaction_type = request.form["transaction_type"]
        quantity = int(request.form["quantity"])

        product = db.session.get(Product, product_id)

        if transaction_type == "IN":

            product.current_stock += quantity

        elif transaction_type == "OUT":

            if quantity > product.current_stock:
                return "Error: Not enough stock available."

            product.current_stock -= quantity

        transaction = Transaction(
            product_id=product.id,
            transaction_type=transaction_type,
            quantity=quantity
        )

        db.session.add(transaction)
        db.session.commit()

        return redirect(url_for("stock"))

    products = Product.query.all()

    transactions = Transaction.query.order_by(
        Transaction.date.desc()
    ).all()

    return render_template(
        "stock.html",
        products=products,
        transactions=transactions
    )

@app.route("/suppliers", methods=["GET", "POST"])
def suppliers():

    if request.method == "POST":

        supplier = Supplier(
            name=request.form["name"],
            phone=request.form["phone"],
            email=request.form["email"]
        )

        db.session.add(supplier)
        db.session.commit()

        return redirect(url_for("suppliers"))

    all_suppliers = Supplier.query.all()

    return render_template(
        "suppliers.html",
        suppliers=all_suppliers
    )

@app.route("/inventory")
def inventory():

    products = Product.query.all()

    return render_template(
        "inventory.html",
        products=products
    )

@app.route("/replenishment")
def replenishment():
    products = Product.query.order_by(Product.name).all()
    recommendations = []

    for product in products:
        planning = ProductPlanning.query.filter_by(
            product_id=product.id
        ).first()

        # Temporary defaults until planning settings are configurable.
        avg_demand = planning.avg_daily_demand if planning else 1.0
        demand_stddev = planning.demand_stddev if planning else 0.0
        lead_time = planning.lead_time_days if planning else 3
        service_level = planning.service_level if planning else 0.95
        ordering_cost = float(planning.ordering_cost) if planning else 50.0
        holding_cost = (
            float(planning.holding_cost_per_unit_year)
            if planning else 10.0
        )
        minimum_order = planning.minimum_order_qty if planning else 1

        unit_cost = float(planning.unit_cost) if planning else 0.0

        # Count quantities ordered but not yet received.
        pending_items = (
            db.session.query(PurchaseOrderItem, PurchaseOrder)
            .join(
                PurchaseOrder,
                PurchaseOrder.id == PurchaseOrderItem.purchase_order_id
            )
            .filter(
                PurchaseOrderItem.product_id == product.id,
                PurchaseOrder.status.notin_(["RECEIVED", "CANCELLED"])
            )
            .all()
        )

        on_order = sum(
            max(0, item.quantity - item.received_quantity)
            for item, order in pending_items
        )

        position = inventory_position(
            on_hand=product.current_stock or 0,
            on_order=on_order,
        )

        ss = safety_stock(
            demand_stddev, lead_time, service_level
        )

        rop = reorder_point(
            avg_demand,
            lead_time,
            demand_stddev,
            service_level,
        )

        annual_demand = avg_demand * 365

        eoq = economic_order_quantity(
            annual_demand,
            ordering_cost,
            holding_cost,
        )

        needs_reorder = position <= rop

        suggested_quantity = (
            max(minimum_order, eoq) if needs_reorder else 0
        )

        recommendations.append({
            "product": product,
            "on_order": on_order,
            "position": position,
            "safety_stock": ss,
            "reorder_point": rop,
            "eoq": eoq,
            "needs_reorder": needs_reorder,
            "suggested_quantity": suggested_quantity,
            "unit_cost": unit_cost,
        })

    return render_template(
        "replenishment.html",
        recommendations=recommendations,
    )


@app.route("/planning/<int:product_id>", methods=["GET", "POST"])
def edit_planning(product_id):
    product = db.get_or_404(Product, product_id)

    planning = ProductPlanning.query.filter_by(
        product_id=product.id
    ).first()

    if planning is None:
        planning = ProductPlanning(product_id=product.id)
        db.session.add(planning)

    if request.method == "POST":
        try:
            avg_demand = float(request.form["avg_daily_demand"])
            demand_stddev = float(request.form["demand_stddev"])
            lead_time = int(request.form["lead_time_days"])
            unit_cost = float(request.form["unit_cost"])
            ordering_cost = float(request.form["ordering_cost"])
            holding_cost = float(
                request.form["holding_cost_per_unit_year"]
            )
            service_level = float(request.form["service_level"])
            minimum_order = int(request.form["minimum_order_qty"])

            if (
                avg_demand < 0
                or demand_stddev < 0
                or lead_time < 0
                or unit_cost < 0
                or ordering_cost < 0
                or holding_cost <= 0
                or not 0 < service_level < 1
                or minimum_order < 1
            ):
                raise ValueError("One or more values are outside the allowed range.")

            planning.avg_daily_demand = avg_demand
            planning.demand_stddev = demand_stddev
            planning.lead_time_days = lead_time
            planning.unit_cost = unit_cost
            planning.ordering_cost = ordering_cost
            planning.holding_cost_per_unit_year = holding_cost
            planning.service_level = service_level
            planning.minimum_order_qty = minimum_order

            db.session.commit()
            return redirect(url_for("replenishment"))

        except (ValueError, TypeError):
            db.session.rollback()
            return render_template(
                "planning.html",
                product=product,
                planning=planning,
                error="Enter valid numbers. Holding cost must be positive, "
                      "service level must be between 0 and 1, and minimum "
                      "order quantity must be at least 1."
            )

    return render_template(
        "planning.html",
        product=product,
        planning=planning,
        error=None
    )


@app.route("/purchase-orders", methods=["GET", "POST"])
def purchase_orders():
    error = None

    if request.method == "POST":
        try:
            supplier_value = request.form.get("supplier_id", "").strip()
            supplier_id = int(supplier_value) if supplier_value else None

            product_id = int(request.form["product_id"])
            quantity = int(request.form["quantity"])
            unit_cost = float(request.form["unit_cost"])

            expected_value = request.form.get(
                "expected_delivery_date", ""
            ).strip()

            expected_date = (
                datetime.strptime(expected_value, "%Y-%m-%d")
                if expected_value else None
            )

            if quantity <= 0:
                raise ValueError("Quantity must be positive.")

            if not math.isfinite(unit_cost) or unit_cost < 0:
                raise ValueError("Unit cost must be a finite, non-negative number.")

            product = db.get_or_404(Product, product_id)

            if supplier_id is not None:
                db.get_or_404(Supplier, supplier_id)

            order = PurchaseOrder(
                supplier_id=supplier_id,
                expected_delivery_date=expected_date,
                status="PENDING",
                total_cost=round(quantity * unit_cost, 2),
            )

            order.items.append(
                PurchaseOrderItem(
                    product_id=product.id,
                    quantity=quantity,
                    received_quantity=0,
                    unit_cost=unit_cost,
                )
            )

            db.session.add(order)
            db.session.commit()

            # Important: creating an order does NOT increase stock.
            return redirect(url_for("purchase_orders"))

        except (ValueError, TypeError):
            db.session.rollback()
            error = "Enter a valid product, quantity, cost and delivery date."

    products = Product.query.order_by(Product.name).all()
    suppliers = Supplier.query.order_by(Supplier.name).all()
    orders = PurchaseOrder.query.order_by(
        PurchaseOrder.order_date.desc()
    ).all()

    return render_template(
        "purchase_orders.html",
        products=products,
        suppliers=suppliers,
        orders=orders,
        error=error,
    )


@app.route("/purchase-orders/<int:order_id>/receive", methods=["POST"])
def receive_purchase_order(order_id):
    order = db.get_or_404(PurchaseOrder, order_id)

    if order.status in ("RECEIVED", "CANCELLED"):
        return "This order cannot receive more stock.", 400

    try:
        item_id = int(request.form["item_id"])
        receive_quantity = int(request.form["quantity"])

        if receive_quantity <= 0:
            raise ValueError("Quantity must be positive.")

        item = next(
            (item for item in order.items if item.id == item_id),
            None
        )

        if item is None:
            return "Item does not belong to this order.", 400

        remaining = item.quantity - item.received_quantity

        if receive_quantity > remaining:
            return f"Only {remaining} units remain outstanding.", 400

        product = db.session.get(Product, item.product_id)

        if product is None:
            return "Product not found.", 404

        stock_before = product.current_stock or 0
        stock_after = stock_before + receive_quantity

        # Update the item and physical stock together.
        item.received_quantity += receive_quantity
        product.current_stock = stock_after

        db.session.add(Transaction(
            product_id=product.id,
            transaction_type="IN",
            quantity=receive_quantity,
        ))

        db.session.add(InventoryEvent(
            simulation_run_id=None,
            product_id=product.id,
            event_type="DELIVERY_RECEIVED",
            quantity=receive_quantity,
            stock_before=stock_before,
            stock_after=stock_after,
            unit_price=item.unit_cost,
            notes=f"Purchase order #{order.id}",
        ))

        all_received = all(
            order_item.received_quantity == order_item.quantity
            for order_item in order.items
        )

        any_received = any(
            order_item.received_quantity > 0
            for order_item in order.items
        )

        if all_received:
            order.status = "RECEIVED"
        elif any_received:
            order.status = "PARTIALLY_RECEIVED"
        else:
            order.status = "PENDING"

        db.session.commit()

    except (ValueError, TypeError, KeyError):
        db.session.rollback()
        return "Invalid delivery quantity.", 400

    return redirect(url_for("purchase_orders"))





@app.route("/simulation", methods=["GET", "POST"])
def simulation():
    previous_runs = (
        SimulationRun.query
        .order_by(SimulationRun.created_at.desc())
        .limit(10)
        .all()
    )

    if request.method == "GET":
        return render_template(
            "simulation.html",
            results=None,
            ml_forecast=None,
            previous_runs=previous_runs,
            demand_source="warehouse_demo",
            duration_days=7,
            random_seed=42,
            store_type="electronics",
            scenario="normal",
        )

    try:
        random_seed = int(request.form.get("random_seed", 42))
        if not 0 <= random_seed <= 2147483647:
            raise ValueError("Seed must be between 0 and 2147483647.")

        # The college requirement is a fixed seven-day simulation.
        duration_days = 7
        submitted_duration = request.form.get("duration_days", "7")
        if str(submitted_duration).strip() not in ("", "7"):
            raise ValueError("This project is configured for exactly 7 simulation days.")

        store_type = request.form.get("store_type", "electronics")
        scenario = request.form.get("scenario", "normal")
        scenario_options = {
            "normal": "Normal demand",
            "demand_spike": "Demand spike (higher demand)",
            "supplier_delay": "Supplier delays (+2 days)",
            "low_stock": "Low opening stock (50%)",
            "combined_stress": "Combined stress test",
        }
        if scenario not in scenario_options:
            raise ValueError("Select a valid simulation scenario.")

        # Select product catalog according to selected store_type
        # Work on copies only: scenario adjustments never alter demo definitions
        # or live inventory. Both policies receive identical scenario conditions.
        simulation_products = get_demo_products(store_type)
        if scenario in ("demand_spike", "combined_stress"):
            for item in simulation_products:
                item["avg_daily_demand"] = max(1, round(item["avg_daily_demand"] * 1.8))
                item["demand_stddev"] = max(1, round(item["demand_stddev"] * 1.5))
        if scenario in ("supplier_delay", "combined_stress"):
            for item in simulation_products:
                item["lead_time_days"] += 2
        if scenario in ("low_stock", "combined_stress"):
            for item in simulation_products:
                item["initial_stock"] = max(1, item["initial_stock"] // 2)

        results = {}

        for policy in ("fixed_threshold", "reorder_point"):
            result = run_simulation(
                simulation_products,
                duration_days=duration_days,
                policy=policy,
                random_seed=random_seed,
            )
            result["scenario"] = scenario
            result["scenario_label"] = scenario_options[scenario]

            run = SimulationRun(
                store_type=store_type,
                duration_days=duration_days,
                random_seed=random_seed,
                policy_name=policy,
                status="COMPLETED",
                config_json=json.dumps({
                    "store_type": store_type,
                    "duration_days": duration_days,
                    "random_seed": random_seed,
                    "policy": policy,
                    "scenario": scenario,
                    "scenario_label": scenario_options[scenario],
                    "demand_source": "warehouse_demo",
                    "products": simulation_products,
                }),
                results_json=json.dumps(result),
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )
            db.session.add(run)
            db.session.flush()

            for event in result["events"]:
                db.session.add(InventoryEvent(
                    simulation_run_id=run.id,
                    product_id=None,
                    event_type=event["event_type"],
                    quantity=event["quantity"],
                    stock_before=event["stock_before"],
                    stock_after=event["stock_after"],
                    unit_price=next(
                        (
                            item["unit_cost"]
                            for item in simulation_products
                            if item["id"] == event["product_id"]
                        ),
                        None,
                    ),
                    notes=f'Simulation day {event["day"]}: {event["notes"]}',
                ))

            results[policy] = {"run_id": run.id, **result}

        # Train Machine Learning 7-Day Forward Sales & Demand Forecast (Days 8-14)
        try:
            ml_forecast = train_and_forecast_sales(results["reorder_point"], simulation_products, forecast_days=7)
        except Exception as ml_err:
            app.logger.warning(f"ML forecast error: {ml_err}")
            ml_forecast = {}

        db.session.commit()

        return render_template(
            "simulation.html",
            results=results,
            ml_forecast=ml_forecast,
            previous_runs=(
                SimulationRun.query
                .order_by(SimulationRun.created_at.desc())
                .limit(10)
                .all()
            ),
            demand_source="warehouse_demo",
            duration_days=duration_days,
            random_seed=random_seed,
            store_type=store_type,
            scenario=scenario,
        )

    except (ValueError, TypeError, KeyError) as exc:
        db.session.rollback()
        return render_template(
            "simulation.html",
            results=None,
            previous_runs=previous_runs,
            demand_source="warehouse_demo",
            duration_days=request.form.get("duration_days", 7),
            random_seed=request.form.get("random_seed", 42),
            store_type=request.form.get("store_type", "electronics"),
            scenario=request.form.get("scenario", "normal"),
            error=str(exc),
        ), 400

    except Exception:
        db.session.rollback()
        app.logger.exception("Warehouse demo simulation failed")
        return render_template(
            "simulation.html",
            results=None,
            previous_runs=previous_runs,
            demand_source="warehouse_demo",
            duration_days=request.form.get("duration_days", 7),
            random_seed=request.form.get("random_seed", 42),
            store_type=request.form.get("store_type", "electronics"),
            error="The simulation failed. Check the application logs.",
        ), 500


@app.route("/simulation/report/<int:fixed_run_id>/<int:rop_run_id>/<file_format>")
def download_simulation_report(fixed_run_id, rop_run_id, file_format):
    """Download an editable report or visual executive PDF for a completed two-policy simulation."""
    if file_format not in ("xlsx", "csv", "pdf"):
        abort(404)

    fixed_run = db.get_or_404(SimulationRun, fixed_run_id)
    rop_run = db.get_or_404(SimulationRun, rop_run_id)
    if fixed_run.policy_name != "fixed_threshold" or rop_run.policy_name != "reorder_point":
        abort(400, description="The selected runs are not a matching policy comparison.")
    if fixed_run.duration_days != 7 or rop_run.duration_days != 7:
        abort(400, description="Reports are available for seven-day simulation runs only.")
    if fixed_run.random_seed != rop_run.random_seed or fixed_run.store_type != rop_run.store_type:
        abort(400, description="The selected runs do not belong to the same simulation configuration.")

    fixed_result = json.loads(fixed_run.results_json or "{}")
    rop_result = json.loads(rop_run.results_json or "{}")
    fixed_config = json.loads(fixed_run.config_json or "{}")
    rop_config = json.loads(rop_run.config_json or "{}")

    # Train forward Machine Learning Forecast for inclusion in downloadable reports
    try:
        ml_forecast = train_and_forecast_sales(rop_result, fixed_config.get("products", []), forecast_days=7)
    except Exception:
        ml_forecast = {}

    try:
        start_day = min(int(fixed_run.duration_days), max(1, int(request.args.get("start_day", 1))))
        end_day = min(int(fixed_run.duration_days), max(1, int(request.args.get("end_day", fixed_run.duration_days))))
    except (TypeError, ValueError):
        start_day, end_day = 1, int(fixed_run.duration_days)
    if start_day > end_day:
        start_day, end_day = end_day, start_day

    products_by_id = {
        str(p["id"]): p for p in fixed_config.get("products", [])
    }

    def in_period(item):
        return start_day <= int(item.get("day", 0)) <= end_day

    def selected_daily(result):
        return [item for item in result.get("daily", []) if in_period(item)]

    def period_metrics(result):
        daily = selected_daily(result)
        sold = sum(int(x.get("units_sold", 0)) for x in daily)
        lost = sum(int(x.get("lost_units", 0)) for x in daily)
        revenue = round(sum(float(x.get("revenue", 0)) for x in daily), 2)
        demand = sold + lost
        return {
            "Units Sold": sold,
            "Lost Units": lost,
            "Total Demand": demand,
            "Service Level (%)": round(sold / demand * 100, 2) if demand else 100.0,
            "Revenue (₹)": revenue,
        }

    def lost_sales_revenue(result, config):
        product_prices = {str(p["id"]): float(p.get("selling_price", 0)) for p in config.get("products", [])}
        return {
            str(item.get("product_id")): int(item.get("lost_units", 0)) * product_prices.get(str(item.get("product_id")), 0)
            for item in result.get("products", [])
        }

    if file_format == "pdf":
        pdf_buf = generate_simulation_pdf_report(
            fixed_run, rop_run, fixed_result, rop_result, fixed_config, rop_config, start_day, end_day
        )
        return send_file(
            pdf_buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"smart_warehouse_visual_report_days_{start_day}_{end_day}.pdf",
        )

    if file_format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["SMART WAREHOUSE SIMULATION REPORT"])
        writer.writerow(["Store type", fixed_run.store_type, "Duration (days)", fixed_run.duration_days, "Random seed", fixed_run.random_seed])
        writer.writerow(["Report period", f"Day {start_day} to Day {end_day}"])
        writer.writerow([])
        writer.writerow(["Selected-Period Comparison"])
        writer.writerow(["Metric", "Fixed Threshold", "Reorder Point"])
        fm, rm = period_metrics(fixed_result), period_metrics(rop_result)
        for key in fm:
            writer.writerow([key, fm[key], rm[key]])
        writer.writerow([])
        writer.writerow(["Daily Results"])
        writer.writerow(["Policy", "Day", "Units Sold", "Lost Units", "Revenue (₹)"])
        for label, result in (("Fixed Threshold", fixed_result), ("Reorder Point", rop_result)):
            for item in selected_daily(result):
                writer.writerow([label, item.get("day"), item.get("units_sold", 0), item.get("lost_units", 0), item.get("revenue", 0)])

        if ml_forecast and ml_forecast.get("daily_forecast"):
            writer.writerow([])
            writer.writerow(["MACHINE LEARNING 7-DAY FORWARD SALES FORECAST (DAYS 8-14)"])
            writer.writerow(["Model", ml_forecast["summary"]["model_name"], "R2 Score", ml_forecast["summary"]["r2_score"], "MAE", ml_forecast["summary"]["mae"], "RMSE", ml_forecast["summary"]["rmse"]])
            writer.writerow(["Total Predicted Units", ml_forecast["summary"]["total_predicted_units"], "Total Projected Revenue (₹)", ml_forecast["summary"]["total_predicted_revenue"]])
            writer.writerow([])
            writer.writerow(["Daily Machine Learning Projections"])
            writer.writerow(["Day", "Predicted Units Sold", "95% Lower Bound", "95% Upper Bound", "Projected Daily Revenue (₹)", "Confidence Level"])
            for d in ml_forecast["daily_forecast"]:
                writer.writerow([f"Day {d['day']}", d["predicted_units"], d["lower_bound_units"], d["upper_bound_units"], d["predicted_revenue"], f"{d['confidence_pct']}%"])
            writer.writerow([])
            writer.writerow(["Product-Level Forward Stockout Risk Assessment"])
            writer.writerow(["Product ID", "Product Name", "Category", "Selling Price (₹)", "Ending Stock (Day 7)", "7-Day Forecast Demand", "Projected Stock (Day 14)", "Stockout Day", "Risk Assessment", "Action Recommendation"])
            for p in ml_forecast.get("product_forecast", []):
                writer.writerow([
                    p["product_id"], p["product_name"], p["category"], p["selling_price"],
                    p["ending_stock"], p["forecast_units"], p["projected_stock"],
                    f"Day {p['stockout_day']}" if p.get("stockout_day") else "None",
                    p["risk_label"], p["recommendation"]
                ])

        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8-sig")),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"smart_warehouse_report_days_{start_day}_{end_day}.csv",
        )

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.chart import BarChart, LineChart, DoughnutChart, Reference
        from openpyxl.drawing.image import Image
        from openpyxl.utils import get_column_letter
    except ImportError:
        abort(500, description="Excel export requires openpyxl. Install it with: pip install openpyxl")

    wb = Workbook()

    # Sheet 1: Dedicated Visual Dashboard with High-Resolution Diverse Charts
    charts = generate_all_simulation_charts(fixed_result, rop_result, start_day, end_day, fixed_config, ml_forecast)
    dash_ws = wb.active
    dash_ws.title = "Visual Dashboard"
    dash_ws.views.sheetView[0].showGridLines = True

    dash_ws["B2"] = "SMART WAREHOUSE — SIMULATION GRAPHICAL DASHBOARD"
    dash_ws["B2"].font = Font(size=15, bold=True, color="17365D")
    scenario_str = fixed_result.get("scenario_label", fixed_config.get("scenario", "Normal demand"))
    dash_ws["B3"] = f"Store: {fixed_run.store_type.title()}  |  Scenario: {scenario_str}  |  Seed: {fixed_run.random_seed}  |  Report Period: Day {start_day} to Day {end_day} (Full: {fixed_run.duration_days} Days)"
    dash_ws["B3"].font = Font(size=10, italic=True, color="475569")

    chart_cards = [
        ("B5", "B6", "1. OPERATIONAL COST STRUCTURE COMPOSITION (DONUT CHART)", charts["cost_donut"]),
        ("M5", "M6", "2. FULFILLMENT SERVICE LEVEL (%) (GAUGE CHART)", charts["service_gauge"]),
        ("B25", "B26", "3. TOTAL WAREHOUSE INVENTORY LEVEL TRAJECTORY (LINE & AREA)", charts["inventory_trajectory"]),
        ("M25", "M26", "4. DAILY SALES REVENUE TRAJECTORY (MULTI-LINE TREND)", charts["daily_revenue"]),
        ("B45", "B46", "5. SKU VOLATILITY & LEAD-TIME RISK MATRIX (SCATTER PLOT)", charts["sku_risk_scatter"]),
        ("M45", "M46", "6. DAILY DEMAND FULFILLMENT: SOLD VS. LOST (GROUPED BAR)", charts["daily_fulfillment"]),
    ]

    for title_cell, img_cell, title_text, buf in chart_cards:
        dash_ws[title_cell] = title_text
        dash_ws[title_cell].font = Font(size=10, bold=True, color="17365D")
        img = Image(io.BytesIO(buf.getvalue()))
        img.width = 540
        img.height = 288
        dash_ws.add_image(img, img_cell)

    if "ml_forecast" in charts:
        dash_ws["B66"] = "7. MACHINE LEARNING 7-DAY FORWARD SALES FORECAST (DAYS 8–14)"
        dash_ws["B66"].font = Font(size=10, bold=True, color="17365D")
        img_ml = Image(io.BytesIO(charts["ml_forecast"].getvalue()))
        img_ml.width = 540
        img_ml.height = 288
        dash_ws.add_image(img_ml, "B67")
        dash_ws.merge_cells("B83:K83")
        dash_ws["B83"] = "• Explanatory Guide: Predictive Ridge Regression curve with 95% confidence interval band forecasting daily warehouse sales for Days 8–14."
        dash_ws["B83"].font = Font(size=9.5, italic=True, color="334155")
        dash_ws["B83"].fill = PatternFill("solid", fgColor="F5F3FF")
        dash_ws["B83"].alignment = Alignment(vertical="center", indent=1)
        dash_ws.row_dimensions[83].height = 24

    # Explanatory text lines underneath each chart explaining the visualizations and reference lines
    explanation_rows = [
        ("B22", "B22:K22", 22, "• Explanatory Guide: Donut rings display cost proportion. Purchasing dominates (>90%); ROP optimizes batch ordering to cut holding costs."),
        ("M22", "M22:V22", 22, "• Explanatory Guide: The red tick mark indicates the 95% Target Line. Gauge readings above this line avoid order stockout penalties."),
        ("B42", "B42:K42", 42, "• Explanatory Guide: The red dashed line marks the Safety Stock Buffer Line. Upward spikes reflect incoming replenishment deliveries."),
        ("M42", "M42:V42", 42, "• Explanatory Guide: The dotted amber line is the Average Daily Revenue Baseline. Trajectory dips below indicate lost revenue from stockouts."),
        ("B62", "B62:K62", 62, "• Explanatory Guide: Crosshairs (Avg Lead Time & Avg Demand) divide SKUs into 4 risk quadrants. Top-right items need highest safety buffers."),
        ("M62", "M62:V62", 62, "• Explanatory Guide: The dotted purple line marks the Average Demand Target Line. Red hatched bars mark unfulfilled lost customer demand."),
    ]
    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )
    for cell_id, merge_range, row_idx, expl_text in explanation_rows:
        dash_ws.merge_cells(merge_range)
        dash_ws[cell_id] = expl_text
        dash_ws[cell_id].font = Font(size=9.5, italic=True, color="334155")
        dash_ws[cell_id].fill = PatternFill("solid", fgColor="F1F5F9")
        dash_ws[cell_id].alignment = Alignment(vertical="center", indent=1)
        dash_ws.row_dimensions[row_idx].height = 24

    dash_ws.column_dimensions["A"].width = 3
    for col in ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]:
        dash_ws.column_dimensions[col].width = 10
    dash_ws.column_dimensions["L"].width = 4
    for col in ["M", "N", "O", "P", "Q", "R", "S", "T", "U", "V"]:
        dash_ws.column_dimensions[col].width = 10

    # Sheet: Machine Learning Sales & Inventory Forecast
    if ml_forecast and ml_forecast.get("daily_forecast"):
        ml_ws = wb.create_sheet("ML Sales Forecast")
        ml_ws.append(["MACHINE LEARNING 7-DAY FORWARD SALES & INVENTORY FORECAST"])
        ml_ws.append(["Model Architecture", ml_forecast["summary"]["model_name"]])
        ml_ws.append(["Forecast Window", ml_forecast["summary"]["forecast_window"], "Confidence Level", "95%"])
        ml_ws.append(["Model R² Score", ml_forecast["summary"]["r2_score"], "Mean Absolute Error (MAE)", f"{ml_forecast['summary']['mae']} units", "RMSE", f"{ml_forecast['summary']['rmse']} units"])
        ml_ws.append(["Total Predicted 7-Day Sales", f"{ml_forecast['summary']['total_predicted_units']} units", "Total Projected 7-Day Revenue", f"₹{ml_forecast['summary']['total_predicted_revenue']:,.2f}"])
        ml_ws.append([])
        ml_ws.append(["Daily Machine Learning Forward Forecast"])
        ml_ws.append(["Timeline Day", "Predicted Sales (Units)", "95% Lower Bound", "95% Upper Bound", "Projected Daily Revenue (₹)", "Confidence Level"])
        for d in ml_forecast.get("daily_forecast", []):
            ml_ws.append([f"Day {d['day']}", d["predicted_units"], d["lower_bound_units"], d["upper_bound_units"], d["predicted_revenue"], f"{d['confidence_pct']}%"])
        ml_ws.append([])
        ml_ws.append(["Product-Level Forward Demand & Stockout Risk Assessment"])
        ml_ws.append(["Product ID", "Product Name", "Category", "Selling Price (₹)", "Ending Stock (Day 7)", "7-Day Forecast Demand", "Projected Stock (Day 14)", "Stockout Day", "Risk Status", "Restock Recommendation"])
        for p in ml_forecast.get("product_forecast", []):
            ml_ws.append([
                p["product_id"], p["product_name"], p["category"], p["selling_price"],
                p["ending_stock"], p["forecast_units"], p["projected_stock"],
                f"Day {p['stockout_day']}" if p.get("stockout_day") else "None",
                p["risk_label"], p["recommendation"]
            ])
        if "ml_forecast" in charts:
            ml_img = Image(io.BytesIO(charts["ml_forecast"].getvalue()))
            ml_img.width = 520
            ml_img.height = 280
            ml_ws.add_image(ml_img, "K4")

    # Sheet 2: Management Summary
    summary_ws = wb.create_sheet("Management Summary")
    summary_ws.append(["SMART WAREHOUSE — MANAGEMENT REPORT"])
    summary_ws.append(["Store Type", fixed_run.store_type, "Simulation Days", fixed_run.duration_days, "Random Seed", fixed_run.random_seed])
    summary_ws.append(["Selected Report Period", f"Day {start_day}–Day {end_day}"])
    summary_ws.append([])
    summary_ws.append(["Selected-Period Comparison", "Fixed Threshold", "Reorder Point"])
    fm, rm = period_metrics(fixed_result), period_metrics(rop_result)
    for metric in fm:
        summary_ws.append([metric, fm[metric], rm[metric]])
    summary_ws.append([])
    summary_ws.append(["Full-Simulation Cost Comparison", "Fixed Threshold", "Reorder Point"])
    fs, rs = fixed_result.get("summary", {}), rop_result.get("summary", {})
    cost_start_row = summary_ws.max_row
    for label, key in (
        ("Total Purchase Cost (₹)", "total_purchase_cost"),
        ("Total Ordering Cost (₹)", "total_ordering_cost"),
        ("Total Holding Cost (₹)", "total_holding_cost"),
        ("Total Revenue (₹)", "total_revenue"),
        ("Total Units Sold", "total_units_sold"),
        ("Total Lost Units", "total_lost_units"),
        ("Full-Simulation Service Level (%)", "service_level"),
    ):
        left, right = fs.get(key, 0), rs.get(key, 0)
        if key == "service_level":
            left, right = round(float(left) * 100, 2), round(float(right) * 100, 2)
        summary_ws.append([label, left, right])
    summary_ws.append([])
    summary_ws.append(["Interpretation"])
    summary_ws.append(["Lost sales revenue is an estimate (lost units × configured selling price), not net profit."])
    summary_ws.append(["The report is based on simulated demo data and does not change live inventory."])

    # Dynamic openpyxl Native DoughnutChart for Cost Structure
    donut_chart = DoughnutChart()
    donut_chart.title = "Cost Structure Share (Donut Chart)"
    donut_chart.width = 15
    donut_chart.height = 10
    donut_chart.add_data(Reference(summary_ws, min_col=2, min_row=cost_start_row, max_row=cost_start_row + 3), titles_from_data=True)
    donut_chart.set_categories(Reference(summary_ws, min_col=1, min_row=cost_start_row + 1, max_row=cost_start_row + 3))
    summary_ws.add_chart(donut_chart, "E4")

    # Sheet 3: Daily Metrics
    daily_ws = wb.create_sheet("Daily Metrics")
    daily_ws.append(["Day", "Fixed Units Sold", "Fixed Lost Units", "Fixed Revenue (₹)", "ROP Units Sold", "ROP Lost Units", "ROP Revenue (₹)"])
    f_daily = {int(x["day"]): x for x in selected_daily(fixed_result)}
    r_daily = {int(x["day"]): x for x in selected_daily(rop_result)}
    for day in range(start_day, end_day + 1):
        f, r = f_daily.get(day, {}), r_daily.get(day, {})
        daily_ws.append([day, f.get("units_sold", 0), f.get("lost_units", 0), f.get("revenue", 0),
                         r.get("units_sold", 0), r.get("lost_units", 0), r.get("revenue", 0)])

    sales_chart = BarChart()
    sales_chart.type = "col"
    sales_chart.style = 10
    sales_chart.title = "Daily Units Sold by Policy"
    sales_chart.y_axis.title = "Units Sold"
    sales_chart.x_axis.title = "Simulation Day"
    sales_chart.width = 16
    sales_chart.height = 9
    sales_chart.add_data(Reference(daily_ws, min_col=2, max_col=2, min_row=1, max_row=daily_ws.max_row), titles_from_data=True)
    sales_chart.add_data(Reference(daily_ws, min_col=5, max_col=5, min_row=1, max_row=daily_ws.max_row), titles_from_data=True)
    sales_chart.set_categories(Reference(daily_ws, min_col=1, min_row=2, max_row=daily_ws.max_row))
    daily_ws.add_chart(sales_chart, "I2")

    lost_chart = BarChart()
    lost_chart.type = "col"
    lost_chart.style = 12
    lost_chart.title = "Daily Lost Units by Policy"
    lost_chart.y_axis.title = "Lost Units"
    lost_chart.x_axis.title = "Simulation Day"
    lost_chart.width = 16
    lost_chart.height = 9
    lost_chart.add_data(Reference(daily_ws, min_col=3, max_col=3, min_row=1, max_row=daily_ws.max_row), titles_from_data=True)
    lost_chart.add_data(Reference(daily_ws, min_col=6, max_col=6, min_row=1, max_row=daily_ws.max_row), titles_from_data=True)
    lost_chart.set_categories(Reference(daily_ws, min_col=1, min_row=2, max_row=daily_ws.max_row))
    daily_ws.add_chart(lost_chart, "I17")

    rev_chart = LineChart()
    rev_chart.title = "Daily Revenue Trajectory (₹)"
    rev_chart.y_axis.title = "Revenue (₹)"
    rev_chart.x_axis.title = "Simulation Day"
    rev_chart.width = 16
    rev_chart.height = 9
    rev_chart.add_data(Reference(daily_ws, min_col=4, max_col=4, min_row=1, max_row=daily_ws.max_row), titles_from_data=True)
    rev_chart.add_data(Reference(daily_ws, min_col=7, max_col=7, min_row=1, max_row=daily_ws.max_row), titles_from_data=True)
    rev_chart.set_categories(Reference(daily_ws, min_col=1, min_row=2, max_row=daily_ws.max_row))
    daily_ws.add_chart(rev_chart, "I32")

    # Sheet 4: Product-level stock status
    product_ws = wb.create_sheet("Product Stock & Restock")
    product_ws.append(["Policy", "Product", "Final Stock", "On Order", "Reorder Threshold", "Lead Time (days)",
                       "Units Sold", "Lost Units", "Stockout Days", "Orders Placed", "Estimated Lost Sales Revenue (₹)", "Restock Status"])
    for label, result, config in (
        ("Fixed Threshold", fixed_result, fixed_config),
        ("Reorder Point", rop_result, rop_config),
    ):
        config_by_id = {str(p["id"]): p for p in config.get("products", [])}
        lost_rev = lost_sales_revenue(result, config)
        for product in result.get("products", []):
            cfg = config_by_id.get(str(product.get("product_id")), {})
            threshold = cfg.get("reorder_level", 0)
            if label == "Reorder Point":
                try:
                    threshold = reorder_point(float(cfg.get("avg_daily_demand", 0)), int(cfg.get("lead_time_days", 0)),
                                              float(cfg.get("demand_stddev", 0)), float(cfg.get("service_level", 0.95)))
                except (ValueError, TypeError):
                    threshold = cfg.get("reorder_level", 0)
            stock = int(product.get("final_stock", 0))
            on_order = int(product.get("outstanding_quantity", 0))
            if stock + on_order <= int(threshold):
                status = "REVIEW / RESTOCK"
            elif int(product.get("stockout_days", 0)) > 0:
                status = "STOCKOUT OCCURRED"
            else:
                status = "Monitor"
            product_ws.append([
                label, product.get("product_name"), stock, on_order, threshold, cfg.get("lead_time_days", ""),
                product.get("units_sold", 0), product.get("lost_units", 0), product.get("stockout_days", 0),
                product.get("orders_placed", 0), round(lost_rev.get(str(product.get("product_id")), 0), 2), status
            ])

    # Sheet 5: Per-product, per-day inventory movements
    inventory_ws = wb.create_sheet("Daily Inventory Detail")
    inventory_ws.append(["Policy", "Day", "Product", "Opening Stock", "Demand", "Units Sold", "Lost Units",
                         "Deliveries Received", "Units Ordered", "Closing Stock", "On Order", "Reorder Level",
                         "Lead Time (days)", "Status"])
    for label, result in (("Fixed Threshold", fixed_result), ("Reorder Point", rop_result)):
        for item in result.get("daily_items", []):
            if not in_period(item):
                continue
            cfg = products_by_id.get(str(item.get("product_id")), {})
            threshold = item.get("reorder_level", 0)
            if label == "Reorder Point":
                try:
                    threshold = reorder_point(float(cfg.get("avg_daily_demand", 0)), int(item.get("lead_time_days", 0)),
                                              float(cfg.get("demand_stddev", 0)), float(cfg.get("service_level", 0.95)))
                except (ValueError, TypeError):
                    pass
            stock_position = int(item.get("closing_stock", 0)) + int(item.get("on_order", 0))
            status = "RESTOCK TRIGGER" if stock_position <= int(threshold) else "OK"
            inventory_ws.append([
                label, item.get("day"), item.get("product_name"), item.get("opening_stock", 0),
                item.get("demand", 0), item.get("units_sold", 0), item.get("lost_units", 0),
                item.get("deliveries_received", 0), item.get("units_ordered", 0), item.get("closing_stock", 0),
                item.get("on_order", 0), threshold, item.get("lead_time_days", 0), status
            ])

    # Sheet 6: Event Log
    events_ws = wb.create_sheet("Event Log")
    events_ws.append(["Policy", "Day", "Product ID", "Event", "Quantity", "Stock Before", "Stock After", "Notes"])
    for label, result in (("Fixed Threshold", fixed_result), ("Reorder Point", rop_result)):
        for event in result.get("events", []):
            if in_period(event):
                events_ws.append([label, event.get("day"), event.get("product_id"), event.get("event_type"),
                                  event.get("quantity"), event.get("stock_before"), event.get("stock_after"), event.get("notes", "")])

    # Apply formatting: auto-filter, headers, and column dimensions to tabular sheets
    for ws in wb.worksheets:
        if ws.title == "Visual Dashboard":
            continue
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="17365D")
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 30
        for col in ws.columns:
            col_letter = get_column_letter(col[0].column)
            max_len = max((len(str(c.value)) if c.value is not None else 0 for c in col[:100]), default=10)
            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 34)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        if ws.title == "Management Summary":
            ws.column_dimensions["A"].width = 42
            ws.column_dimensions["B"].width = 24
            ws.column_dimensions["C"].width = 24

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"smart_warehouse_report_days_{start_day}_{end_day}.xlsx",
    )


def seed_initial_data():
    """Seed realistic warehouse products and suppliers if database is empty."""
    if Supplier.query.count() == 0:
        suppliers_data = [
            {"name": "TechSource Global", "phone": "+91 98201 12345", "email": "supply@techsourceglobal.com"},
            {"name": "AudioMax Instruments", "phone": "+91 98450 67890", "email": "orders@audiomax.in"},
            {"name": "Nexus Cable & Wire Co.", "phone": "+91 97110 54321", "email": "sales@nexuscables.com"},
            {"name": "Prime Office & Ergonomics", "phone": "+91 98765 43210", "email": "support@primeoffice.com"},
            {"name": "FreshHarvest Foods Pvt Ltd", "phone": "+91 99887 76655", "email": "distribution@freshharvest.in"},
            {"name": "Apex Apparel & Textiles", "phone": "+91 91234 56789", "email": "b2b@apexapparel.com"},
        ]
        created_suppliers = []
        for s in suppliers_data:
            supp = Supplier(name=s["name"], phone=s["phone"], email=s["email"])
            db.session.add(supp)
            created_suppliers.append(supp)
        db.session.commit()

        products_data = [
            {"name": "Wireless Ergonomic Mouse", "category": "Electronics", "price": 550.0, "current_stock": 35, "reorder_level": 20, "supplier_idx": 0,
             "planning": {"unit_cost": 350.0, "avg_daily_demand": 8.0, "demand_stddev": 2.5, "lead_time_days": 2, "ordering_cost": 100.0, "holding_cost_per_unit_year": 120.0, "service_level": 0.95, "minimum_order_qty": 10}},
            {"name": "Studio USB Microphone", "category": "Audio", "price": 1400.0, "current_stock": 18, "reorder_level": 15, "supplier_idx": 1,
             "planning": {"unit_cost": 950.0, "avg_daily_demand": 4.0, "demand_stddev": 1.8, "lead_time_days": 3, "ordering_cost": 120.0, "holding_cost_per_unit_year": 180.0, "service_level": 0.95, "minimum_order_qty": 5}},
            {"name": "27-inch 4K Monitor", "category": "Displays", "price": 8400.0, "current_stock": 12, "reorder_level": 10, "supplier_idx": 0,
             "planning": {"unit_cost": 6500.0, "avg_daily_demand": 3.0, "demand_stddev": 1.2, "lead_time_days": 4, "ordering_cost": 220.0, "holding_cost_per_unit_year": 650.0, "service_level": 0.95, "minimum_order_qty": 3}},
            {"name": "Mechanical Gaming Keyboard", "category": "Peripherals", "price": 750.0, "current_stock": 25, "reorder_level": 15, "supplier_idx": 0,
             "planning": {"unit_cost": 500.0, "avg_daily_demand": 6.0, "demand_stddev": 2.2, "lead_time_days": 2, "ordering_cost": 110.0, "holding_cost_per_unit_year": 130.0, "service_level": 0.95, "minimum_order_qty": 10}},
            {"name": "Fast-Charge USB-C Cable", "category": "Accessories", "price": 180.0, "current_stock": 42, "reorder_level": 25, "supplier_idx": 2,
             "planning": {"unit_cost": 90.0, "avg_daily_demand": 12.0, "demand_stddev": 3.5, "lead_time_days": 2, "ordering_cost": 80.0, "holding_cost_per_unit_year": 40.0, "service_level": 0.95, "minimum_order_qty": 20}},
            {"name": "Aluminum Laptop Stand", "category": "Office", "price": 540.0, "current_stock": 16, "reorder_level": 12, "supplier_idx": 3,
             "planning": {"unit_cost": 320.0, "avg_daily_demand": 3.0, "demand_stddev": 1.1, "lead_time_days": 3, "ordering_cost": 100.0, "holding_cost_per_unit_year": 95.0, "service_level": 0.95, "minimum_order_qty": 5}},
            {"name": "Wireless ANC Headphones", "category": "Audio", "price": 1150.0, "current_stock": 8, "reorder_level": 15, "supplier_idx": 1,
             "planning": {"unit_cost": 750.0, "avg_daily_demand": 5.0, "demand_stddev": 1.9, "lead_time_days": 3, "ordering_cost": 130.0, "holding_cost_per_unit_year": 160.0, "service_level": 0.95, "minimum_order_qty": 5}},
            {"name": "1080p Ultra HD Webcam", "category": "Peripherals", "price": 1750.0, "current_stock": 14, "reorder_level": 10, "supplier_idx": 0,
             "planning": {"unit_cost": 1250.0, "avg_daily_demand": 2.0, "demand_stddev": 1.0, "lead_time_days": 4, "ordering_cost": 150.0, "holding_cost_per_unit_year": 210.0, "service_level": 0.95, "minimum_order_qty": 4}},
            {"name": "20000mAh Fast Power Bank", "category": "Accessories", "price": 1250.0, "current_stock": 19, "reorder_level": 15, "supplier_idx": 2,
             "planning": {"unit_cost": 850.0, "avg_daily_demand": 5.0, "demand_stddev": 2.0, "lead_time_days": 3, "ordering_cost": 110.0, "holding_cost_per_unit_year": 170.0, "service_level": 0.95, "minimum_order_qty": 5}},
            {"name": "8K Braided HDMI 2.1 Cable", "category": "Accessories", "price": 260.0, "current_stock": 0, "reorder_level": 18, "supplier_idx": 2,
             "planning": {"unit_cost": 140.0, "avg_daily_demand": 7.0, "demand_stddev": 2.1, "lead_time_days": 2, "ordering_cost": 80.0, "holding_cost_per_unit_year": 50.0, "service_level": 0.95, "minimum_order_qty": 10}},
            {"name": "Insulated Stainless Steel Bottle", "category": "General", "price": 449.0, "current_stock": 30, "reorder_level": 20, "supplier_idx": 3,
             "planning": {"unit_cost": 240.0, "avg_daily_demand": 9.0, "demand_stddev": 2.6, "lead_time_days": 2, "ordering_cost": 100.0, "holding_cost_per_unit_year": 80.0, "service_level": 0.95, "minimum_order_qty": 12}},
            {"name": "Classic Cotton Crewneck T-Shirt", "category": "Apparel", "price": 499.0, "current_stock": 45, "reorder_level": 30, "supplier_idx": 5,
             "planning": {"unit_cost": 220.0, "avg_daily_demand": 12.0, "demand_stddev": 3.6, "lead_time_days": 3, "ordering_cost": 120.0, "holding_cost_per_unit_year": 80.0, "service_level": 0.95, "minimum_order_qty": 20}},
        ]

        for p in products_data:
            supp = created_suppliers[p["supplier_idx"]]
            prod = Product(
                name=p["name"],
                category=p["category"],
                price=p["price"],
                current_stock=p["current_stock"],
                reorder_level=p["reorder_level"],
                supplier_id=supp.id,
            )
            db.session.add(prod)
            db.session.flush()

            pl = p["planning"]
            planning = ProductPlanning(
                product_id=prod.id,
                unit_cost=pl["unit_cost"],
                avg_daily_demand=pl["avg_daily_demand"],
                demand_stddev=pl["demand_stddev"],
                lead_time_days=pl["lead_time_days"],
                ordering_cost=pl["ordering_cost"],
                holding_cost_per_unit_year=pl["holding_cost_per_unit_year"],
                service_level=pl["service_level"],
                minimum_order_qty=pl["minimum_order_qty"],
            )
            db.session.add(planning)

            if prod.current_stock > 0:
                tx = Transaction(
                    product_id=prod.id,
                    transaction_type="IN",
                    quantity=prod.current_stock,
                )
                db.session.add(tx)

        db.session.commit()


@app.route("/seed-demo-data", methods=["GET", "POST"])
def seed_demo_data():
    """Manual endpoint to seed or restore example products and suppliers."""
    seed_initial_data()
    return redirect(url_for("dashboard"))


# Create any missing tables on startup without dropping or replacing existing tables/data.
os.makedirs(app.instance_path, exist_ok=True)
with app.app_context():
    db.create_all()
    seed_initial_data()

if __name__ == "__main__":
    app.run(debug=False)
