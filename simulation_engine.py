"""Isolated, reproducible inventory simulation engine."""

import random

from inventory_logic import economic_order_quantity, reorder_point


def run_simulation(
    products,
    duration_days=30,
    policy="reorder_point",
    random_seed=42,
    demand_sequences=None,
):
    """Simulate inventory operations without changing real inventory.

    Each product dictionary must contain:
      id, name, initial_stock, avg_daily_demand, demand_stddev,
      lead_time_days, service_level, ordering_cost,
      holding_cost_per_unit_year, unit_cost, minimum_order_qty,
      and reorder_level. ``selling_price`` is optional and defaults to 0.
    """
    if duration_days < 1:
        raise ValueError("Simulation duration must be at least 1 day.")

    if policy not in ("reorder_point", "fixed_threshold"):
        raise ValueError("Unsupported replenishment policy.")

    rng = random.Random(random_seed)
    states = {}
    events = []
    daily_results = []
    next_order_id = 1

    # Build private simulation state; never mutate input dictionaries.
    for product in products:
        product_id = product["id"]

        if product_id in states:
            raise ValueError(f"Duplicate product id: {product_id}")

        stock = int(product["initial_stock"])
        if stock < 0:
            raise ValueError("Initial stock cannot be negative.")

        lead_time = int(product["lead_time_days"])
        if lead_time < 0:
            raise ValueError("Lead time cannot be negative.")

        mean = float(product["avg_daily_demand"])
        stddev = float(product["demand_stddev"])
        if mean < 0 or stddev < 0:
            raise ValueError("Demand values cannot be negative.")

        if float(product["ordering_cost"]) < 0:
            raise ValueError("Ordering cost cannot be negative.")
        if float(product["holding_cost_per_unit_year"]) <= 0:
            raise ValueError("Annual holding cost must be greater than zero.")
        if int(product["minimum_order_qty"]) < 1:
            raise ValueError("Minimum order quantity must be at least 1.")
        if float(product["unit_cost"]) < 0:
            raise ValueError("Unit cost cannot be negative.")

        states[product_id] = {
            "config": dict(product),
            "stock": stock,
            "on_order": 0,
            "pending_deliveries": [],
            "units_sold": 0,
            "lost_units": 0,
            "stockout_days": 0,
            "revenue": 0.0,
            "purchase_cost": 0.0,
            "ordering_cost_total": 0.0,
            "holding_cost_total": 0.0,
            "orders_placed": 0,
        }

    if demand_sequences is not None:
        for product_id in states:
            if product_id not in demand_sequences:
                raise ValueError(
                    f"Missing historical demand for product {product_id}."
                )

            sequence = demand_sequences[product_id]
            if len(sequence) < duration_days:
                raise ValueError(
                    f"Historical demand for product {product_id} "
                    f"must contain at least {duration_days} days."
                )

            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in sequence[:duration_days]
            ):
                raise ValueError(
                    f"Historical demand for product {product_id} "
                    "must contain non-negative integers."
                )

    def record(day, product_id, event_type, quantity, before, after, notes=""):
        events.append(
            {
                "day": day,
                "product_id": product_id,
                "event_type": event_type,
                "quantity": quantity,
                "stock_before": before,
                "stock_after": after,
                "notes": notes,
            }
        )

    daily_item_results = []

    for day in range(1, duration_days + 1):
        day_sold = 0
        day_lost = 0
        day_revenue = 0.0
        opening_stock = {
            product_id: state["stock"]
            for product_id, state in states.items()
        }
        day_demand = {}
        day_item_sold = {}
        day_item_lost = {}
        day_deliveries = {product_id: 0 for product_id in states}
        day_orders = {product_id: 0 for product_id in states}

        # 1. Receive deliveries scheduled for this day.
        for product_id, state in states.items():
            arriving = [
                order
                for order in state["pending_deliveries"]
                if order["arrival_day"] == day
            ]

            for order in arriving:
                before = state["stock"]
                quantity = order["quantity"]
                state["stock"] += quantity
                state["on_order"] -= quantity
                day_deliveries[product_id] += quantity

                record(
                    day,
                    product_id,
                    "DELIVERY_RECEIVED",
                    quantity,
                    before,
                    state["stock"],
                    f"Simulated order #{order['order_id']}",
                )

            state["pending_deliveries"] = [
                order
                for order in state["pending_deliveries"]
                if order["arrival_day"] != day
            ]

        # 2. Generate demand and fulfill sales.
        for product_id, state in states.items():
            config = state["config"]
            mean = float(config["avg_daily_demand"])
            stddev = float(config["demand_stddev"])

            # Use historical demand when supplied; otherwise generate demand.
            if demand_sequences is not None:
                demand = int(demand_sequences[product_id][day - 1])
            else:
                demand = max(0, round(rng.gauss(mean, stddev)))
            before = state["stock"]
            sold = min(before, demand)
            lost = demand - sold
            day_demand[product_id] = demand
            day_item_sold[product_id] = sold
            day_item_lost[product_id] = lost

            state["stock"] -= sold
            state["units_sold"] += sold
            state["lost_units"] += lost
            selling_price = float(config.get("selling_price", 0))
            sale_revenue = sold * selling_price
            state["revenue"] += sale_revenue

            day_sold += sold
            day_lost += lost
            day_revenue += sale_revenue

            if sold:
                record(
                    day,
                    product_id,
                    "SALE",
                    sold,
                    before,
                    state["stock"],
                )

            if lost:
                record(
                    day,
                    product_id,
                    "LOST_SALE",
                    lost,
                    state["stock"],
                    state["stock"],
                    "Demand could not be fulfilled",
                )
                state["stockout_days"] += 1

        # 3. Evaluate replenishment after today's demand.
        for product_id, state in states.items():
            config = state["config"]
            position = state["stock"] + state["on_order"]

            if policy == "reorder_point":
                threshold = reorder_point(
                    float(config["avg_daily_demand"]),
                    int(config["lead_time_days"]),
                    float(config["demand_stddev"]),
                    float(config["service_level"]),
                )
            else:
                threshold = int(config["reorder_level"])

            if position > threshold:
                continue

            if policy == "reorder_point":
                annual_demand = float(config["avg_daily_demand"]) * 365
                quantity = economic_order_quantity(
                    annual_demand,
                    float(config["ordering_cost"]),
                    float(config["holding_cost_per_unit_year"]),
                )
                quantity = max(int(config["minimum_order_qty"]), quantity)
            else:
                # Fixed Threshold policy uses static fixed batch ordering rather than dynamic EOQ
                fixed_batch = int(config.get("fixed_order_qty", int(config["reorder_level"]) * 2))
                quantity = max(int(config["minimum_order_qty"]), fixed_batch)

            if quantity <= 0:
                continue

            order_id = next_order_id
            next_order_id += 1
            arrival_day = day + int(config["lead_time_days"])

            state["on_order"] += quantity
            state["orders_placed"] += 1
            day_orders[product_id] += quantity
            state["purchase_cost"] += quantity * float(config["unit_cost"])
            # Ordering cost is a fixed cost per order, not per unit.
            state["ordering_cost_total"] += float(config["ordering_cost"])

            # Record the order before recording a possible same-day receipt.
            record(
                day,
                product_id,
                "ORDER_PLACED",
                quantity,
                state["stock"],
                state["stock"],
                f"Order #{order_id}; arrival day {arrival_day}",
            )

            if arrival_day <= day:
                before = state["stock"]
                state["stock"] += quantity
                state["on_order"] -= quantity
                day_deliveries[product_id] += quantity
                record(
                    day,
                    product_id,
                    "DELIVERY_RECEIVED",
                    quantity,
                    before,
                    state["stock"],
                    f"Simulated order #{order_id}; same-day delivery",
                )
            else:
                state["pending_deliveries"].append(
                    {
                        "order_id": order_id,
                        "quantity": quantity,
                        "arrival_day": arrival_day,
                    }
                )

        # Charge holding cost against closing on-hand inventory for this day.
        for state in states.values():
            annual_holding_cost = float(
                state["config"]["holding_cost_per_unit_year"]
            )
            state["holding_cost_total"] += (
                state["stock"] * annual_holding_cost / 365
            )

        daily_results.append(
            {
                "day": day,
                "units_sold": day_sold,
                "lost_units": day_lost,
                "revenue": round(day_revenue, 2),
            }
        )

        for product_id, state in states.items():
            config = state["config"]
            daily_item_results.append(
                {
                    "day": day,
                    "product_id": product_id,
                    "product_name": config["name"],
                    "category": config.get("category", ""),
                    "supplier": config.get("supplier", "Not specified"),
                    "opening_stock": opening_stock[product_id],
                    "demand": day_demand.get(product_id, 0),
                    "units_sold": day_item_sold.get(product_id, 0),
                    "lost_units": day_item_lost.get(product_id, 0),
                    "deliveries_received": day_deliveries[product_id],
                    "units_ordered": day_orders[product_id],
                    "closing_stock": state["stock"],
                    "on_order": state["on_order"],
                    "reorder_level": int(config["reorder_level"]),
                    "lead_time_days": int(config["lead_time_days"]),
                    "inventory_value": round(
                        state["stock"] * float(config["unit_cost"]), 2
                    ),
                }
            )

    product_results = []
    for product_id, state in states.items():
        config = state["config"]
        product_results.append(
            {
                "product_id": product_id,
                "product_name": config["name"],
                "final_stock": state["stock"],
                "outstanding_quantity": state["on_order"],
                "units_sold": state["units_sold"],
                "lost_units": state["lost_units"],
                "stockout_days": state["stockout_days"],
                "orders_placed": state["orders_placed"],
                "revenue": round(state["revenue"], 2),
                "purchase_cost": round(state["purchase_cost"], 2),
                "ordering_cost": round(state["ordering_cost_total"], 2),
                "holding_cost": round(state["holding_cost_total"], 2),
            }
        )

    total_demand = sum(
        item["units_sold"] + item["lost_units"] for item in product_results
    )
    total_sold = sum(item["units_sold"] for item in product_results)
    total_lost = sum(item["lost_units"] for item in product_results)

    return {
        "duration_days": duration_days,
        "policy": policy,
        "random_seed": random_seed,
        "products": product_results,
        "daily": daily_results,
        "daily_items": daily_item_results,
        "events": events,
        "summary": {
            "total_units_sold": total_sold,
            "total_lost_units": total_lost,
            "total_demand": total_demand,
            "service_level": round(total_sold / total_demand, 4)
            if total_demand
            else 1.0,
            "total_revenue": round(
                sum(item["revenue"] for item in product_results), 2
            ),
            "total_purchase_cost": round(
                sum(item["purchase_cost"] for item in product_results), 2
            ),
            "total_ordering_cost": round(
                sum(item["ordering_cost"] for item in product_results), 2
            ),
            "total_holding_cost": round(
                sum(item["holding_cost"] for item in product_results), 2
            ),
        },
    }
