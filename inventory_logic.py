"""Inventory planning calculations for Smart Warehouse."""

from math import ceil, sqrt
from statistics import NormalDist


def safety_stock(demand_stddev, lead_time_days, service_level=0.95):
    """Calculate safety stock for variable daily demand and fixed lead time."""
    if demand_stddev < 0:
        raise ValueError("Demand standard deviation cannot be negative.")
    if lead_time_days < 0:
        raise ValueError("Lead time cannot be negative.")
    if not 0 < service_level < 1:
        raise ValueError("Service level must be between 0 and 1.")

    z_score = NormalDist().inv_cdf(service_level)
    return ceil(z_score * demand_stddev * sqrt(lead_time_days))


def reorder_point(avg_daily_demand, lead_time_days, demand_stddev,
                  service_level=0.95):
    """Calculate the stock position at which replenishment is needed."""
    if avg_daily_demand < 0:
        raise ValueError("Average daily demand cannot be negative.")
    if lead_time_days < 0:
        raise ValueError("Lead time cannot be negative.")

    buffer_stock = safety_stock(
        demand_stddev, lead_time_days, service_level
    )

    expected_lead_time_demand = avg_daily_demand * lead_time_days

    return ceil(expected_lead_time_demand + buffer_stock)


def economic_order_quantity(annual_demand, ordering_cost,
                            annual_holding_cost_per_unit):
    """Calculate EOQ using the classic inventory model."""
    if annual_demand < 0:
        raise ValueError("Annual demand cannot be negative.")
    if ordering_cost < 0:
        raise ValueError("Ordering cost cannot be negative.")
    if annual_holding_cost_per_unit <= 0:
        raise ValueError("Annual holding cost must be greater than zero.")

    quantity = sqrt(
        (2 * annual_demand * ordering_cost)
        / annual_holding_cost_per_unit
    )

    return ceil(quantity)


def inventory_position(on_hand, on_order=0, backorders=0):
    """Calculate available inventory position for replenishment decisions."""
    if min(on_hand, on_order, backorders) < 0:
        raise ValueError("Inventory quantities cannot be negative.")

    return on_hand + on_order - backorders
