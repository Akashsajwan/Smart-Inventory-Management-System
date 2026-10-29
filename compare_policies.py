
from simulation_engine import run_simulation

products = [
    {
        "id": 1,
        "name": "Wireless Mouse",
        "initial_stock": 30,
        "avg_daily_demand": 5,
        "demand_stddev": 2,
        "lead_time_days": 3,
        "service_level": 0.95,
        "ordering_cost": 50,
        "holding_cost_per_unit_year": 10,
        "unit_cost": 500,
        "selling_price": 899,
        "minimum_order_qty": 1,
        "reorder_level": 10,
    },
    {
        "id": 2,
        "name": "Microphones",
        "initial_stock": 5,
        "avg_daily_demand": 4,
        "demand_stddev": 1.5,
        "lead_time_days": 5,
        "service_level": 0.95,
        "ordering_cost": 50,
        "holding_cost_per_unit_year": 100,
        "unit_cost": 700,
        "selling_price": 1299,
        "minimum_order_qty": 2,
        "reorder_level": 10,
    },
]

for policy in ("fixed_threshold", "reorder_point"):
    result = run_simulation(
        products,
        duration_days=30,
        policy=policy,
        random_seed=42,
    )

    summary = result["summary"]

    print(f"\n--- {policy.upper()} ---")
    print("Units sold:", summary["total_units_sold"])
    print("Lost units:", summary["total_lost_units"])
    print("Service level:", f'{summary["service_level"]:.2%}')
    print("Revenue:", f'₹{summary["total_revenue"]:,.2f}')
    print("Total purchase cost:", f'₹{summary["total_purchase_cost"]:,.2f}')
    print("Total ordering cost:", f'₹{summary["total_ordering_cost"]:,.2f}')
    print("Total holding cost:", f'₹{summary["total_holding_cost"]:,.2f}')

    for item in result["products"]:
        print(
            f'{item["product_name"]}: '
            f'{item["orders_placed"]} orders, '
            f'{item["stockout_days"]} stockout days, '
            f'holding cost ₹{item["holding_cost"]:,.2f}'
        )

