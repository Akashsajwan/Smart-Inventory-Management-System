import copy
import unittest

from simulation_engine import run_simulation


def sample_product(**overrides):
    product = {
        "id": 1,
        "name": "Test Product",
        "initial_stock": 20,
        "avg_daily_demand": 4,
        "demand_stddev": 1,
        "lead_time_days": 3,
        "service_level": 0.95,
        "ordering_cost": 50,
        "holding_cost_per_unit_year": 100,
        "unit_cost": 10,
        "selling_price": 20,
        "minimum_order_qty": 2,
        "reorder_level": 10,
    }
    product.update(overrides)
    return product


class SimulationEngineTests(unittest.TestCase):

    def test_simulation_is_reproducible(self):
        products = [sample_product()]

        first = run_simulation(
            products,
            duration_days=30,
            policy="reorder_point",
            random_seed=42,
        )
        second = run_simulation(
            products,
            duration_days=30,
            policy="reorder_point",
            random_seed=42,
        )

        self.assertEqual(first, second)

    def test_input_products_are_not_modified(self):
        products = [sample_product()]
        original = copy.deepcopy(products)

        run_simulation(
            products,
            duration_days=10,
            policy="fixed_threshold",
            random_seed=42,
        )

        self.assertEqual(products, original)

    def test_both_policies_return_valid_results(self):
        for policy in ("fixed_threshold", "reorder_point"):
            with self.subTest(policy=policy):
                result = run_simulation(
                    [sample_product()],
                    duration_days=10,
                    policy=policy,
                    random_seed=42,
                )

                self.assertEqual(result["policy"], policy)
                self.assertEqual(result["duration_days"], 10)
                self.assertIn("summary", result)
                self.assertIn("daily", result)
                self.assertIn("events", result)
                self.assertEqual(len(result["daily"]), 10)

    def test_zero_day_delivery_is_received_immediately(self):
        product = sample_product(
            initial_stock=0,
            avg_daily_demand=0,
            demand_stddev=0,
            lead_time_days=0,
            reorder_level=0,
            minimum_order_qty=5,
        )

        result = run_simulation(
            [product],
            duration_days=1,
            policy="fixed_threshold",
            random_seed=42,
        )

        item = result["products"][0]

        self.assertEqual(item["final_stock"], 5)
        self.assertEqual(item["outstanding_quantity"], 0)

        event_types = [event["event_type"] for event in result["events"]]
        self.assertLess(
            event_types.index("ORDER_PLACED"),
            event_types.index("DELIVERY_RECEIVED"),
        )

    def test_stock_conservation(self):
        result = run_simulation(
            [sample_product()],
            duration_days=30,
            policy="reorder_point",
            random_seed=42,
        )

        item = result["products"][0]

        received = sum(
            event["quantity"]
            for event in result["events"]
            if event["event_type"] == "DELIVERY_RECEIVED"
        )

        sold = sum(
            event["quantity"]
            for event in result["events"]
            if event["event_type"] == "SALE"
        )

        expected_final_stock = 20 + received - sold
        self.assertEqual(item["final_stock"], expected_final_stock)

    def test_ordering_and_purchase_costs(self):
        product = sample_product()

        result = run_simulation(
            [product],
            duration_days=30,
            policy="reorder_point",
            random_seed=42,
        )

        item = result["products"][0]
        summary = result["summary"]

        self.assertEqual(
            item["ordering_cost"],
            item["orders_placed"] * product["ordering_cost"],
        )

        placed_quantity = sum(
            event["quantity"]
            for event in result["events"]
            if event["event_type"] == "ORDER_PLACED"
        )

        self.assertEqual(
            item["purchase_cost"],
            placed_quantity * product["unit_cost"],
        )

        self.assertEqual(
            summary["total_ordering_cost"],
            item["ordering_cost"],
        )

        self.assertEqual(
            summary["total_purchase_cost"],
            item["purchase_cost"],
        )

        self.assertGreaterEqual(item["holding_cost"], 0)
        self.assertEqual(
            summary["total_holding_cost"],
            item["holding_cost"],
        )

    def test_service_level_matches_fulfilled_demand(self):
        result = run_simulation(
            [sample_product()],
            duration_days=30,
            policy="fixed_threshold",
            random_seed=42,
        )

        summary = result["summary"]

        if summary["total_demand"]:
            expected = round(
                summary["total_units_sold"] / summary["total_demand"],
                4,
            )
        else:
            expected = 1.0

        self.assertEqual(summary["service_level"], expected)

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            run_simulation(
                [sample_product()],
                duration_days=10,
                policy="invalid_policy",
                random_seed=42,
            )

    def test_invalid_duration_is_rejected(self):
        with self.assertRaises(ValueError):
            run_simulation(
                [sample_product()],
                duration_days=0,
                policy="fixed_threshold",
                random_seed=42,
            )


    def test_historical_demand_replays_exact_sequence(self):
        demand = [3, 0, 5, 2]
        product = sample_product(
            initial_stock=100,
            avg_daily_demand=2,
            demand_stddev=0,
            lead_time_days=3,
        )
        result = run_simulation(
            [product],
            duration_days=4,
            policy="fixed_threshold",
            demand_sequences={1: demand},
        )
        self.assertEqual(result["summary"]["total_demand"], sum(demand))
        self.assertEqual(result["summary"]["total_units_sold"], sum(demand))
        self.assertEqual(result["summary"]["total_lost_units"], 0)

    def test_historical_demand_requires_every_product(self):
        with self.assertRaises(ValueError):
            run_simulation(
                [sample_product()],
                duration_days=3,
                policy="fixed_threshold",
                demand_sequences={},
            )

    def test_historical_demand_requires_enough_days(self):
        with self.assertRaises(ValueError):
            run_simulation(
                [sample_product()],
                duration_days=3,
                policy="fixed_threshold",
                demand_sequences={1: [1, 2]},
            )

    def test_historical_demand_preserves_stock_conservation(self):
        product = sample_product(initial_stock=10)
        demand = [8, 7, 2, 5, 1]
        result = run_simulation(
            [product],
            duration_days=len(demand),
            policy="reorder_point",
            demand_sequences={1: demand},
        )
        item = result["products"][0]
        received = sum(
            event["quantity"]
            for event in result["events"]
            if event["event_type"] == "DELIVERY_RECEIVED"
        )
        sold = sum(
            event["quantity"]
            for event in result["events"]
            if event["event_type"] == "SALE"
        )
        self.assertEqual(item["final_stock"], 10 + received - sold)
        self.assertEqual(result["summary"]["total_demand"], sum(demand))


if __name__ == "__main__":
    unittest.main()

