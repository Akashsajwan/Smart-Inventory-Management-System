import unittest
from simulation_engine import run_simulation
from warehouse_demo import DEMO_PRODUCTS, DEMO_PRODUCTS_BY_STORE, get_demo_products


class SevenDayProjectTests(unittest.TestCase):
    def test_demo_dataset_has_ten_products(self):
        self.assertEqual(len(DEMO_PRODUCTS), 10)

    def test_all_store_catalogs_have_ten_products(self):
        for store in ("electronics", "grocery", "clothing", "general"):
            catalog = get_demo_products(store)
            self.assertEqual(len(catalog), 10, f"Store '{store}' must contain exactly 10 products")

    def test_both_policies_produce_exactly_seven_daily_records(self):
        for policy in ("fixed_threshold", "reorder_point"):
            result = run_simulation(
                [dict(product) for product in DEMO_PRODUCTS],
                duration_days=7,
                policy=policy,
                random_seed=42,
            )
            self.assertEqual(result["duration_days"], 7)
            self.assertEqual([row["day"] for row in result["daily"]], list(range(1, 8)))

    def test_policy_comparison_uses_same_demand_seed(self):
        fixed = run_simulation(
            [dict(product) for product in DEMO_PRODUCTS],
            duration_days=7,
            policy="fixed_threshold",
            random_seed=42,
        )
        reorder = run_simulation(
            [dict(product) for product in DEMO_PRODUCTS],
            duration_days=7,
            policy="reorder_point",
            random_seed=42,
        )
        self.assertEqual(fixed["random_seed"], reorder["random_seed"])
        self.assertEqual(fixed["random_seed"], 42)
        self.assertEqual(fixed["duration_days"], reorder["duration_days"])
        self.assertEqual(fixed["duration_days"], 7)


if __name__ == "__main__":
    unittest.main()

