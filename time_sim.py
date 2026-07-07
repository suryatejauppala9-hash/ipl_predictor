import time
import main

t0 = time.time()
bat1, bowl1 = main._best_xi_names("Chennai Super Kings", "balanced")
bat2, bowl2 = main._best_xi_names("Mumbai Indians", "balanced")

main._run_sim(bat1, bowl1, bat2, bowl2, 500, "Chennai Super Kings", "Mumbai Indians")
t1 = time.time()
print(f"500 simulations took: {t1-t0:.2f} seconds")
