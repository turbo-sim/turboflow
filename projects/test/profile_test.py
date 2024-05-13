import cProfile
import pstats
import run_test  # assuming your code is in run_test.py

profiler = cProfile.Profile()
profiler.enable()

# Call your code here
run_test()  # Assuming there's a main function in run_test.py

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')  # You can change the sorting method as needed
stats.print_stats()