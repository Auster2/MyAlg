# test epsl_run.py
from problem import CSMOP1
from alg.epsl_run import EPSLRunner

epsl_runner = EPSLRunner(problem=CSMOP1(), hv_value_true=0.87661645, n_steps=10, n_sample=5, n_pref_update=2, device='cpu', n_run=1)
def test_epsl_run():
    result = epsl_runner.run_once()
    print("Test passed successfully!")

if __name__ == "__main__":
    test_epsl_run()
    print("EPSLRunner test completed.")