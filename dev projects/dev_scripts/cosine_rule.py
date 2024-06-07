import numpy as np

# Modified formula the include the effect of exit wedge angle
theta = np.asarray(
    [65 * np.pi / 180, -61.6 * np.pi / 180, 55.2 * np.pi / 180, -48.9 * np.pi / 180]
)
eps = np.asarray([5, 5, 5, 5]) * np.pi / 180
cosine_rule = np.cos(theta)
cosine_modified = np.cos(theta) * (1 - np.tan(np.abs(theta)) * np.tan(eps / 2))
print("Effect of exit wedge angle")
print(f"{'Cosine rule':>12s} {'Modified':>12s} {'Ratio (%)':>12s}")
for o1, o2 in zip(cosine_rule, cosine_modified):
    print(f"{o1:12.4f} {o2:12.4f} {o2/o1*100:12.4f}")

# Modified formula the include the effect of trailing edge thickness
t_te = np.asarray([2 * 0.025e-2, 2 * 0.025e-2, 2 * 0.025e-2, 2 * 0.025e-2])
o = np.asarray([0.7731e-2, 0.7249e-2, 0.8469e-2, 0.9538e-2])
print("\nEffect of trailing edge thickness")
print(f"{'Opening':>12s} {'Thickness':>12s} {'Ratio (%)':>12s}")
for _t_te, _o in zip(t_te, o):
    print(f"{_o:12.4f} {_t_te:12.4f} {_t_te/_o*100:12.4f}")
