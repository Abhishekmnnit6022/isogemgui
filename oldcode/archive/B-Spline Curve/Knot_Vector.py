import numpy as np
degree = 11
control_points = 29


# Create a list to store the knot vector
knot_vector = []

# Generate the non-uniform knot values
knot_values = [0.0] * degree + [i * 0.1 for i in range(1, control_points)] + [1.0] * degree

print(len(knot_values))
# Populate the knot vector
for knot in knot_values:
    knot_vector.append(knot)

# Print the knot vector
print(knot_vector)
print(type(knot_vector[0]))
knot_vector_array = np.array(knot_vector)
print(knot_vector_array)
