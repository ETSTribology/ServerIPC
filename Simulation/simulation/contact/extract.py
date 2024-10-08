def save_to_csv(data, filename='collision_data.csv'):
    # Define the header and write the data to the CSV file
    header = ['Collision Type', 'Tangential Velocity', 'Relative Velocity', 'Weights', 'Coefficients', 'Normal Force Magnitude', 'f0_SF', 'Friction']

    # Save the data to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

collision_data = []

def handle_collisions(fconstraints, cmesh, BX, BXdot, epsv, EF):
    global collision_data
    # Handle vertex-vertex collisions
    for i, collision in enumerate(fconstraints.vv_collisions):
        tangent_basis = collision.compute_tangent_basis(BX)
        relative_velocity = collision.relative_velocity(BXdot)
        tangent_rel_velocity = np.dot(tangent_basis.transpose(), relative_velocity)
        f0_SF_value = ipctk.f0_SF(np.linalg.norm(tangent_rel_velocity), epsv)
        
        # Append data to the list
        collision_data.append(["Vertex-Vertex", tangent_rel_velocity, relative_velocity,
                               collision.weight[i], collision.mu[i], collision.normal_force_magnitude[i], f0_SF_value, EF])

    # Handle edge-vertex collisions
    for i, collision in enumerate(fconstraints.ev_collisions):
        tangent_basis = collision.compute_tangent_basis(BX)
        relative_velocity = collision.relative_velocity(BXdot)
        tangent_rel_velocity = np.dot(tangent_basis.transpose(), relative_velocity)
        f0_SF_value = ipctk.f0_SF(np.linalg.norm(tangent_rel_velocity), epsv)

        # Append data to the list
        collision_data.append(["Edge-Vertex", tangent_rel_velocity, relative_velocity,
                               collision.weight[i], collision.mu[i], collision.normal_force_magnitude[i], f0_SF_value, EF])

    # Handle edge-edge collisions
    for i, collision in enumerate(fconstraints.ee_collisions):
        edge0_vertices = cmesh.edges[collision.edge0_id]
        edge1_vertices = cmesh.edges[collision.edge1_id]
        tangent_rel_velocity = None
        relative_velocity = None
        # No need for velocity computation in the script for edge-edge

        # Append data to the list
        collision_data.append(["Edge-Edge", tangent_rel_velocity, relative_velocity,
                               collision.weight, collision.mu, collision.normal_force_magnitude, None, EF])

    # Handle face-vertex collisions
    for i, collision in enumerate(fconstraints.fv_collisions):
        # No need for velocity computation in the script for face-vertex
        tangent_rel_velocity = None
        relative_velocity = None
        
        # Append data to the list
        collision_data.append(["Face-Vertex", tangent_rel_velocity, relative_velocity,
                               collision.weight, collision.mu, None, None, EF])
        
    # Save the collision data to a CSV file
    save_to_csv(collision_data)