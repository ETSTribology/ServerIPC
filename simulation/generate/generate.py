import os

import ipywidgets as widgets
import meshio
import numpy as np
import plotly.graph_objects as go
import tetgen as tg
import trimesh
from IPython.display import display
from noise import pnoise3
from PIL import Image

from simulation.generate.noise import NoiseFunction

class HeightmapMeshGenerator:
    def __init__(self, noise_function: NoiseFunction, size_x=100, size_y=100, amplitude=1.0, generate_displacement_map=False, **kwargs):
        """
        Initialize the generator with given parameters.
        
        :param noise_function: An instance of a NoiseFunction subclass.
        :param size_x: Number of points along the X-axis.
        :param size_y: Number of points along the Y-axis.
        :param amplitude: Scaling factor for the heightmap.
        :param generate_displacement_map: Boolean to enable/disable displacement map generation.
        :param kwargs: Additional parameters.
        """
        self.size_x = size_x
        self.size_y = size_y
        self.amplitude = amplitude
        self.noise_function = noise_function
        self.generate_displacement_map = generate_displacement_map

        self.heightmap = None
        self.mesh_vertices = None
        self.mesh_faces = None
        self.tetrahedral_mesh = None

        # Attributes for normal map
        self.normals = None
        self.normal_map_scale = kwargs.get('normal_map_scale', 1)
        self.normal_map_resolution = kwargs.get('normal_map_resolution', (512, 512))

        # Attributes for displacement map
        self.displacement_map_scale = kwargs.get('displacement_map_scale', 1)
        self.displacement_map_resolution = kwargs.get('displacement_map_resolution', (512, 512))

        # Output folder setup
        self.output_folder = kwargs.get('output_folder', "output")
        os.makedirs(self.output_folder, exist_ok=True)

        # Visualization
        self.fig_widget = go.FigureWidget()

    def generate_heightmap(self):
        """Generate the heightmap using the selected noise function."""
        x = np.linspace(-1, 1, self.size_x)
        y = np.linspace(-1, 1, self.size_y)
        x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
        
        self.heightmap = self.noise_function.generate(x_grid, y_grid)
        self.heightmap = self.normalize_heightmap(self.heightmap) * self.amplitude

    def normalize_heightmap(self, heightmap):
        """Normalize the heightmap to range [0, 1]."""
        min_val = np.min(heightmap)
        max_val = np.max(heightmap)
        if max_val - min_val == 0:
            raise ValueError("Heightmap has no variation.")
        return (heightmap - min_val) / (max_val - min_val)

    def heightmap_to_closed_mesh(self):
        """Convert the heightmap to a closed 3D mesh."""
        size_x, size_y = self.heightmap.shape

        # Create grid of indices
        i = np.arange(size_x)
        j = np.arange(size_y)
        ii, jj = np.meshgrid(i, j, indexing='ij')

        # Flatten the grid indices
        ii_flat = ii.flatten()
        jj_flat = jj.flatten()
        height_flat = self.heightmap.flatten()

        # Generate top and bottom vertices
        top_vertices = np.column_stack((ii_flat, jj_flat, height_flat))
        bottom_vertices = np.column_stack((ii_flat, jj_flat, np.zeros_like(height_flat)))

        # Combine vertices
        vertices = np.vstack((top_vertices, bottom_vertices))

        # Index offset for bottom vertices
        offset = size_x * size_y

        # Create vertex indices grid
        vertex_indices = np.arange(size_x * size_y).reshape((size_x, size_y))
        vertex_indices_bottom = vertex_indices + offset

        # Indices for cells (quads)
        v0 = vertex_indices[:-1, :-1]
        v1 = vertex_indices[1:, :-1]
        v2 = vertex_indices[:-1, 1:]
        v3 = vertex_indices[1:, 1:]

        # Flatten the indices
        v0_flat = v0.flatten()
        v1_flat = v1.flatten()
        v2_flat = v2.flatten()
        v3_flat = v3.flatten()

        # Create faces for the top surface
        faces_top = np.vstack([
            np.column_stack([v0_flat, v1_flat, v2_flat]),
            np.column_stack([v1_flat, v3_flat, v2_flat])
        ])

        # Bottom surface faces (reverse the order to flip normals)
        v0b = vertex_indices_bottom[:-1, :-1]
        v1b = vertex_indices_bottom[1:, :-1]
        v2b = vertex_indices_bottom[:-1, 1:]
        v3b = vertex_indices_bottom[1:, 1:]

        v0b_flat = v0b.flatten()
        v1b_flat = v1b.flatten()
        v2b_flat = v2b.flatten()
        v3b_flat = v3b.flatten()

        faces_bottom = np.vstack([
            np.column_stack([v0b_flat, v2b_flat, v1b_flat]),
            np.column_stack([v1b_flat, v2b_flat, v3b_flat])
        ])

        # Side faces
        # Left side (j=0)
        v_top_left = vertex_indices[:, 0]
        v_bot_left = vertex_indices_bottom[:, 0]

        faces_left = np.vstack([
            np.column_stack([v_top_left[:-1], v_bot_left[:-1], v_bot_left[1:]]),
            np.column_stack([v_top_left[:-1], v_bot_left[1:], v_top_left[1:]])
        ])

        # Right side (j=size_y-1)
        v_top_right = vertex_indices[:, -1]
        v_bot_right = vertex_indices_bottom[:, -1]

        faces_right = np.vstack([
            np.column_stack([v_top_right[:-1], v_bot_right[1:], v_bot_right[:-1]]),
            np.column_stack([v_top_right[:-1], v_top_right[1:], v_bot_right[1:]])
        ])

        # Front side (i=0)
        v_top_front = vertex_indices[0, :]
        v_bot_front = vertex_indices_bottom[0, :]

        faces_front = np.vstack([
            np.column_stack([v_top_front[:-1], v_bot_front[:-1], v_bot_front[1:]]),
            np.column_stack([v_top_front[:-1], v_bot_front[1:], v_top_front[1:]])
        ])

        # Back side (i=size_x-1)
        v_top_back = vertex_indices[-1, :]
        v_bot_back = vertex_indices_bottom[-1, :]

        faces_back = np.vstack([
            np.column_stack([v_top_back[:-1], v_bot_back[1:], v_bot_back[:-1]]),
            np.column_stack([v_top_back[:-1], v_top_back[1:], v_bot_back[1:]])
        ])

        # Combine all faces
        faces = np.vstack((faces_top, faces_bottom, faces_left, faces_right, faces_front, faces_back))

        self.mesh_vertices = vertices
        self.mesh_faces = faces

    def repair_mesh(self):
        """Repair the mesh if it's not watertight."""
        mesh = trimesh.Trimesh(vertices=self.mesh_vertices, faces=self.mesh_faces)
        if not mesh.is_watertight:
            print("Mesh is not watertight. Attempting to repair...")
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fix_inversion(mesh)
            mesh.remove_duplicate_faces()
            mesh.remove_duplicate_vertices()
            mesh.remove_unreferenced_vertices()

            if mesh.is_watertight:
                print("Mesh successfully repaired and is now watertight.")
            else:
                print("Warning: Mesh is still not watertight after repair.")
        else:
            print("Mesh is already watertight.")

        self.mesh_vertices = mesh.vertices
        self.mesh_faces = mesh.faces

    def simplify_mesh(self, target_face_count):
        """Simplify the mesh to a target number of faces."""
        mesh = trimesh.Trimesh(vertices=self.mesh_vertices, faces=self.mesh_faces)
        simplified_mesh = mesh.simplify_quadratic_decimation(target_face_count)
        self.mesh_vertices = simplified_mesh.vertices
        self.mesh_faces = simplified_mesh.faces
        print(f"Simplified mesh to {len(self.mesh_faces)} faces.")

    def save_mesh_to_stl(self, filename):
        """Save the mesh to an STL file."""
        meshio_mesh = meshio.Mesh(points=self.mesh_vertices, cells=[("triangle", self.mesh_faces)])
        meshio.write(os.path.join(self.output_folder, filename), meshio_mesh)
        print(f"Mesh saved to {os.path.join(self.output_folder, filename)}")

    def tetrahedralize(self, input_file='heightmap.stl', output_file='heightmap_tet.msh'):
        """Perform tetrahedralization on an STL file and save the result."""
        input_path = os.path.join(self.output_folder, input_file)
        output_path = os.path.join(self.output_folder, output_file)

        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input STL file '{input_path}' not found.")

        imesh = meshio.read(input_path)
        V = imesh.points
        F = imesh.cells_dict.get("triangle", [])

        if len(F) == 0:
            raise ValueError(f"No triangular faces found in '{input_path}'.")

        # Create TetGen mesher object
        mesher = tg.TetGen(V, F)

        try:
            # Perform tetrahedralization
            Vtg, Ctg = mesher.tetrahedralize(order=1, mindihedral=5.0, minratio=1.0)
        except RuntimeError as e:
            raise RuntimeError("Failed to tetrahedralize. Ensure the surface is manifold and watertight.") from e

        # Create and save the tetrahedral mesh
        omesh = meshio.Mesh(points=Vtg, cells=[("tetra", Ctg)])
        meshio.write(output_path, omesh)
        self.tetrahedral_mesh = omesh
        print(f"Tetrahedral mesh saved to {output_path}")

    def compute_normals(self):
        """Compute the normals for the heightmap mesh."""
        dzdx = np.gradient(self.heightmap, axis=0)
        dzdy = np.gradient(self.heightmap, axis=1)
        normals = np.dstack((-dzdx, -dzdy, np.ones_like(self.heightmap)))
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        norm[norm == 0] = 1  # Prevent division by zero
        self.normals = normals / norm

    def save_normal_map_as_png(self, filename='normal_map.png'):
        """Save the normal map as a PNG image."""
        if self.normals is None:
            self.compute_normals()

        # Map normals from [-1, 1] to [0, 255]
        normals_normalized = (self.normals + 1.0) / 2.0
        normals_image = (normals_normalized * 255).astype(np.uint8)

        # Create the image
        img = Image.fromarray(normals_image, 'RGB')
        img = img.resize(self.normal_map_resolution, Image.NEAREST)

        # Apply scaling
        if self.normal_map_scale != 1:
            new_size = (img.width * self.normal_map_scale, img.height * self.normal_map_scale)
            img = img.resize(new_size, Image.NEAREST)

        # Save the image
        img.save(os.path.join(self.output_folder, filename))
        print(f"Normal map saved to {os.path.join(self.output_folder, filename)}")

    def save_displacement_map_as_png(self, filename='displacement_map.png'):
        """Save the displacement map as a grayscale PNG image."""
        if not self.generate_displacement_map:
            print("Displacement map generation is disabled.")
            return

        if self.heightmap is None:
            self.generate_heightmap()

        # Normalize heightmap to [0, 255]
        displacement_normalized = (self.heightmap * 255).astype(np.uint8)
        displacement_image = Image.fromarray(displacement_normalized, 'L')
        displacement_image = displacement_image.resize(self.displacement_map_resolution, Image.NEAREST)

        # Apply scaling
        if self.displacement_map_scale != 1:
            new_size = (displacement_image.width * self.displacement_map_scale, displacement_image.height * self.displacement_map_scale)
            displacement_image = displacement_image.resize(new_size, Image.NEAREST)

        # Save the image
        displacement_image.save(os.path.join(self.output_folder, filename))
        print(f"Displacement map saved to {os.path.join(self.output_folder, filename)}")

    def update_plotly_figure(self):
        """Update the Plotly figure with the current heightmap."""
        if self.heightmap is None:
            self.generate_heightmap()

        x = np.linspace(-1, 1, self.size_x)
        y = np.linspace(-1, 1, self.size_y)
        x_grid, y_grid = np.meshgrid(x, y)

        self.fig_widget.data = []  # Clear previous data
        self.fig_widget.add_surface(z=self.heightmap, x=x_grid, y=y_grid, colorscale='Viridis')

        self.fig_widget.update_layout(
            title='Heightmap',
            autosize=True,
            scene=dict(
                zaxis=dict(title='Height'),
                xaxis=dict(title='X Axis'),
                yaxis=dict(title='Y Axis')
            ),
            margin=dict(l=65, r=50, b=65, t=90)
        )

    def display_plotly_figure(self):
        """Display the Plotly figure."""
        display(self.fig_widget)

    def display_mesh_with_plotly(self):
        """Visualize the closed mesh using Plotly."""
        if self.mesh_vertices is None or self.mesh_faces is None:
            raise ValueError("Mesh not generated yet.")

        fig = go.Figure(data=[
            go.Mesh3d(
                x=self.mesh_vertices[:, 0],
                y=self.mesh_vertices[:, 1],
                z=self.mesh_vertices[:, 2],
                i=self.mesh_faces[:, 0],
                j=self.mesh_faces[:, 1],
                k=self.mesh_faces[:, 2],
                color='lightblue',
                opacity=0.5,
                name='Closed Mesh',
                showscale=True,
                colorbar=dict(title="Surface Height (µm)")
            )
        ])

        fig.update_layout(
            title="3D Closed Rough Surface Mesh",
            scene=dict(
                xaxis_title="Width (µm)",
                yaxis_title="Height (µm)",
                zaxis_title="Surface Height (µm)",
                camera=dict(
                    eye=dict(x=1.25, y=1.25, z=1.25)
                ),
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            height=700,
            width=800
        )

        fig.show()
