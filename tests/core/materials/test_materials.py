import logging

import pytest

from simulation.core.materials.materials import (
    Material,
    add_custom_material,
    initial_material,
    materials,
)


class TestMaterial:
    def test_material_initialization(self):
        """Test basic Material class initialization."""
        material = Material(
            name="TEST_MATERIAL",
            young_modulus=100e9,
            poisson_ratio=0.3,
            density=7800,
            color=(255, 0, 0),
        )

        assert material.name == "TEST_MATERIAL"
        assert material.young_modulus == 100e9
        assert material.poisson_ratio == 0.3
        assert material.density == 7800
        assert material.color == (255, 0, 0)
        assert material.hardness is None

    def test_material_to_dict(self):
        """Test the to_dict method of Material class."""
        material = Material(
            name="TEST_MATERIAL",
            young_modulus=100e9,
            poisson_ratio=0.3,
            density=7800,
            color=(255, 0, 0),
            hardness=50.0,
        )

        material_dict = material.to_dict()

        assert material_dict == {
            "name": "TEST_MATERIAL",
            "young_modulus": 100e9,
            "poisson_ratio": 0.3,
            "density": 7800,
            "color": (255, 0, 0),
            "hardness": 50.0,
        }


class TestMaterialRegistry:
    def test_predefined_materials_exist(self):
        """Test that predefined materials are in the registry."""
        predefined_materials = [
            "WOOD",
            "STEEL",
            "ALUMINUM",
            "CONCRETE",
            "RUBBER",
            "COPPER",
            "GLASS",
            "TITANIUM",
            "BRASS",
            "PLA",
            "ABS",
            "PETG",
            "HYDROGEL",
            "POLYACRYLAMIDE",
        ]

        for material_name in predefined_materials:
            assert material_name in materials, f"{material_name} not found in materials registry"

            # Verify material properties
            material = materials[material_name]
            assert material.name == material_name
            assert material.young_modulus > 0
            assert 0 <= material.poisson_ratio <= 0.5
            assert material.density > 0
            assert len(material.color) == 3
            assert all(0 <= c <= 255 for c in material.color)

    def test_material_properties_range(self):
        """Test that material properties are within expected ranges."""
        for material_name, material in materials.items():
            # Young's Modulus: 1e4 to 1e12 Pa
            assert (
                1e4 <= material.young_modulus <= 1e12
            ), f"Invalid Young's Modulus for {material_name}"

            # Poisson's Ratio: 0 to 0.5
            assert (
                0 <= material.poisson_ratio <= 0.5
            ), f"Invalid Poisson's Ratio for {material_name}"

            # Density: 1 to 20000 kg/m^3
            assert 1 <= material.density <= 20000, f"Invalid Density for {material_name}"


class TestAddCustomMaterial:
    def test_add_custom_material_success(self):
        """Test adding a custom material."""
        add_custom_material(
            name="CUSTOM_MATERIAL",
            young_modulus=50e9,
            poisson_ratio=0.35,
            density=5000,
            color=(100, 150, 200),
        )

        assert "CUSTOM_MATERIAL" in materials
        material = materials["CUSTOM_MATERIAL"]
        assert material.name == "CUSTOM_MATERIAL"
        assert material.young_modulus == 50e9
        assert material.poisson_ratio == 0.35
        assert material.density == 5000
        assert material.color == (100, 150, 200)

    def test_add_custom_material_overwrite(self):
        """Test overwriting an existing material."""
        # First add
        add_custom_material(
            name="OVERWRITE_TEST",
            young_modulus=50e9,
            poisson_ratio=0.35,
            density=5000,
            color=(100, 150, 200),
        )

        # Overwrite
        add_custom_material(
            name="OVERWRITE_TEST",
            young_modulus=60e9,
            poisson_ratio=0.4,
            density=6000,
            color=(200, 100, 150),
            overwrite=True,
        )

        material = materials["OVERWRITE_TEST"]
        assert material.young_modulus == 60e9
        assert material.poisson_ratio == 0.4
        assert material.density == 6000
        assert material.color == (200, 100, 150)

    def test_add_custom_material_invalid_color(self):
        """Test adding a material with invalid color values."""
        with pytest.raises(ValueError, match="Invalid color"):
            add_custom_material(
                name="INVALID_COLOR_MATERIAL",
                young_modulus=50e9,
                poisson_ratio=0.35,
                density=5000,
                color=(256, 150, 200),  # Invalid color value
            )

    def test_add_custom_material_duplicate(self):
        """Test adding a duplicate material without overwrite."""
        with pytest.raises(ValueError, match="already exists"):
            add_custom_material(
                name="STEEL",  # Existing material
                young_modulus=50e9,
                poisson_ratio=0.35,
                density=5000,
                color=(100, 150, 200),
            )


class TestInitialMaterial:
    def test_initial_material_default(self, caplog):
        """Test initial_material with default parameters."""
        caplog.set_level(logging.INFO)

        material = initial_material()

        # Check logging
        assert any("Material 'DEFAULT' initialized" in record.message for record in caplog.records)

        # Check material properties
        assert material.name == "DEFAULT"
        assert material.young_modulus == 1e6
        assert material.poisson_ratio == 0.45
        assert material.density == 1000.0
        assert material.color == (128, 128, 128)

    def test_initial_material_existing(self):
        """Test initial_material with an existing material."""
        material = initial_material(name="STEEL")

        assert material.name == "STEEL"
        assert material.young_modulus == 210e9
        assert material.poisson_ratio == 0.3
        assert material.density == 7850
        assert material.color == (192, 192, 192)

    def test_initial_material_custom(self):
        """Test initial_material with custom parameters."""
        material = initial_material(
            name="CUSTOM_INITIAL_MATERIAL",
            young_modulus=75e9,
            poisson_ratio=0.33,
            density=5500,
            color=(50, 100, 150),
        )

        assert material.name == "CUSTOM_INITIAL_MATERIAL"
        assert material.young_modulus == 75e9
        assert material.poisson_ratio == 0.33
        assert material.density == 5500
        assert material.color == (50, 100, 150)


def test_material_csv_consistency():
    """Verify that the materials.csv file matches the materials dictionary."""
    import csv
    import os

    csv_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "simulation",
        "core",
        "materials",
        "materials.csv",
    )

    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        csv_materials = {row["name"]: row for row in reader}

    for material_name, material in materials.items():
        assert material_name in csv_materials, f"Material {material_name} not found in CSV"

        csv_material = csv_materials[material_name]
        assert float(csv_material["young_modulus"]) == material.young_modulus
        assert float(csv_material["poisson_ratio"]) == material.poisson_ratio
        assert float(csv_material["density"]) == material.density

        # Convert color tuple to string representation
        csv_color = tuple(map(int, csv_material["color"].strip("()").split(",")))
        assert csv_color == material.color
