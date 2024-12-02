# test_simulation_state.py

import pytest
from simulation.core.states.state import SimulationState

def test_add_attribute():
    sim_state = SimulationState()
    sim_state.add_attribute('temperature', 300)
    assert sim_state.get_attribute('temperature') == 300

    # Test adding attribute with aliases
    sim_state.add_attribute('pressure', 101.3, aliases=['p', 'press'])
    assert sim_state.get_attribute('pressure') == 101.3
    assert sim_state.get_attribute('p') == 101.3
    assert sim_state.get_attribute('press') == 101.3

    # Test adding an existing attribute
    with pytest.raises(KeyError):
        sim_state.add_attribute('temperature', 350)

    # Test adding an attribute with an existing alias
    with pytest.raises(KeyError):
        sim_state.add_attribute('volume', 1.0, aliases=['p'])

def test_update_attribute():
    sim_state = SimulationState()
    sim_state.add_attribute('velocity', 10)

    # Update existing attribute
    sim_state.update_attribute('velocity', 20)
    assert sim_state.get_attribute('velocity') == 20

    # Update using alias
    sim_state.add_attribute('mass', 5, aliases=['m'])
    sim_state.update_attribute('m', 10)
    assert sim_state.get_attribute('mass') == 10

    # Attempt to update non-existing attribute without create flag
    with pytest.raises(KeyError):
        sim_state.update_attribute('density', 1000, create=False)

    # Update non-existing attribute with create flag
    sim_state.update_attribute('density', 1000)
    assert sim_state.get_attribute('density') == 1000

def test_get_attribute():
    sim_state = SimulationState()
    sim_state.add_attribute('energy', 50)

    # Get existing attribute
    assert sim_state.get_attribute('energy') == 50

    # Get non-existing attribute with default
    assert sim_state.get_attribute('power', default=0) == 0

    # Get non-existing attribute without default
    assert sim_state.get_attribute('power') is None

def test_remove_attribute():
    sim_state = SimulationState()
    sim_state.add_attribute('length', 100, aliases=['l', 'len'])
    sim_state.remove_attribute('length')

    # Ensure attribute and aliases are removed
    assert sim_state.get_attribute('length') is None
    assert sim_state.get_attribute('l') is None
    assert sim_state.get_attribute('len') is None

    # Attempt to remove non-existing attribute
    with pytest.raises(KeyError):
        sim_state.remove_attribute('width')

def test_list_attributes():
    sim_state = SimulationState()
    sim_state.add_attribute('height', 200, aliases=['h'])
    sim_state.add_attribute('weight', 75)

    attributes = sim_state.list_attributes()
    expected = {
        'height': {'value': 200, 'aliases': ['h']},
        'weight': {'value': 75, 'aliases': []}
    }
    assert attributes == expected

def test_to_dict_and_from_dict():
    sim_state = SimulationState()
    sim_state.add_attribute('speed', 60)
    sim_state.add_attribute('acceleration', 9.8)

    state_dict = sim_state.to_dict()
    expected_dict = {'speed': 60, 'acceleration': 9.8}
    assert state_dict == expected_dict

    # Create a new SimulationState and load from dict
    new_sim_state = SimulationState()
    new_sim_state.from_dict(state_dict)
    assert new_sim_state.get_attribute('speed') == 60
    assert new_sim_state.get_attribute('acceleration') == 9.8

def test_update_from():
    sim_state1 = SimulationState()
    sim_state1.add_attribute('force', 100)

    sim_state2 = SimulationState()
    sim_state2.add_attribute('force', 150)
    sim_state2.add_attribute('momentum', 200)

    sim_state1.update_from(sim_state2)
    assert sim_state1.get_attribute('force') == 150
    assert sim_state1.get_attribute('momentum') == 200

def test_check_required_attributes():
    sim_state = SimulationState()
    sim_state.add_attribute('density', 1000)
    sim_state.add_attribute('volume', 2)

    # Check when all required attributes are present
    sim_state.check_required_attributes(['density', 'volume'])

    # Check when some attributes are missing
    with pytest.raises(ValueError) as exc_info:
        sim_state.check_required_attributes(['density', 'mass'])
    assert "Missing required attributes: mass" in str(exc_info.value)

def test_alias_handling():
    sim_state = SimulationState()
    sim_state.add_attribute('position', [0, 0, 0], aliases=['pos', 'p'])
    assert sim_state.get_attribute('pos') == [0, 0, 0]
    assert sim_state.get_attribute('p') == [0, 0, 0]

    # Update using alias
    sim_state.update_attribute('p', [1, 1, 1])
    assert sim_state.get_attribute('position') == [1, 1, 1]

    # Remove using alias
    sim_state.remove_attribute('pos')
    assert sim_state.get_attribute('position') is None
    assert sim_state.get_attribute('p') is None

def test_conflicting_aliases():
    sim_state = SimulationState()
    sim_state.add_attribute('temperature', 300)

    # Attempt to add an attribute with an alias that conflicts with an existing attribute
    with pytest.raises(KeyError):
        sim_state.add_attribute('pressure', 101.3, aliases=['temperature'])

    # Attempt to add an alias that already exists
    sim_state.add_attribute('volume', 1.0, aliases=['v'])
    with pytest.raises(KeyError):
        sim_state.add_attribute('density', 1000, aliases=['v'])

def test_repr():
    sim_state = SimulationState()
    sim_state.add_attribute('frequency', 50, aliases=['freq'])
    expected_repr = "SimulationState({'frequency': {'value': 50, 'aliases': ['freq']}})"
    assert repr(sim_state) == expected_repr
