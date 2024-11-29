import copy
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SimulationState:
    """Represents the state of the simulation, encapsulating all necessary parameters and objects."""

    def __init__(self):
        """Initializes an empty simulation state.
        Attributes can be dynamically added or modified using the provided methods.
        """
        self.state = {}
        self.aliases = {}

    def add_attribute(self, key: str, value: Any, aliases: List[str] = None) -> None:
        """Adds a new attribute to the simulation state.

        Parameters
        ----------
        key : str
            The name of the attribute to add.
        value : Any
            The value of the attribute.
        aliases : List[str], optional
            A list of aliases for the attribute, by default None.

        Raises
        ------
        KeyError
            If the key or any of its aliases already exist.

        """
        if key in self.state or key in self.aliases:
            raise KeyError(
                f"Attribute '{key}' already exists. Use 'update_attribute' to modify it."
            )
        self.state[key] = value

        # Handle aliases
        if aliases:
            for alias in aliases:
                if alias in self.state or alias in self.aliases:
                    raise KeyError(
                        f"Alias '{alias}' conflicts with an existing attribute or alias."
                    )
                self.aliases[alias] = key

    def update_attribute(self, key: str, value: Any, create: bool = True) -> None:
        """Updates an existing attribute in the simulation state.

        Parameters
        ----------
        key : str
            The name or alias of the attribute to update.
        value : Any
            The new value of the attribute.

        Raises
        ------
        KeyError
            If the key does not exist.

        """
        primary_key = self.aliases.get(key, key)
        if primary_key not in self.state:
            if not create:
                raise KeyError(
                    f"Attribute '{key}' does not exist. Use 'add_attribute' to create it."
                )
            self.add_attribute(key, value)

        self.state[primary_key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Retrieves the value of an attribute from the simulation state.

        Parameters
        ----------
        key : str
            The name or alias of the attribute to retrieve.
        default : Any, optional
            The default value to return if the attribute does not exist, by default None.

        Returns
        -------
        Any
            The value of the attribute, or the default value if the attribute does not exist.

        """
        primary_key = self.aliases.get(key, key)
        return self.state.get(primary_key, default)

    def remove_attribute(self, key: str) -> None:
        """Removes an attribute and its aliases from the simulation state.

        Parameters
        ----------
        key : str
            The name or alias of the attribute to remove.

        Raises
        ------
        KeyError
            If the attribute does not exist.

        """
        primary_key = self.aliases.pop(key, key)
        if primary_key not in self.state:
            raise KeyError(f"Attribute '{key}' does not exist.")
        del self.state[primary_key]

        # Remove all associated aliases
        self.aliases = {
            alias: target
            for alias, target in self.aliases.items()
            if target != primary_key
        }

    def list_attributes(self) -> Dict[str, Dict[str, Any]]:
        """Lists all attributes with their values and aliases.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            A dictionary of attributes with their values and associated aliases.

        """
        result = {}
        for key, value in self.state.items():
            aliases = [alias for alias, target in self.aliases.items() if target == key]
            result[key] = {"value": value, "aliases": aliases}
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Converts the simulation state to a dictionary.

        Returns
        -------
        Dict[str, Any]
            The state as a dictionary.

        """
        return copy.deepcopy(self.state)

    def from_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the simulation state from a dictionary.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            The state as a dictionary.

        """
        self.state = copy.deepcopy(state_dict)
        self.aliases = {}

    def update_from(self, new_state: "SimulationState") -> None:
        """Updates the current simulation state with values from another state.

        Parameters
        ----------
        new_state : SimulationState
            The new state to update from.

        """
        self.state.update(new_state.to_dict())

    def check_required_attributes(self, required_attrs: List[str]) -> None:
        """Checks if required attributes are present in the simulation state.

        Parameters
        ----------
        required_attrs : List[str]
            List of required attribute names.

        Raises
        ------
        ValueError
            If any required attributes are missing.

        """
        missing_attrs = [
            attr for attr in required_attrs if self.get_attribute(attr) is None
        ]
        if missing_attrs:
            raise ValueError(f"Missing required attributes: {', '.join(missing_attrs)}")

    def __repr__(self) -> str:
        """Returns a string representation of the simulation state."""
        return f"SimulationState({self.list_attributes()})"
